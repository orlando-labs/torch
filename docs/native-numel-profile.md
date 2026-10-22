# Numel и направление native, 2026-10-08

`Tensor.numel` медленный из-за стоимости Sputnik-вызовов вокруг очень дешёвого
чтения LibTorch. Для обычного dense Tensor `at::TensorBase::numel()` вызывает
`TensorImpl::numel()`: проверяет CustomSizes policy и возвращает сохранённое
поле `numel_`. GPU kernel, синхронизация и пересчёт произведения размеров здесь
не нужны. Адаптер использует отдельный `TENSOR_QUERY`, без ExecutionGuard.

## Что именно измерялось

Apple M4, macOS 26.6.2, LibTorch/PyTorch 2.14.0, Python 3.14.6 без JIT.
Все builds, benchmarks и samples последовательные. FFI cases имеют одинаковый
loop/captured-block harness; C++ loop — другой нижний контроль с реальным
вызовом из отдельного translation unit без LTO. Разности обычных rows являются
контролируемыми контрастами; их нельзя складывать как независимые цены стадий.
Checksums проверяются вне таймера. Native имеет полное покрытие без VM fallback.

Свежий baseline, CPU, size=1, три процесса, пять внутренних повторов:

| Case, нс/итерацию | VM | Native | Python |
| --- | ---: | ---: | ---: |
| loop + captured block | 245,75 | 28,64 | 15,34 |
| обычный Sputnik method | 379,82 | 55,07 | 29,78 |
| минимальный C ABI, Int identity | 372,63 | 52,44 | — |
| C ABI, проверка tensor handle | 413,76 | 61,97 | — |
| C ABI, свободная функция numel | 397,95 | 64,21 | — |
| TensorCore.numel | 379,83 | 71,06 | 41,62 |
| Tensor.numel | 529,81 | 106,00 | 41,99 |
| C++ numel внутри одного ABI call | 0,80 | 0,78 | — |

[Полные результаты](../build/bench/native-stage5/baseline-cpu1/results.json).
Python вызывает `x.numel()` из lambda, включая поиск метода; Sputnik вызывает
`x.numel` из captured block. Публичный Python Tensor уже является PyTorch object,
а Sputnik Tensor добавляет свой метод над отдельным foreign TensorCore.

## Доказанные расходы native

Путь публичного вызова:

1. Captured block читает Tensor из capture cell.
2. Cached user send проверяет class/method flags, bare-call syntax и сигнатуру.
3. Создаются method invocation и frame; `LoadSelf` копирует self в регистр.
4. `LoadIvar` читает `@core`, второй user send вызывает TensorCore.numel.
5. C ABI arena удерживает self; runtime создаёт opaque context и вызывает thunk.
6. `sputnik_handle_ptr` проверяет тип, tag и lifetime; LibTorch читает numel.
7. `sputnik_make_int` возвращает immediate Integer через arena, затем scopes
   освобождают временные ownership references.

В `numel` нет создания нового Tensor/TensorCore. Sputnik Integer хранится
непосредственно в Value; выделение heap для результата не требуется.
Проверка handle сравнивает существующий tag с C string и не создаёт новую
`std::string`. Объяснять этот конкретный case Tensor allocations неверно.

Отдельный symbolized optimized sample: шесть секунд, интервал 1 мс,
4499 samples активного FFI worker. Startup main thread и ожидание остальных
threads исключены. Exclusive leaves, а не суммы inclusive stack weights:

| Семейство | Samples | Доля |
| --- | ---: | ---: |
| NativeFrame constructor/destructor | 680 | 15,1% |
| User send и его invocation helper | 689 | 15,3% |
| Dispatch по code id | 340 | 7,6% |
| Signature/shaping/auto-assign queries | 297 | 6,6% |
| Выключенный BoundaryProfileScope ctor/dtor | 88 | 2,0% |
| Allocator leaves в активном worker | 0 | 0% |

Это распределение sampled CPU всего case с harness; проценты не являются
прогнозом ускорения. [Sample и подсчёт](../build/bench/native-stage5/sample-before-numel/summary.json),
[полный стек](../build/bench/native-stage5/sample-before-numel/native.sample.txt).

## Общие исправления compiler/runtime

### Не копировать receiver через условное выражение

`current_closure == nullptr ? NativeValue::nullv() : current_closure->self`
создавало временный владеющий Value даже при передаче в `const NativeValue &`.
Новый `native_closure_self` возвращает ссылку на self или неизменяемый null.
Arena по-прежнему получает свою владеющую копию перед входом в C ABI.

### Frame заимствует self/block у живой invocation

Frame повторно удерживал immutable self/block, которыми уже владеет closure
Value вызывающего кода или stack invocation. Теперь он хранит указатели на эти
слоты. `LoadSelf`, `LoadBlock`, handler invocation и escaping closure всё ещё
получают обычные владеющие Values. Seeded handler заимствует слоты живого parent
frame. Регистры, captures и nonlocal-return target сохраняют прежнее ownership.

### Method cache содержит свойства сигнатуры

Даже на cache hit выполнялись отдельные switches для descriptor, необходимости
shaping, bare-call eligibility и auto-assigns. Неизменяемая запись метода теперь
содержит parameter count и эти флаги; горячий путь использует указатель на неё.
Одна atomic запись продолжает публиковать class key + method-table index, поэтому
новые поля не создают гонку раздельной публикации. Сигнатуры берутся из той же
canonical записи по code id, что и прежний descriptor, включая aliases.
Keywords/defaults/rest/clauses продолжают использовать общий shaping path.

Изолированные A/B используют одинаковые runtime archive, extension objects и
flags, пять процессов с вращением lanes. Изменяется только generated C++:
[frame/metadata ablations](../build/bench/native-stage5/frame-metadata-cpu1/results.json),
[receiver-copy и forwarding probe](../build/bench/native-stage5/probes-cpu1/results.json).
Forwarding probe вручную убрал frame только у numel-wrapper и получил
105,36 → 86,95 нс. Этот probe — оценка потенциала, не production optimization
и не специальный compiler case для Torch.

## Финальный FFI A/B

CPU, size=1, пять вращающихся процессов старого native, нового native, VM и
Python, 500000 host iterations, 10000 kernel iterations, пять timed повторов.
Обе native сборки содержат одинаковые исходники адаптера.

| Case, нс/итерацию | Native до | Native после | VM | Python |
| --- | ---: | ---: | ---: | ---: |
| loop + captured block | 28,76 | 24,17 | 245,31 | 15,15 |
| обычный method | 55,32 | 45,13 | 383,89 | 30,56 |
| свободный C ABI numel | 63,72 | 60,87 | 398,29 | — |
| TensorCore.numel | 71,35 | 63,92 | 378,48 | 41,94 |
| Tensor.numel | 106,08 | 92,11 | 533,57 | 41,53 |
| public relu | 691,76 | 678,69 | 2327,78 | 408,82 |

Медиана paired after/before: loop 0,8402, method 0,8158, Core.numel 0,8886,
Tensor.numel 0,8694; каждый выигрывает 5/5. Public relu 0,9799, 4/5. VM код
на этом этапе не менялся. Новый native numel всё ещё в 2,2 раза дороже Python;
исправления не устранили весь разрыв.
[Полная таблица и provenance](../build/bench/native-stage5/final-cpu1/results.json).

### Влияние Python small-int cache

Для sizes 1 и 256 два живых результата Python numel являются одним объектом;
для 257 и 25088 — разными. В установленной Python GIL включён, free-threading=0.
Измерения трех процессов, миллион iterations и пять повторов:

| Size | Native после, нс | Python, нс |
| --- | ---: | ---: |
| 256 | 91,88 | 41,59 |
| 257 | 92,69 | 45,27 |
| 25088 | 92,32 | 45,61 |

На 25088 VM 531,98 нс, старый native 106,67 нс. Переход за границу кэша
заметен в Python, но объясняет примерно 4 нс, а не двукратный разрыв. Sputnik
Integer остаётся immediate независимо от этих sizes. Native использует
atomic intrusive refcounts для managed values; GIL-сборка Python и Sputnik
имеют разные требования к синхронизации ownership. Это основание уменьшать
число ненужных leases, а не отключать atomics без доказательства confinement.

[256](../build/bench/native-stage5/numel-256/results.json),
[257](../build/bench/native-stage5/numel-257/results.json),
[25088 с VM и старым native](../build/bench/native-stage5/numel-25088/results.json).

## Проверка на MNIST

SmallResNet, 19706 parameters, train/test=2048/512, три epochs, batch=32,
seed=7, четыре intraop threads, десять interop threads. Sputnik использует
execution scope `run`, Python — packed optimizer. Для каждого backend проведён
один warmup и пять измерений каждой lane в отдельных процессах с вращением
порядка старого native, нового native, VM и Python. MPS настоящий, fallback=0.
Loss совпадает с tolerance 1e-5, accuracy точно; checkpoint roundtrip проходит.
Executables, source inputs, dataset и LibTorch library проверены по hashes.

В текущем Sputnik MNIST `numel` используется для подсчёта параметров модели
в setup (`bench/mnist/benchmark.s:39`) и для размера labels в метриках каждого
batch (`examples/mnist/training.s`). Первоначальная формулировка «только setup»
была ошибочной: поиск в `src/` не охватил `examples/`. Эти вызовы намного реже
kernel calls; numel служит probe стоимости вызова, но его вклад в обучение
нужно измерять, а не исключать из анализа по расположению setup-вызова.
Общие frame/metadata изменения затрагивают и другие методы, но их вклад в
train+eval нужно измерять отдельно.

Медианы суммы train+eval за три epochs, без setup/checkpoint:

| Backend | Native до, с | Native после, с | VM, с | Python, с |
| --- | ---: | ---: | ---: | ---: |
| CPU | 2,85068 | 2,83685 | 2,92211 | 2,86262 |
| MPS | 0,76424 | 0,76570 | 0,86806 | 0,78264 |

Для сравнения используются медианы **paired ratios**, а не отношения двух
медиан из таблицы:

| Backend | Native после / до | Native после / Python | VM / Python |
| --- | ---: | ---: | ---: |
| CPU | 0,99972; выигрыши 3/5 | 0,99182; выигрыши 4/5 | 1,01934; выигрыши 0/5 |
| MPS | 0,99794; выигрыши 3/5 | 0,98122; выигрыши 5/5 | 1,11244; выигрыши 0/5 |

Устойчивого ускорения всей тренировки от текущих изменений не подтвердилось:
измеряемый выигрыш маленьких host calls размывается kernel work. На MPS native
опережает Python примерно на 1,9% во всех пяти парах; на CPU — примерно на 0,8%
по paired median, но один из пяти runs проигрывает на 1,7%. Запас native над
Python остаётся небольшим. VM на CPU отстаёт примерно на 1,9%, на MPS — на 11,2%.
Цель «native уверенно быстрее, VM на уровне» этим этапом ещё не достигнута.

Первый отдельный build/validation run без warmup также сохранён: native
2,97761 с против Python 2,85687 с, то есть native проиграл на 4,2%. Его первый
train epoch занимал 1,03732 с, последующие — около 0,88174 с. Этот одиночный run
не заменяет вращающуюся серию и не исключён из сохранённых данных.

[CPU A/B](../build/bench/native-stage5/mnist-cpu32/results.json),
[MPS A/B](../build/bench/native-stage5/mnist-mps32/results.json),
[первый build/validation run](../build/bench/native-stage5/mnist-build-control/results.json).
Batch=128 на этом этапе заново не измерялся; предыдущая матрица приведена в
[native dispatch](native-dispatch-profile.md).

## Дополнительная проблема native GC

Существующий bounded-RSS fixture с 300000 cyclic allocations сначала провалился:
peak RSS 706281472 bytes. Старый compiler до нынешних изменений воспроизвёл
то же поведение: 705429504 bytes. Frame borrowing не вызвало этот дефект.

Native игнорировал `Opcode::Safepoint`, который emitter помещает в while/until
loop header. Pending cycles публиковались лишь на внешней task boundary.
Теперь этот opcode вызывает существующий `native_cycle_checkpoint_if_due`:
локальный allocation threshold включает collection/cooperation только при
необходимости; application locks сохраняют прежние правила отложенной сборки.
Тот же fixture после исправления прошёл: peak RSS 90996736 bytes, ниже лимита
128 MiB, результат 300000. Это проверка явных while/until safepoints, а не
доказательство всех возможных путей сборки циклов.

[Исправленный run](../../sputnik-lang/build/native-stage5/cycle-after.log),
[fixture](../../sputnik-lang/tests/fixtures/native_cycle_lifetime_core/main.s).

## Приоритет следующих изменений

1. **Guarded direct calls и inline маленьких методов.** Сохранять runtime guard
   по class/method record для override/mixin/polymorphism, после него вызывать
   известную native body напрямую. Убирать общий code-id dispatch и frame
   простых forwarding bodies. Forwarding probe показывает измеримый резерв;
   общая реализация должна сохранить source traces, ошибки и fallback receivers.
2. **Ownership/liveness в native IR.** Не делать `LoadSelf` → owning register →
   `LoadIvar`, когда промежуточный self нигде больше не нужен. Различать borrowed
   argument views и escaping values с доказанной lifetime, включая callbacks,
   уничтожение receiver и concurrent access. Blanket удаление retain/release
   небезопасно; оно не заменяет dataflow analysis.
3. **Опциональный typed native ABI для дешёвых leaf calls.** Scalar return и
   проверенный receiver передавать без arena/callback цепочки make_int. Оставить
   совместимый opaque ABI для общих расширений. VM может использовать тот же
   entry point, добавляя своё boxing. Требуются явные type/effect descriptors,
   проверки tag/lifetime, fault mapping и запрет неописанных callbacks/suspension.
   Первый общий handle→Int64 entry для numel уже реализован:
   [typed leaf ABI и A/B](typed-leaf-abi-profile.md). Он уменьшает время native
   Core/public numel примерно на 22%/19%, но не даёт ускорения VM; другие
   signatures и лёгкие VM frames остаются следующей работой.
4. **Сократить allocations tensor результатов.** В отличие от numel, relu и
   другие kernels создают TensorCore, runtime foreign lifetime/metadata,
   native foreign wrapper и Sputnik Tensor instance. У Python PyTorch object
   другой layout. Проверять embedding/coallocation lifetime metadata, interned
   type/tag IDs и lowering простых wrapper constructors. Сохранить escaping
   `_core` aliases, `destroy!` tombstones, readonly и VM bridge semantics.
   Предыдущие allocation samples и handle ablations:
   [native call overhead](native-call-overhead-profile.md).
5. **Сужать ABI plumbing по эффектам.** Разделить дешёвый metadata boundary и
   произвольный callback/blocking call; выключенный profiler не должен вызывать
   out-of-line ctor/dtor. LTO/PGO проверять после определения прямых call targets:
   один `-O3` не устраняет архитектурные динамические переходы.
6. **VM: лёгкие leaf frames и fused instructions.** Cached nullary metadata и
   proven forwarding bodies должны обходить лишнее shaping/frame setup, сохраняя
   trace/rescue/ensure/cancellation и method-version guards. Для сравнения с
   Python нужны отдельные бюджеты host overhead и реального training workload.

Native — основной приоритет: interprocedural specialization, borrowed values
и typed leaf boundary дают больший потенциал, чем ускорение интерпретации
инструкций. VM должна получать общие ABI улучшения и собственный лёгкий call path.
Numel — чувствительный probe этих расходов; его microbenchmark не доказывает
ускорение всей тренировки MNIST.

## Проверки корректности

`native_call_buffers` проверяет dropping caller references внутри callback,
escaping self/block, rescue/ensure, nonlocal return, polymorphic signature
shaping, defaults, list arenas и native handle lifetime в VM/native.
Native-only `native_dispatch_cache` добавляет восемь tasks × 50000 calls одного
site между required и optional сигнатурами, поверх прежнего method/property
race test. Native language idioms, native/ext/VM unit suites и bounded-RSS test
проходят. Python используется только как опциональный benchmark/reference;
новой Python зависимости у torch нет.
