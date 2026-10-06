# Разложение Sputnik → LibTorch, 2026-10-05

Последующая оптимизация ABI buffers, native captures и foreign handles описана
в [профиле обвязки 2026-10-06](native-call-overhead-profile.md).

Оптимизация найденного LOOKUP_CONST и свежие A/B/samples/MNIST опубликованы
в [отдельном отчёте](lookup-const-profile.md). Ниже сохранён исходный профиль
до этой оптимизации.

Основные расходы VM находятся в вызовах, работе с Value/frames, разрешении
межмодульных констант и создании Tensor. Минимальный C ABI call имеет тот же
порядок стоимости, что и обычный Sputnik call; zero-argument control дешевле,
поскольку native leaf не требует Sputnik bytecode frame. В native заметны
временные allocations вокруг ABI. Эта итерация добавляет диагностику;
runtime, алгоритмы MNIST и `attr … from` не изменялись.

## Протокол

[calls.s](../bench/ffi/calls.s) содержит 31 случай. Диагностические C++
entry points подключаются только для этого entry; production adapter и
notebook library их не содержат. Python используется опционально.

- CPU и настоящий MPS; LibTorch/PyTorch 2.14.0; четыре threads; MPS fallback=0.
- Три процесса на lane, вращающийся порядок VM/native/Python; прогрев каждого
  случая и пять внутренних повторов; медиана медиан процессов.
- Размер 25088 float32 элементов (число элементов MNIST batch 32), а также
  один элемент. Основные host cases: 300000 iterations; kernels: 3000.
  В контроле size=1: 10000 iterations обоих типов.
- Checksum вне таймера; MPS synchronization на границах kernel-циклов.
  Время означает стоимость операции в цикле, включая harness, на MPS —
  также работу очереди/device. Compilation/cache preparation, samples и
  другие нагрузки выполняются отдельно от измерений.
- Native: `cpp-bytecode-direct-v1`, full coverage, bytecode fallback=false.
  Все четыре матрицы использовали одинаковый executable SHA256:
  `dc47f501404fad473bd9f9b0f88950985ad9e22de21fb916de8dd4598794c225`.
- JSON сохраняет повторы, команды, версии, sources/executable/compiler/LibTorch
  hashes. CMake fingerprint и runner отклоняют изменённые build inputs;
  отдельно проверено отклонение устаревшего C++ fingerprint. Финальная
  короткая проверка всех rows VM/native/Python также прошла.

Sputnik harness использует `iterations.times` и captured block; Python —
`range` и lambda. Это языковые idioms с различным frame count: у `times`
есть callback итерации. Внутри lane схема одинакова, её цена видна в loop.
Дополнительные calls сравниваются относительно этой схемы.
Python numel тоже вызывается внутри lambda: lookup метода входит в таймер.
В старом MNIST atoms он был заранее привязан; абсолютные timings этих
двух драйверов нельзя напрямую смешивать. Custom Sputnik ABI controls не
имеют Python counterparts, поэтому соответствующие ячейки пустые.

## Host calls

CPU, размер 25088, наносекунды на итерацию с harness:

| Case | VM | native | Python |
| --- | ---: | ---: | ---: |
| loop + constant block | 329,79 | 43,13 | 14,81 |
| Sputnik free constant call | 531,41 | 65,26 | 25,42 |
| Sputnik method constant call | 515,56 | 68,91 | 30,25 |
| ABI constant, zero arguments | 476,19 | 83,33 | — |
| ABI identity, one Int | 537,49 | 102,05 | — |
| ABI identity, four Ints | 630,77 | 114,93 | — |
| ABI identity, tensor handle | 553,48 | 105,14 | — |
| checked handle identity | 554,11 | 111,82 | — |
| free ABI numel | 566,22 | 115,07 | — |
| core.numel | 506,39 | 101,25 | 45,67 |
| Tensor.numel | 716,18 | 133,72 | 45,39 |
| execution_mode | 748,92 | 89,81 | 47,52 |
| List[4] length | 557,15 | 112,46 | — |
| List[4] conversion + sum | 757,12 | 262,18 | — |
| List[32] conversion + sum | 1587,40 | 732,88 | — |

Host-only cases на MPS дают близкие значения. Они читают handles/metadata,
без kernel и копирования tensor data через ABI.

Контролируемые разности: сначала разность row medians внутри процесса,
затем медиана трёх разностей. Она может отличаться от разности итоговых
медиан. Это не независимые слагаемые и не цена инструкции C-call.

| Контраст, CPU, нс | VM | native |
| --- | ---: | ---: |
| Sputnik free constant call − loop | 201,62 | 22,49 |
| Sputnik method constant call − loop | 185,77 | 25,77 |
| ABI identity(Int) − loop | 207,69 | 59,28 |
| ABI constant − Sputnik constant | −54,05 | 18,03 |
| Ещё три scalar arguments | 97,59 | 12,76 |
| Int decoding/reboxing + boundary | 0,60 | 5,98 |
| Handle validation + boundary | 2,06 | 7,17 |
| Read numel + новый Int result | 10,68 | 3,22 |
| List[4] conversion относительно length | 199,69 | 148,63 |
| Ещё 28 list elements | 834,33 | 471,78 |

Разности в несколько нс близки к шуму: handle validation на MPS даёт
0,52 нс VM, в коротком CPU size=1 — отрицательную разность. Это не означает
бесплатную проверку. Устойчивы крупные эффекты: около 200 нс VM call поверх
harness, масштабирование по аргументам и element-by-element conversion.

`amt_bench_identity` возвращает существующий SputnikValue, без decoding,
нового result или kernel. Caller всё ещё делает dispatch, argument packing,
arena/handles и result extraction. Zero-argument control создаёт Int result.
Checked controls используют настоящие adapter helpers/exception boundary.

## Kernel и обвязка

Полное время, мкс на операцию; размер 25088:

| Устройство, case | VM | native | Python |
| --- | ---: | ---: | ---: |
| CPU, specialized relu → core | 2,391 | 1,999 | 1,824 |
| CPU, generic relu → core | 2,551 | 1,999 | 1,835 |
| CPU, generic relu → Tensor | 3,381 | 2,079 | 1,833 |
| CPU, public x.relu | 5,402 | 2,335 | 1,735 |
| MPS, specialized relu → core | 2,965 | 2,530 | 2,539 |
| MPS, generic relu → core | 3,075 | 2,536 | 2,405 |
| MPS, generic relu → Tensor | 3,896 | 2,596 | 2,403 |
| MPS, public x.relu | 5,610 | 2,698 | 2,389 |

Specialized entry принимает core; generic использует настоящий `amt_unary`
с core, operation string, options и mode. Следующий уровень создаёт Tensor;
public использует обычные helpers и execution mode. requires_grad=false, mode=0.

| Paired contrast, мкс | CPU VM | CPU native | MPS VM | MPS native |
| --- | ---: | ---: | ---: | ---: |
| generic − specialized | 0,167 | 0,015 | 0,081 | −0,008 |
| Tensor wrapper − core | 0,846 | 0,093 | 0,846 | 0,066 |
| public − explicit Tensor wrapper | 2,040 | 0,249 | 1,718 | 0,116 |

Tensor contrast включает вызов/создание instance и дополнительные instructions;
он не является отдельным allocator/init timer. Public contrast включает
dispatch, reads и forwarding helpers, а не только execution_mode. Небольшие
native kernel differences показывают шум, не преимущество generic.

`cpp_*` держит цикл внутри одного ABI call. Функции — в другом translation
unit без LTO; дизассемблер подтвердил `bl` и loop back-edge. Цена включает
C++ loop/call, а packing/results приходятся на весь batch. Другой harness
не позволяет вычитать `loop_scalar`. ExecutionGuard здесь вне цикла;
в adapter он выполняется на каждом call. Поэтому разность включает также
guard, TensorCore/handle lifetime и wrapping.

| C++ relu baseline, мкс | CPU 25088 | MPS 25088 | CPU 1 | MPS 1 |
| --- | ---: | ---: | ---: | ---: |
| VM-hosted C++ batch | 1,557 | 2,163 | 0,268 | 1,470 |
| native-hosted C++ batch | 1,589 | 2,151 | 0,268 | 1,479 |

Raw C++ constant/identity/numel в длинном цикле — около 0,8–0,9 нс/итерацию.
Для одного элемента public relu: CPU **3,476 / 0,786 / 0,411 мкс**,
MPS **5,063 / 2,063 / 1,722 мкс** (VM/native/Python). Размер вычисления
уменьшился, большой разрыв VM сохранился.

## Samples и конкретный линейный поиск

Пять отдельных CPU runs, macOS sample по 6 секунд, интервал 1 мс. Использован
активный поток; wait samples других threads не складываются с его CPU.

| Run | Active samples | Наблюдение |
| --- | ---: | --- |
| VM loop_scalar | 4605 | Runtime: step, Value copy/release, finish/recycle/acquire frame, cancellation |
| VM abi_identity_int | 4602 | 462 allocator samples (10,0%); 2 в теле пустого thunk |
| native abi_identity_int | 4666 | 1717 allocator samples (36,8%); 13 в теле thunk |
| VM public relu | 4551 | 2874 runtime (63,1%), 1634 LibTorch (35,9%), 43 adapter (0,9%) |
| native List[32] conversion | 4657 | 1643 allocator samples (35,3%); 86 exclusive sputnik_list_at samples |

Тело thunk и обработка ABI различаются: packing/arena живут в runtime.
Маленький вес thunk не определяет цену всей границы. `amt_execution_batch`
является предком почти всех samples, поскольку вызывает Sputnik callback;
его inclusive weight нельзя назвать FFI cost.

В native 54,9% samples минимального ABI control и 47,6% List[32] — `???`
в optimized executable. Allocator symbols доступны, но распределение остальных
samples по generated functions требует debug symbols. Данные не доказывают,
что все allocations принадлежат конкретной vector; исходник отдельно показывает
per-call arena/handles allocations в `sputnik_native_extension_invoke_inline`.

В VM public relu **533/4551 samples (11,7%)** входят в `Vm::lookup_constant`;
из них **338/4551 (7,4%)** — в `RuntimeErrorRegistry::has_error_namespace`.
В последней есть 100 exclusive samples самой функции, memcmp и создание/free
prefix string. Это вложенная часть lookup_constant, не дополнительная доля.

`lookup_constant` проверяет native bindings/error namespaces до lookup живого
module-level binding. `has_error_namespace` создаёт `name + "."` и перебирает
весь `error_ids_`, сравнивая prefixes. Обычные межмодульные function references
платят за этот поиск при отсутствии namespace. Лишняя работа локализована;
эффект её исправления на MNIST ещё не измерен.

## Следующая оптимизация

1. Индекс error namespace prefixes при регистрации ошибок вместо scan.
   Сохранить native binding precedence, новые регистрации, qualified prefixes
   и пустой namespace; затем отдельный A/B.
2. Оставшийся lookup_constant, block frame entry/exit и создание instance:
   уменьшать Value copies и повторную подготовку. Сохранить watch, task locals,
   cancellation и redefining semantics.
3. Native: symbols для optimized code, затем small buffers коротких ABI
   argument arrays/arenas с большим fallback и сохранением lifetime.
4. После каждого изменения — paired atomic A/B и MNIST CPU/MPS с несколькими
   batch sizes. Диагностика сама по себе не ускоряет MNIST и не доказывает
   устойчивого превосходства над Python.

## Дополнительный native correctness gap

Первоначальный initializer `long_text = "x" * 256` прошёл VM, но native
завершился `NativeCodeError: c604:102: native bailout` при full coverage=true.
Generated instruction использовал `native_numeric_fast_mul` для Str receiver.
Fixture теперь содержит 256-byte literal, вне таймера. Runtime gap здесь
не исправлялся; full coverage требует дополнения фактическим выполнением.

## Артефакты

- [CPU 25088, все rows](../build/bench/ffi/cpu-25088-verified/report.md)
- [MPS 25088, все rows](../build/bench/ffi/mps-25088/report.md)
- [CPU 1](../build/bench/ffi/cpu-1/report.md), [MPS 1](../build/bench/ffi/mps-1/report.md)
- [Sample family weights](../build/bench/ffi/sample-families.json)
- [Дизассемблер controls](../build/bench/ffi/raw-calls-disassembly.txt)
- Raw sample txt/summary.json/provenance.json: `build/bench/ffi/samples-<lane>-<case>/`.
- [Драйвер, описание и команды](../bench/ffi/README.md)

```sh
./tools/build bench/ffi/calls.s > build/ffi-calls-build.log 2>&1
/opt/homebrew/opt/pytorch/libexec/bin/python bench/ffi/run.py \
  --device cpu --output build/bench/ffi/new-cpu
/opt/homebrew/opt/pytorch/libexec/bin/python bench/ffi/run.py \
  --device mps --output build/bench/ffi/new-mps
```

Для size=1: `--size 1 --iterations 10000 --kernel-iterations 10000`.
Samples выполняет `bench/ffi/sample_calls.py`; macOS может требовать
разрешения attach к собственному дочернему процессу.
