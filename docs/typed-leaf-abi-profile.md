# Typed leaf ABI: numel, 2026-10-08

Первый typed entry реализован для `TensorCore.numel`: nullary foreign-handle
method → signed Int64. Это общий Sputnik ABI contract, без compiler cases
для Torch, selector `numel` или Tensor layout. Публичные Sputnik declarations
и методы остаются прежними.

Расширение дополнительно экспортирует
`amt_tensor_numel_sputnik_leaf_v1()`, возвращающий immutable descriptor с
version, size, signature, receiver tag и typed function pointer. VM находит
его через optional `dlsym`; native получает сильное определение пакета поверх
слабого null default. Отсутствие sidecar сохраняет обычный C ABI. Старый host
игнорирует новый export и продолжает использовать существующий thunk.

Host проверяет descriptor при загрузке; несовместимые version/size/signature,
пустой tag/function pointer и typed sidecar у blocking symbol отвергаются.
Leaf должен быть синхронным, не удерживать borrowed pointer, не вызывать
пользовательский код и не выпускать C++ exceptions за границу. Контракт:
[typed leaf ABI v1](../../sputnik-lang/docs/engineering/typed-leaf-abi-v1.md).

## Что исчезает на успешном пути

Runtime проверяет receiver tag/liveness и передаёт borrowed foreign pointer.
Adapter читает `TensorCore.tensor.numel()` и записывает raw Int64. Native и
VM создают собственный immediate Integer. `SputnikCtx`, argument/result arena,
opaque result handle, `sputnik_handle_ptr` и `sputnik_make_int` callbacks на
этом пути не нужны. VM получает обычный и typed entry из одной registry lookup.

На ошибке разрешён отдельный fault callback: текст потребляется синхронно и
ошибка проходит прежний rescuable путь. Ошибка без diagnostic становится
RuntimeError; reported fault имеет приоритет даже при ошибочном OK status.
Tag/tombstone guards предшествуют входу в typed function. Borrowing не добавляет
новой гарантии concurrent destroy поверх существующего handle contract.

`Tensor.numel` по-прежнему выполняет свой Sputnik wrapper. User method
resolution, invocation/frame, ownership промежуточных Values и dispatch по
code id остаются. Этот этап меняет ABI boundary; guarded direct calls и
interprocedural inlining остаются отдельной следующей работой.

## CPU A/B

Apple M4, macOS 26.6.2, LibTorch/PyTorch 2.14.0, Python 3.14.6 без JIT.
Size=1, 200000 host iterations, 10000 kernel iterations, пять внутренних
повторов и пять процессов на lane после отдельного warmup. VM до/после,
native до/после и Python выполнялись последовательно с вращением порядка.
Native имеет полное покрытие без bytecode fallback. Checksums проверяются
вне таймера; hashes исходников и executable проверены до и после серии.

| Case, нс/итерацию | Native до | Native после | VM до | VM после | Python |
| --- | ---: | ---: | ---: | ---: | ---: |
| loop + captured block | 23,18 | 23,60 | 245,32 | 246,23 | 14,47 |
| обычный Sputnik method | 45,00 | 45,76 | 385,31 | 387,53 | 30,44 |
| минимальный C ABI, Int identity | 47,98 | 48,52 | 373,67 | 378,20 | — |
| свободный C ABI numel | 61,32 | 61,02 | 401,54 | 401,66 | — |
| TensorCore.numel | 63,92 | 49,66 | 382,25 | 386,64 | 41,57 |
| Tensor.numel | 92,64 | 74,55 | 533,58 | 538,68 | 41,64 |
| specialized Core relu | 465,47 | 468,92 | 872,60 | 886,67 | 483,27 |
| public relu | 683,07 | 670,78 | 2350,49 | 2340,68 | 409,82 |

Числа в таблице — медианы process medians. Для A/B используются медианы
paired after/before: Core.numel **0,7769**, Tensor.numel **0,8064**,
оба выигрывают **5/5**, худшая пара 0,7894 и 0,8148 соответственно.
Native устранил около 14 нс на Core и 18 нс на публичном вызове; снижение
времени составляет примерно 22% и 19%. Native всё ещё проигрывает Python:
Core примерно в 1,2 раза, публичный wrapper примерно в 1,8 раза.

У обычного native method и ABI identity наблюдается небольшое увеличение:
paired ratios 1,0177 и 1,0103, каждый проигрывает 5/5, абсолютная разница
медиан меньше 1 нс. Общий extension helper теперь проверяет наличие leaf;
изменились registry layout и generated code layout. Серия не изолирует вклад
каждого из этих факторов, поэтому отсутствие регрессий не утверждается.
Public relu имеет ratio 0,9799 и выигрыши 3/5; его ускорение не следует
приписывать typed numel, поскольку сам relu использует обычный ABI.

У VM ускорения нет: paired ratios Core.numel 1,0168 и Tensor.numel 1,0235.
Один process round дал примерно 10% отклонение сразу в нескольких control
rows; остальные пары тоже не подтверждают полезного выигрыша numel.
Новый combined registry избегает второй lookup для leaf descriptor, но не
устраняет стоимость SEND/frame и captured-block harness.

Отдельный instrumented run проверил фактический маршрут VM:
`amt_tensor_numel_leaf` вызван 2000 раз — 1000 warmup и 1000 timed calls;
обычного `amt_tensor_numel` в профиле нет. Поэтому результат VM нельзя
объяснить неработающим выбором нового entry. Профиль с включённой
инструментацией не используется для timing comparisons.

VM обычный ABI уже использовал собственный `SputnikCtx` и прямые runtime
helpers; native ordinary ABI вызывает value readers/builders через
`direct_ops`. Удаление этой границы даёт разный эффект в двух реализациях.
Точное распределение оставшихся расходов VM этим A/B не определено;
следующий VM этап должен измерять dispatch/frame отдельно.

[Все 31 Sputnik / 15 Python cases и provenance](../build/bench/typed-leaf/cpu1-final/results.json),
[доказательство typed route в VM](../build/bench/typed-leaf/vm-route.log).

## MPS и большой numel

Настоящий MPS, `PYTORCH_ENABLE_MPS_FALLBACK=0`, size=25088; миллион iterations,
пять внутренних повторов, пять rotating processes и отдельный warmup.
`numel` читает metadata и не запускает GPU kernel. Этот размер также выводит
результат Python за пределы small-int cache.

| Tensor.numel, нс/итерацию | Native до | Native после | VM до | VM после | Python |
| --- | ---: | ---: | ---: | ---: | ---: |
| MPS, size=25088 | 92,63 | 74,62 | 531,50 | 530,68 | 45,97 |

Native paired ratio **0,8068**, выигрыши **5/5**, худшая пара 0,9338.
VM ratio **1,0092**, выигрыши 2/5: полезное ускорение опять не подтверждено.
Разрыв нового public native с Python составляет около 1,6 раза.
[MPS A/B и provenance](../build/bench/typed-leaf/mps25088/results.json).

## Проверки

- `sputnik_ext_tests`: Int64 min/max, non-nullary fallback, tag/tombstone guards,
  faults и отсутствующие diagnostics, version/size/signature validation,
  blocking rejection и removal stale leaf при обычной замене thunk.
- `native_call_buffers`: обязательный выбор typed entry в VM и full-native,
  failure/rescue/recovery, collected handle aliases и deterministic destroy.
- `native_language_idioms`: прежние native extensions без sidecar продолжают
  собираться и выполняться; VM unit suite проходит.
- Torch checks покрывают оба уровня Tensor/Core, empty/scalar/noncontiguous
  Tensor и `numel` после destroy; CPU и настоящий MPS проходят в VM и full-native,
  включая checkpoint `map_location` CUDA→MPS. Недостающий checkpoint fixture
  создан штатным C++ generator, без Python test/build dependency.

[Language ABI tests](../../sputnik-lang/build/typed-leaf-ext-tests.log),
[VM suite](../../sputnik-lang/build/typed-leaf-vm-tests.log),
[native fixtures](../../sputnik-lang/build/typed-leaf-fixtures.log),
[CPU native](../build/bench/typed-leaf/torch-cpu-native.log),
[CPU VM](../build/bench/typed-leaf/torch-cpu-vm.log),
[MPS native](../build/bench/typed-leaf/torch-mps-native.log),
[MPS VM](../build/bench/typed-leaf/torch-mps-vm.log).

Исходники ABI/runtime, snapshots generated C++, executable hashes и ссылки
на verification logs сохранены в
[provenance](../build/bench/typed-leaf/provenance.json).

## Следующий резерв

Native Core уже занимает около 50 нс; публичный wrapper добавляет ещё около
25 нс в том же harness. Следующий compiler этап — guarded direct calls и
inlining простых forwarding methods с сохранением overrides, trace/rescue и
ownership. Расширять набор typed signatures стоит по отдельным measured cases,
а не переводить сложные callback/blocking calls на leaf contract.

VM сначала требует отдельного измерения nullary SEND/invocation/frame и
guarded forwarding path. Подключение того же typed function само по себе
не обеспечило цель «VM на уровне Python».

Numel используется в setup MNIST и для размера labels в метриках каждого
train/eval batch (`examples/mnist/training.s`). Отдельный numel microbenchmark
измеряет ABI overhead и не является доказательством ускорения тренировки.
