# Обвязка VM/native вызовов, 2026-10-06

В native действительно была существенная лишняя работа вокруг вызова: временные
ABI vectors выделяли память на каждом коротком вызове, а вызов функции/блока
копировал lexical captures и делал atomic retain/release каждого captured cell.
Это расходы runtime; native здесь выполняет C++, но не получает автоматически
стоимость обычного C++ function call.

Предыдущая [оптимизация LOOKUP_CONST](lookup-const-profile.md) уже присутствует
в обеих сторонах текущего A/B. Алгоритмы MNIST, `attr … from`, модель, optimizer,
число kernels и execution scope сохранены.

## Изменения

- VM и generated native используют общий `NativeCallValueArena`: 16 значений
  внутри объекта вызова, затем vector overflow. Конструируются только занятые
  slots; ABI handles остаются индексами, а первые slots не перемещаются при росте.
- ABI argument handles используют contiguous buffer с восемью inline slots и
  heap fallback для больших calls. Короткие calls обходятся без этих allocations.
- Native synchronous call читает исходное окружение closure. Когда требуется
  заменить incoming block или сформировать optional/keyword arguments, временное
  окружение заимствует captures. Copy/move и escaping closures материализуют
  ownership; lexical mutation и lifetime captured cells сохраняются.
- C++ adapter `integers` один раз читает List length и заранее резервирует
  выходной vector. Sputnik-алгоритмы не переносились через границу в C++.
- Native foreign handle хранит shared lifetime/tombstone напрямую, без
  дополнительного runtime Value tail box. При реальном экспорте через runtime
  bridge Value создаётся с тем же shared handle: aliases сохраняют tombstone,
  owned destroy и collected reclaim.
- Native type tag сравнивается с constant `string_view`, без повторного
  hex decode. VM/native перемещают уже созданный tag в handle вместо копирования.

## Изолированный native A/B

CPU, один float32 элемент, пять процессов с вращением before/arena/captures/Python,
300000 host iterations, 10000 kernel iterations; прогрев и пять timed повторов.
Время ниже включает loop/captured-block harness. Общие runtime и extension objects
идентичны и собраны с production `-O2 -DNDEBUG`; изменяется только generated C++.
Изменение adapter List conversion намеренно исключено из этого A/B.

| Case, нс/итерацию | До | Inline arena | Arena + borrowed captures | Python |
| --- | ---: | ---: | ---: | ---: |
| loop_scalar | 43,23 | 42,96 | 28,23 | 14,73 |
| Sputnik free constant | 65,19 | 65,04 | 44,26 | 26,08 |
| Sputnik method constant | 69,48 | 69,90 | 54,82 | 30,42 |
| ABI identity Int | 101,02 | 68,58 | 50,07 | — |
| ABI identity 4 Ints | 114,02 | 79,37 | 59,37 | — |
| ABI identity handle | 104,29 | 74,88 | 56,66 | — |
| ABI String[256] length | 134,14 | 103,56 | 79,76 | — |
| ABI List[4] conversion/sum | 260,66 | 181,99 | 163,14 | — |
| ABI List[32] conversion/sum | 724,31 | 715,18 | 694,67 | — |
| specialized relu → core | 532,81 | 506,70 | 476,70 | 474,29 |
| public relu | 784,04 | 753,80 | 700,06 | 413,59 |
| raw C++ relu | 270,38 | 270,93 | 269,01 | — |

ABI identity ускорился в 2,02 раза, empty block harness — в 1,53 раза.
Контроль raw C++ kernel практически неизменен: выигрыш локализован в обвязке.
Небольшие различия kernels близки к шуму. Custom ABI rows не имеют Python
counterpart; `—` не заменяется ctypes или иной несопоставимой границей.

[Полная таблица](../build/bench/native-overhead/isolated-verified-ab/report.md),
[повторы и hashes](../build/bench/native-overhead/isolated-verified-ab/results.json).
Ранние `isolated-ab` и `isolated-borrow-ab` содержат неверную атрибуцию variants:
compile script повторно выбирал arena source. Их calls/captures lanes отброшены;
здесь используется только исправленный `isolated-verified-ab` с явными source paths.

## Второй A/B: упаковка foreign handle

После первой итерации найден ещё один allocation в generated native:
`NativeForeignHandle` хранил runtime Value с отдельным tail box, хотя сам shared
`RuntimeForeignHandle` уже управляет lifetime. Кроме того, class-index resolution
декодировал constant hex tag при каждом Tensor result, а tag копировался в handle.
Новый вариант устраняет box и переносит/сравнивает tag без изменения lifetime.

Тот же size=1 protocol, пять вращающихся процессов; общие current runtime,
extension objects и flags. Меняется только generated handle plumbing.

| Case, нс | Arena/captures | + literal/moved tag | + direct shared handle | Python |
| --- | ---: | ---: | ---: | ---: |
| specialized relu → core | 475,76 | 475,18 | 448,25 | 478,20 |
| generic relu → core | 503,02 | 500,27 | 472,07 | 478,46 |
| generic relu → Tensor | 586,93 | 584,35 | 563,36 | 478,29 |
| public relu | 698,14 | 691,02 | 671,77 | 408,12 |
| raw C++ relu | 273,76 | 272,59 | 269,33 | — |

Сам tag даёт небольшой эффект; удаление лишнего box заметнее. Generic core
достигает порядка Python, но public Tensor wrapper всё ещё дороже.
Контроль C++ и остальные unchanged host rows ограничивают трактовку малых
разностей. Все 31 Sputnik и 15 Python checksums проверены.

[Полная таблица](../build/bench/native-overhead/isolated-handles-ab/report.md),
[sources/executable/object/archive hashes и повторы](../build/bench/native-overhead/isolated-handles-ab/results.json).

## VM: атомарное сравнение первой итерации

Пять процессов старой/новой VM, native и Python в общем вращающемся порядке;
size=1, 300000 host/10000 kernel iterations, пять timed повторов. Две VM читают
общий current manifest/adapter: native captures не относятся к VM, List reserve
общий для обеих. Это сравнение до финального перемещения VM tag.

| Case, нс | VM до | VM arena | Native arena/captures | Python |
| --- | ---: | ---: | ---: | ---: |
| loop_scalar | 332,73 | 333,09 | 28,03 | 15,00 |
| ABI identity Int | 533,67 | 503,52 | 51,19 | — |
| ABI List[4] conversion/sum | 696,75 | 607,97 | 113,44 | — |
| specialized relu → core | 1008,40 | 955,58 | 470,22 | 483,78 |
| generic relu → Tensor | 2180,87 | 2117,21 | 576,57 | 486,07 |
| public relu | 3160,50 | 3091,29 | 696,65 | 408,77 |

Контроль loop не меняется; ABI identity выигрывает около 30 нс, но публичная
обвязка остаётся значительно дороже Python. Arena исправляет одну часть вызова;
она не устраняет VM dispatch, frames и создание Sputnik instance.
[Все случаи и повторы](../build/bench/native-overhead/ffi-cpu-1/results.json).

## Samples с символами

Отдельные runs по шесть секунд с интервалом 1 мс; optimized native с
`-gline-tables-only` и сохранёнными symbols. Анализируется активный FFI worker;
ожидание других threads исключено. After использует финальный adapter с reserve.

| Run | Активных samples до / после | Allocator leaf samples до / после |
| --- | ---: | ---: |
| ABI identity Int | 3196 / 4533 | 36,7% / 0% |
| ABI List[32] conversion/sum | 5170 / 5164 | 34,5% / 26,3% |
| public relu | 5128 / 5114 | 26,4% / 22,2% |

В identity до изменения 594 allocator samples относятся к
`sputnik_native_extension_invoke_inline`, 287 — к уничтожению extension state,
256 — к vector reserve. Copy constructor/destructor `NativeClosure` имеют ещё
470 exclusive samples (14,7%); это отдельные листья, а не сумма inclusive weights.
После изменения copy constructor исчезает из горячего дерева identity, а
основная работа перемещается в frame setup/teardown и dispatch.

В List[32] allocation samples под `integers` уменьшаются с 755 до 91.
Остаются arena overflow, `sputnik_list_at` и повторное `native_materialize_list`.
В relu allocations внутри LibTorch и `amt_tensor_free` сохраняются; видно также
`sputnik_native_ext_make_handle`, runtime Value и native instance lifetime.
Проценты описывают sampled active CPU и не являются процентом ускорения.
Allocator определяется по библиотеке `libsystem_malloc` и leaf `operator new/delete`.
Поиск подстроки `free` в имени функции ошибочно включал бы `sputnik_ext_invoke_direct_free`;
исправленный подсчёт исключает такие обычные ABI functions.

`mach_absolute_time` в старом List sample находится под allocator `_xzm_free`;
он не доказывает включённый BoundaryProfile. `AMT_BOUNDARY_PROFILE` удалён из
environment, а run logs не содержат AMT profiling records.

[Подсчёт семейств](../build/bench/native-overhead/sample-families.json),
[identity до](../build/bench/native-overhead/sample-before-abi_identity_int/native.sample.txt),
[после](../build/bench/native-overhead/sample-after-abi_identity_int/native.sample.txt),
[relu после](../build/bench/native-overhead/sample-after-relu_public/native.sample.txt).

Финальный symbolized build из production generated source также снят отдельно:
[relu после direct-handle](../build/bench/native-overhead/sample-stage2-relu_public/native.sample.txt).
4551 active samples, 1075 allocator leaves (23,6%). Allocations под runtime
`Value::foreign_handle`/`Value::~Value` в горячем relu исчезли; остаются LibTorch,
TensorCore, shared RuntimeForeignHandle и native instance/header lifetime.
Суммарная allocator fraction не обязана монотонно падать после ускорения:
изменяются знаменатель и распределение оставшейся работы. Эффект direct-handle
оценён отдельным A/B выше, а не разностью этих процентов.

## Почему обычные native benchmarks быстрее Python

В обычном арифметическом native benchmark scalar registers и operations могут
быть unboxed и исполняться напрямую. Tensor путь проходит dynamic Values,
Sputnik methods/closures, argument shaping, стабильный C ABI и foreign lifetime.
Поэтому одинаковый backend не означает одинаковую цену вызова. Это подтверждают
раздельные controls и samples; вывод о полном MNIST требует отдельного замера.

Дополнительно проверен сам `ExecutionGuard`: в небольшом C++ probe mode=0
старые `InferenceMode`/`AutoGradMode` дают около 25 нс на пустую итерацию.
Это слишком мало, чтобы объяснить весь разрыв public relu. Условный prototype
не вошёл в production: он не проверен на всех дополнительных AutogradState bits,
которые нормализует `InferenceMode`. Семантика guard сохранена.

## MNIST: первая итерация (arena/captures/List reserve)

Одинаковые SmallResNet и packed optimizer, 19706 parameters, три epochs,
batch=32, seed=7, train=2048/test=512, четыре threads, execution-scope=run.
По одному полному process warmup и пять процессов на lane с вращением порядка.
PyTorch/LibTorch 2.14.0; настоящий MPS, CPU fallback=0. Численные метрики всех
epochs и checkpoint совпадают с Python в пределах loss atol=1e-5, accuracy точно.
Таймер ниже — train + evaluation; data/setup/save/restore исключены.

| Устройство, секунды (медиана) | Native до | Native после | VM до | VM после | Python |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU | 3,02736 | 3,00955 | 3,10601 | 3,12298 | 3,06762 |
| MPS | 0,86641 | 0,85058 | 0,96590 | 0,98382 | 0,87221 |

VM до/после использует общий текущий adapter, поэтому изолирует runtime;
сохранённый native содержит исходный adapter. Native full coverage: 1404/1404
code objects, fallback=false, VM-independent=true. Hash нового executable:
`ff24f092a41dc94086482b4ab7371a0d2f2b95150eeb080f08f1694f49e0a60a`.

Медиана paired native после/до: CPU 0,99515, MPS 0,97933. Native быстрее Python
в 4/5 пар на каждом устройстве; VM — в 1/5 CPU и 0/5 MPS. VM после/до имеет
смешанные результаты: этот замер не доказывает ускорения полного обучения VM.
Микроускорение реальное, но устойчивое превосходство обеих lanes над Python
пока не достигнуто.

[CPU raw/paired results](../build/bench/native-overhead/mnist-cpu-verified/results.json),
[MPS](../build/bench/native-overhead/mnist-mps/results.json),
[проверенный native build](../build/bench/native-overhead/mnist-final-build-provenance.json).
Первый `mnist-cpu` остановился на warmup сохранённой VM из-за отсутствующего
`SPUTNIK_NATIVE_RUNTIME_ROOT`; его данные не участвуют в таблице.

## Финальный production benchmark

После direct shared handle/literal tag/moved tag выполнен новый полный
вращающийся A/B на той же фиксированной подвыборке MNIST. Sources/executables
и dataset hashes проверены до/после; full-native 1404/1404, fallback=false.
Native SHA256: `7bb8602016a8cdb41c4cb8a0372748597f0c5dfcf1f02a36341f68b31f4ce9b3`.
Все epoch metrics и восстановленный checkpoint проходят численную проверку.

| Устройство, train + eval (секунды, медиана) | Native исходный | Native финальный | VM исходная | VM финальная | Python |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU | 2,99261 | 2,99012 | 3,08396 | 3,07454 | 3,01723 |
| MPS | 0,80528 | 0,80303 | 0,91923 | 0,93579 | 0,82270 |

Финальный native выигрывает у Python в 5/5 CPU и 5/5 MPS пар. Median paired
native/Python: 0,98772 CPU и 0,97024 MPS; худшая MPS пара 0,99871 — запас
всего 0,13%, поэтому это не доказательство устойчивого преимущества при шуме.
VM/Python: median paired 1,01244 CPU и 1,13746 MPS (0/5 и 1/5 побед).

Median paired native финальный/исходный равен 1,00021 CPU и 1,00017 MPS;
результат всего обучения находится в пределах разброса. Доказанное сокращение
затрат на короткие calls не превращается в доказанное ускорение MNIST.
Цель устойчивого превосходства обеих Sputnik lanes над Python остаётся открытой.

[CPU raw/paired](../build/bench/native-overhead/mnist-stage2-cpu/results.json),
[MPS raw/paired](../build/bench/native-overhead/mnist-stage2-mps/results.json),
[production native provenance](../build/bench/native-overhead/mnist-stage2-build-provenance.json).

Финальные матрицы проверяют все 31 Sputnik / 15 Python cases, по три процесса
и пять timed repeats, без samples/сборок в ходе timing. CPU size=1 показывает
обвязку; MPS size=25088 соответствует числу элементов MNIST batch=32.

| Case | CPU 1: VM / native / Python, мкс | MPS 25088: VM / native / Python, мкс |
| --- | ---: | ---: |
| specialized relu → core | 0,952 / 0,447 / 0,480 | 2,974 / 2,727 / 2,752 |
| generic relu → core | 1,109 / 0,482 / 0,480 | 3,067 / 2,370 / 2,462 |
| generic relu → Tensor | 2,117 / 0,566 / 0,475 | 4,169 / 2,482 / 2,407 |
| public relu | 3,108 / 0,680 / 0,405 | 5,231 / 2,710 / 2,409 |

MPS differences между соседними core cases могут быть отрицательными из-за
kernel/queue variability; их нельзя объявлять отрицательной стоимостью generic
dispatch. Крупный оставшийся разрыв VM локализован выше kernel: в CPU paired
контролях создание Tensor добавляет около 1,019 мкс, public forwarding — ещё
0,986 мкс. Это вместе с frames/dispatch следующий участок для профилирования;
`attr … from` сохранены.

[CPU все cases](../build/bench/native-overhead/ffi-stage2-cpu-1/results.json),
[MPS все cases](../build/bench/native-overhead/ffi-stage2-mps-25088/results.json).

## Проверки корректности

`vm_tests` и `sputnik_ext_tests` проходят. Новый `test-native-call-buffers`
проходит в VM и full native: 12 аргументов, 2048 временных значений, сохранённые
string/list views, callback и fault/recovery, escaping captures, defaults,
keyword override и closure, очищающая свою внешнюю ссылку. Дополнительные
higher-order/conversion/native-class fixtures дают одинаковый результат VM/native.
Collected handle проходит nested callback, переживает обе арены, сохраняет
alias tombstone, уничтожается ровно один раз; owned-class fixture также проходит
в VM/native. 11 acceptance tests optional MNIST runner проходят.

RSS stress `native_cycle_lifetime_test.py` не проходит на этой машине и до
изменений: peak 706904064 bytes до и 708132864 после arena/captures при лимите 134217728.
Это воспроизводимая существующая проблема, не успешная проверка memory bound.
