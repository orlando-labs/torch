# LOOKUP_CONST: оптимизация и проверка, 2026-10-05

В прогретом разрешении имени больше нет повторной сборки сегментов/строк пути, поисков по
реестрам и хеширования module binding. Результат разрешения адресуется индексом
константы. Для функции сохраняется адрес живого binding, а не копия closure.
`has_error_namespace` использует индекс всех непустых dotted prefixes,
пополняемый при регистрации ошибки, вместо перебора всех типов ошибок.

## Семантика

- Сохранён приоритет Sputnik class → exact native error → exact builtin → error
  namespace → module binding → compatibility class fallback.
- Кеш принадлежит отдельной VM. Scheduler strands не записывают общий кеш.
- Изменение world epoch, bindings revision либо revisions двух реестров
  инвалидирует разрешение. Путь хранится отдельно от результата разрешения.
- Живой binding сохраняет замены значения, closure captures и чтение watch cell.
  Rehash сохраняет адрес; очистка таблицы меняет revision до следующего lookup.
- Error namespace и flow module создаются заново при каждом обращении.
  Task module берётся из текущей VM и её scheduler.
- `LOOKUP_CONST` имеет QuickOpcode: корректные операнды декодируются один раз.
  Ошибочные операнды проходят прежний проверяемый fallback.

Кеш не хранит heap Value/closure snapshots и не добавляет скрытых GC roots.
Bindings остаются собственностью RuntimeState и входят в его обычные roots.
`attr … from`, алгоритмы SputnikTorch и семантика вызовов не менялись.

## Отдельный бенчмарк

[Исходники и запуск](../../sputnik-lang/bench/vm/lookup_constant/README.md).
Пять чередующихся пар процессов before/after, один прогрев и пять внутренних
повторов; 300000 итераций. Bytecode подтверждает `LOOKUP_CONST` в теле цикла
метода, который читает экспортированную функцию provider. Calls блока/метода
выполняются один раз на batch, вне внутреннего цикла. Значения и captures
проверяются после таймера. Сборки, тесты и другие benchmarks не совмещались
с измерениями.

Наносекунды на итерацию с одинаковым while loop, медиана process medians:

| Case | До | После | Ускорение |
| --- | ---: | ---: | ---: |
| local function | 98.42 | 93.73 | 1.050× |
| qualified function | 702.37 | 116.57 | 6.025× |
| Sputnik class | 116.51 | 108.51 | 1.074× |
| builtin type | 162.20 | 111.67 | 1.453× |
| error class | 156.00 | 110.92 | 1.406× |
| task module | 190.61 | 131.75 | 1.447× |

Qualified − local, сначала разность внутри процесса: **603.85 → 22.57 нс**.
Это сравнение двух путей байткода, не чистая стоимость инструкции. Контроль
local тоже изменился; весь выигрыш нельзя приписывать отдельному подэтапу.
Команды, hashes, все повторы и disassembly:
[results.json](../../sputnik-lang/build/lookup-const-atoms/results.json),
[bytecode.txt](../../sputnik-lang/build/lookup-const-atoms/bytecode.txt).

## SputnikTorch

Один и тот же benchmark из 31 cases, три процесса на lane с вращением
VM/native/Python, пять внутренних повторов. PyTorch/LibTorch 2.14.0,
четыре threads, float32 size=25088. Actual MPS с fallback=0; синхронизация
на границах kernel loops. Checksum проверяется вне таймера; native full
coverage и source/compiler/executable fingerprints проверены runner.

| VM case | До, мкс | После, мкс | Снижение времени |
| --- | ---: | ---: | ---: |
| CPU public relu | 5.786 | 4.864 | 15.94% |
| MPS public relu | 5.937 | 5.305 | 10.64% |
| CPU execution_mode | 0.747 | 0.747 | 0.02% |
| CPU loop_scalar | 0.331 | 0.332 | −0.33% |

В `Tensor.relu` исходный bytecode содержит два LOOKUP_CONST: `Tensor` и `unary`;
linker разрешает импорт `unary` в путь exporting module. `execution_mode`
в свободной функции загружается через lexical capture: его контрольный
benchmark не показывает большого изменения.

Полные результаты:
[CPU до](../build/bench/lookup-const/before-cpu-verified/results.json),
[CPU после](../build/bench/lookup-const/after-cpu/results.json),
[MPS до](../build/bench/lookup-const/before-mps/results.json),
[MPS после](../build/bench/lookup-const/after-mps/results.json).

Контроль size=1, только public relu, 10000 kernel iterations; тот же native
binary для before/after VM (runner `--vm-sputnik`). Время VM:

| Устройство | До, мкс | После, мкс | Снижение времени |
| --- | ---: | ---: | ---: |
| CPU | 3.722 | 3.086 | 17.08% |
| MPS | 5.512 | 4.607 | 16.43% |

[CPU до](../build/bench/lookup-const/before-cpu-1/results.json),
[CPU после](../build/bench/lookup-const/after-cpu-1/results.json),
[MPS до](../build/bench/lookup-const/before-mps-1/results.json),
[MPS после](../build/bench/lookup-const/after-mps-1/results.json).

## Свежие samples до/после

Одинаковый CPU `relu_public`, size=25088, по 2000000 kernel iterations,
warmup и один timed batch. Два отдельных macOS sample runs по шесть секунд,
запрошенный интервал 1 мс. Анализируется активный main thread; wait threads
не суммируются с его CPU. Различное число samples требует нормирования,
а не сравнения сырых счётчиков.

| Семейство, inclusive | До, 3928 active samples | После, 2090 active samples |
| --- | ---: | ---: |
| Vm::lookup_constant | 430 / 10.95% | 5 / 0.24% |
| has_error_namespace | 267 / 6.80% | 0 / 0.00% |

Namespace — вложенная часть lookup_constant; эти доли нельзя складывать.
Ноль в samples означает отсутствие наблюдаемых сэмплов в этом run, а не
нулевую стоимость холодного разрешения. Исходник отдельно подтверждает,
что прогретый путь не вызывает реестры. Остаток runtime в профиле: step,
Value lifetime/copies, method lookup, vector<bool> construction и frames.

[Сырые samples до](../build/bench/lookup-const/samples-before/sputnik-vm.sample.txt),
[после](../build/bench/lookup-const/samples-after/sputnik-vm.sample.txt),
[подсчёт семейств](../build/bench/lookup-const/sample-families.json).
Checksums обоих длинных runs совпали.

## MNIST

Три epochs, batch=32, train=2048, test=512; по одному warmup и пять
процессов каждого lane с вращением порядка. `execution_scope=run`,
Python optimizer=packed, четыре threads. CPU и настоящий MPS, fallback=0.
Loss/accuracy сопоставлены с Python; checkpoint restore проверен.
Native сборки имеют full coverage без bytecode fallback.

| Устройство | VM до/после, с | Native до/после, с | Python до/после, с |
| --- | ---: | ---: | ---: |
| CPU | 3.0763 / 3.0022 | 3.0014 / 2.9025 | 3.0062 / 2.9376 |
| MPS | 0.9540 / 0.9260 | 0.8126 / 0.7892 | 0.8107 / 0.8014 |

Это train+eval loop, без startup/checkpoint. Сравнение сначала внутри
process repeat даёт VM/Python **1.0174 → 1.0209** на CPU и
**1.1786 → 1.1513** на MPS. На CPU общий сдвиг controls сопоставим с
изменением VM: эти результаты не доказывают сквозного выигрыша от LOOKUP_CONST.
На MPS разрыв уменьшился примерно на 2.7 процентного пункта, но VM всё ещё
медленнее Python во всех пяти repeats. Native median быстрее Python,
устойчивого превосходства в каждой паре также нет.

Оптимизация LOOKUP_CONST подтверждена отдельным A/B, public relu и samples.
Оставшееся замедление MNIST нужно искать в других частях runtime;
ускорение одной инструкции нельзя переносить на весь workload.

[CPU до](../build/bench/lookup-const/mnist-before-cpu/results.json),
[CPU после](../build/bench/lookup-const/mnist-after-cpu-verified/results.json),
[MPS до](../build/bench/lookup-const/mnist-before-mps/results.json),
[MPS после](../build/bench/lookup-const/mnist-after-mps/results.json).
Проверенная native сборка для after runs:
[provenance](../build/bench/lookup-const/mnist-native-build-provenance.json).

## Проверки

Полные `vm_tests` и `stdlib_registry_tests` прошли. Новые regression tests
проверяют rehash и замену closure с captures, watch mutation, очистку/удаление
binding, новую регистрацию namespace/error, смену категории builtin,
replacement registry с одинаковой revision и новым world epoch, приоритет
Sputnik class, идентичность namespace/flow/task, malformed operands,
неизвестный и неоднозначный class path. Проверены generated bootstrap и
пустой registry, вложенные/неполные prefixes, duplicate/rejected registration,
leading/trailing dots и пустой namespace.

Python drivers проверены компиляцией; Python остаётся опциональным средством
измерения. Package/build/test dependency от него не появляется.
