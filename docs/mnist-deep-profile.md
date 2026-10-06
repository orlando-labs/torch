# MNIST: атомарное профилирование VM, native и Python

Проверка 2026-10-04 на Apple M4, macOS 26.6.2, PyTorch/LibTorch 2.14.0,
4 intra-op threads, MPS без CPU fallback. Модель — SmallResNet, 19 706
параметров; данные, начальные веса и порядок батчей совпадают с Python.
Результат «VM и native стабильно быстрее Python во всех повторах» пока не достигнут.
Профилирование обнаружило конкретные ошибки горячих путей VM; они исправлены.

## Методика и исходные материалы

Все три процесса запускались последовательно, без параллельных сборок и тестов.
Сначала сняты шестисекундные macOS `sample` с интервалом 1 мс для длинного
обучения: MPS 80 эпох, CPU 16 эпох, train/test 2048/512, batch 32.
Затем выделены 15 атомарных случаев: пустой цикл и dispatch, контекст autograd,
метаданные, шаг обучения с готовыми тензорами, AdamW с фиксированными
градиентами, ReLU, свёртка, полный eval forward, батч и эпоха загрузчика.
У каждого случая один прогрев и пять измерений внутри процесса, синхронизация
устройства до/после таймера. Итоговый checksum считается вне таймера.
Пустой цикл включён, поскольку вызов измеряемого блока имеет свою стоимость.

Исходные `sample`, логи, JSON, hashes и таблицы находятся в
`build/bench/mnist/deep-profile/`. В частности:

- `mps-*.sample.txt`, `cpu-*.sample.txt`: исходные полные стеки всех потоков;
- `atoms-cpu/`, `atoms-mps/`: сравнение монomorphic/polymorphic кэша;
- `boundary-cpu/`: отдельные счётчики и время каждого C ABI entry point;
- `batch-hotspot.sample.txt`: целевой sample подготовки батча до исправления;
- `batch-fixed-default.sample.txt`: тот же участок после исправления;
- `forward-{before,fixed}-default.sample.txt`: isolated MPS forward;
- `atoms-defaults-mps/`: matched A/B исправления defaults;
- `call-cache-{before,after}.log`: отдельная проверка CALL/constructor cache.
- `cache-admission-pair.json`: отдельный A/B переполнения polymorphic cache;
- `forward-stable-cache.sample.txt`: forward после исправления переполнения;
- `epoch-before-vm.sample.txt`: isolated DataLoader с повторной генерацией порядка.

`sample_report.py` вычитает веса дочерних узлов из веса родителя. Получаются
exclusive samples по каждому потоку, области выполнения и ближайшему
`amt_*`. Inclusive веса предков служат для поиска стека; складывать их нельзя.
Ожидания разных потоков также нельзя складывать в процент wall time процесса.
Profiler и throughput запускаются отдельно: их времена несопоставимы.

## 1. Defaults запускали дочернюю VM и сканировали модуль

CPU `TensorDataset.batch(indices)` занимал около 80–84 мкс в VM при 7,4 мкс
в native и 5,2–5,3 мкс в Python. Сначала это выглядело как FFI overhead.
Отдельный C ABI profiler показал иное: `amt_tensor` и `amt_tensor_list_index_select`
вместе занимали около 8 мкс на вызов батча, и их стоимость в VM/native была
близка. Большая часть времени находилась снаружи LibTorch.

Целевой sample — 5653 samples главного потока, 5138 (90,9%) отнесены к Sputnik
runtime. Два непересекающихся поддерева внутри `materialize_defaults`:

| Стек | Inclusive samples | Доля потока |
| --- | ---: | ---: |
| `acquire_block_vm → sync_runtime_names_to` | 1485 | 26,3% |
| `execute → prepare_direct_entry_captures → module_closure_for_code` | 1796 | 31,8% |

Связь с исходником: `Dataset.batch → tensor(indices, dtype: "int64")`.
Пропущенные `device: "cpu"` и `requires_grad: false` представлены default
thunks. Disassembly `sputniktorch.tensor.factories.sputnikbc` показывает, что эти
thunks содержат только load literal, close upvalues, return, но наследуют
capture layout всей функции, включая импорты, используемые только в её теле.
Проверка `capture_layout.empty()` ошибочно запрещала быстрый путь для литерала.

Исправление: доказанный трёхинструкционный immutable scalar literal возвращается
непосредственно, независимо от неиспользуемых captures. Обычные defaults с
вычислениями и выделением объектов продолжают выполнять thunk. Их фактические
captures теперь передаются из frame вызываемой функции. Ранее попытка найти
отдельную closure такого thunk могла повторно запустить ещё не завершённую
инициализацию модуля и вызвать stack overflow.

| Случай, MPS suite / CPU data | VM до, мкс | VM после, мкс | native, мкс | Python, мкс |
| --- | ---: | ---: | ---: | ---: |
| `dataset_batch`, 32 примера | 84,050 | 18,395 | 7,492 | 5,200 |
| `data_epoch`, 2048 примеров | 6313,698 | 2076,148 | 1244,754 | 461,650 |
| `model_eval`, готовый batch | 886,883 | 887,068 | 516,695 | 507,148 |

Батч ускорился в 4,57 раза; forward не изменился. Это разные причины.
В длинном отдельном driver после исправления: 2 млн батчей за 35,411 с,
17,706 мкс/батч. Прежние горячие стеки defaults исчезли из sample.

## 2. Кэш метода сбрасывался при смене типа слоя

Кэш SEND/CALL хранил один receiver class. Один site `layer(value)` внутри
`Sequential` работает с разными классами слоёв. Поэтому обычный цикл сети
постоянно разрешал метод заново и создавал три `vector<bool>` для обхода
классов/mixins.

Добавлен ограниченный polymorphic cache до восьми shapes с прежними guards:
receiver class, selector, arity, keywords, block, world epoch, method version.
Проверены смена метода после прогрева и переполнение кэша.

Пустой вызов шести различных слоёв: MPS 5,045 → 4,598 мкс, CPU
5,065 → 4,682 мкс. Однотипные слои почти не изменились. Проверка считает
именно шесть cold misses и 54 hits за десять проходов одного bytecode site;
после изменения метода все alternatives инвалидируются.

Полный forward по-прежнему обновлял кэш после прогрева: общее место
вызова в Module/Sequential видит больше восьми классов. Первоначальное
вытеснение создавало циклические промахи. Заполненный кэш теперь сохраняет
уже прогретые shapes; остальные типы проходят обычное разрешение без
вытеснения. Bound остаётся восемь. После cache hit подсказка для overflow очищается;
две последовательные misses одной новой shape допускают её в кэш, чтобы
сайт мог перейти к новому hot receiver после стадии инициализации.
Полные guards используются независимо от admission hint. При добавлении
alternative копируется
только её flat запись, без копирования всего вектора alternatives root.

Регрессия с десятью классами на одном site проверяет не только правильный
результат, но и сохранение hits после переполнения: восемь дополнительных
misses и 22 hits за три обхода десяти классов после прогрева шести,
а не новый цикл вытеснений. Инвалидация world/method и адаптация к новому hot receiver проверены.

Отдельный MPS A/B, два процесса каждой версии с вращающимся порядком и
пятью in-process измерениями, при одинаковых восстановленных `attr from`:
eval-forward 844,658 → 819,717 мкс (−3,0%), preloaded training step
2903,416 → 2874,691 мкс (−1,0%). Пустой dispatch и batch почти не изменились.
В новом forward sample обновления кэша отсутствуют среди горячих стеков.

Прямые чтения полей `_core` (Tensor и ProcessGroup), `Module.training?`
и `TensorDataset.count` возвращены к `attr … from @field`. Это read-only
properties; вызов с `()` имеет другую семантику и отвергается. VM использует
guarded direct-ivar путь для compiler-generated attr reader. Вычисляемые
метаданные Tensor не выражаются через `attr from`: RFC разрешает там только
прямое instance field, а не `@core.shape` или произвольное выражение.

## 3. CALL искал метод до проверки кэша; init не использовал кэш

В целевом forward sample поиск метода оставался заметен даже после прогрева.
Переход от sample к строкам `Vm::step`, opcode CALL, обнаружил два пути:

- у instance-object CALL `find_method_for_dispatch(..., "call")` выполнялся
  перед `probe_call_cache`, в том числе на cache hit;
- constructor CALL `Tensor(core)` каждый раз разрешал `init` без call-site cache.

Теперь instance CALL сначала проверяет cache; constructor CALL использует тот
же guarded cache для положительно найденного `init`. Отсутствующий init не
кэшируется. Создание экземпляра и выполнение init остаются на каждом вызове.
Регрессия проверяет два класса на одном site, новые экземпляры на каждом
вызове и замену init после прогрева.

Отдельный MPS A/B с неизменной bridge:

| Случай | До, мкс | После, мкс |
| --- | ---: | ---: |
| mono dispatch, 6 слоёв | 4,645 | 3,547 |
| poly dispatch, 6 слоёв | 4,740 | 3,596 |
| ReLU core + Tensor wrapper | 5,056 | 4,845 |
| Conv module | 33,100 | 32,221 |
| model eval | 890,534 | 862,765 |

Шаг обучения: 2938,962 → 2965,875 мкс в этом A/B; общего ускорения обучения
на его основании заявлять нельзя. Пустые calls ускорились на 24%, forward —
на 3,1%, но основной разрыв VM/Python ещё остаётся.

## 4. Создание строк ошибки на успешном C ABI пути

`require(bool, kind, const std::string&)` получал строковые литералы.
Длинное сообщение выделялось даже при истинном условии. Для индекса батча
проверки диапазона повторяются на каждом элементе. Добавлена overload для
`const char*`: строка создаётся только при ошибке; типы ошибок и сообщения
сохранены.

Сравнение двух native бинарников: четыре процесса каждого варианта,
чередующийся порядок; внутри каждого атома пять повторов. Полная native
coverage, одинаковые checksums:

| Случай | До, мкс | После, мкс |
| --- | ---: | ---: |
| core numel | 0,1223 | 0,1026 |
| dataset batch | 7,445 | 5,944 |
| data epoch | 1245,278 | 1154,902 |
| optimizer step | 201,059 | 198,997 |
| preloaded train step | 2661,837 | 2651,523 |
| model eval | 517,843 | 518,021 |

20% выигрыша в подготовке индекса не означают 20% выигрыша всей сети.
Forward в этом эксперименте практически не изменился. Данные:
`guard-pair/results.json`, с hashes обоих бинарников и сырыми повторами.

## 5. Native создавал OpenMP pools на разных вызывающих потоках

All-thread sample CPU показал несколько OpenMP pools. Каждый `execution_batch`
эпохи мог попасть на новый blocking worker, а LibTorch/libomp создавал для
него отдельную группу потоков. Добавлен `--execution-scope run|epoch`;
по умолчанию весь benchmark job исполняется в одной execution_batch.
Алгоритмы обучения остаются в Sputnik. Старый режим сохранён для сравнения
одного и того же бинарника.

16 CPU эпох одного native executable, одинаковые численные результаты:
75 потоков в sample с epoch scope, 27 с run scope — на 48 меньше.
Это соответствует 16 дополнительным группам по три OpenMP workers.
Loop 16,826 → 16,156 с в этих двух профилируемых процессах. Число потоков и
стеки подтверждают проблему; одиночный profiled timing не доказывает
стабильный выигрыш throughput. Данные: `scope-pair/`.

## 6. Tensor → Array создавал view и вызывал item на каждом элементе

Эпоха загрузчика повторно строит порядок через
`Generator.randperm!(2048).to_a`. В `array_value` каждый элемент получался
рекурсивным `select(0, i)`, а каждый scalar — `item`. Для одной перестановки
создавались 2048 отдельных TensorImpl и выполнялись тысячи LibTorch dispatches.

Целевой all-thread sample DataLoader отделён от forward/backward. В VM
`amt_materialize` покрывает 1318 из 4555 samples активного потока (28,9%).
В native — 2233 из 4581 (48,7%). Собственно `amt_generator` имеет
соответственно 19 и 23 samples: основная цена порядка была в преобразовании.

Исправление читает detached CPU tensor через typed const data pointer,
sizes и strides. Оно сохраняет storage offset, noncontiguous views, Bool/Int/Float,
int64 без преобразования через double и вложенные пустые dimensions.
Для иных layouts/dtypes и lazy conjugate/negative views сохранён прежний путь.
Перенос MPS → CPU и ограничения materialization при ONNX export не изменены.

Три отдельных процесса каждой lane, по 3000 операций/процесс, чередующийся
порядок. Checksums массива и эпохи совпали. Для native использован тот же
сгенерированный Sputnik C++, те же runtime sources и остальные bridge objects;
заменён только объект `native/tensor/tensor.cpp`. После замеров hashes входов
проверены повторно. Этот A/B изолирует изменение C++ преобразования.

| Случай | VM до, мкс | VM после, мкс | native до, мкс | native после, мкс |
| --- | ---: | ---: | ---: | ---: |
| to_a, 2048 int64 | 562,015 | 33,434 | 570,595 | 21,469 |
| DataLoader epoch, 2048 samples | 1866,372 | 1330,141 | 1127,380 | 597,667 |

to_a ускорился в 16,8 раза в VM и в 26,6 раза в native; эпоха загрузчика —
в 1,40 и 1,89 раза. В новом native sample доля `amt_materialize` — 146 из
4594 samples (3,2%). Сырые результаты и provenance: `array-pair-results.json`,
`array-{before,after}/`, `epoch-{before,after}-{vm,native}.sample.txt`.
Python reference для этого atom вызывает `Tensor.tolist()`.

`tests/materialization.s` проходит в VM и full-native на CPU и настоящем
MPS без fallback: десять CPU dtype / девять MPS dtype, scalar, пустые
размерности, transpose+narrow со storage offset, типы скаляров и int64
больше 2^53. Native executable проверен с PATH=/nonexistent; Python runtime
библиотеки не загружены. Существующие CPU/MPS tensor/autograd/checkpoint
и MPS training VM-проверки также проходят.

## 7. Рекурсивная проверка stateful добавляет отдельную работу

На готовой eval-модели отдельно измерены три Sputnik пути, по три процесса:
`model.stateful?`, обычный `model(x)` и диагностический `model.call!(x)`.
Root stateful-check обходится в 32,16 мкс. Immutable forward имеет медиану
856,48 мкс, mutating forward на той же eval-модели — 752,91 мкс.
Итоговый tensor checksum в обоих случаях равен 23,9462852478.

Разница 103,57 мкс включает проверку stateful у root и повторные проверки
вложенных контейнеров. Это A/B разных публичных call paths, а не оценка
чистой стоимости одной функции: у forward! есть и другие adapters.
Использовать call! в основном eval-бенчмарке ради выигрыша не стали.
Безопасное устранение повторных проверок должно учитывать независимый
`child.train!`, регистрацию дочернего модуля и пользовательские overrides
stateful?. Простое кэширование false может изменить семантику.
Данные: `stateful-atoms.json` и `guard-{stateful,forward,forward_mutating}-*.log`.

## 8. Где остаётся разрыв

На MPS один main-thread sample isolated forward после исправления defaults:
4453 samples, 2134 (47,9%) runtime, 1748 (39,3%) Metal/MPS, 495 (11,1%)
LibTorch, 44 bridge и 32 waits. Runtime содержит interpreter steps, Value
copy/release, frames, argument shaping и collection callbacks. Это не
свидетельство медленного convolution kernel. В matched атомах core convolution
занимает 28,4 мкс VM, 24,8 мкс native, 24,4 мкс Python; полный eval forward —
887, 517 и 507 мкс соответственно.

Метаданные `numel`: около 0,53 мкс VM, 0,12 мкс native, 0,035 мкс Python.
Полный optimizer step: 267, 202 и 200 мкс. Для CPU, где kernels крупнее,
preloaded train step: 14,7 мс VM против 14,1 мс native и 14,0 мс Python.
Таким образом, relative overhead особенно заметен на мелких операциях MPS.

`--require-full-native` подтверждает отсутствие VM fallback. Backend
`cpp-bytecode-direct-v1` всё ещё использует динамические NativeValue,
аргументы и runtime dispatch; сама полная native coverage не означает
устранения расходов обвязки.

C ABI profiler считает exclusive wall и CPU время вызывающего потока.
CPU время LibTorch/OpenMP workers не включено. Для CPU atom suite число
вызовов C ABI в VM и native совпадает. Внешний `execution_batch` callback
занял 4,49 с CPU runtime в VM против 0,80 с native; при этом время отдельных
LibTorch boundary calls близко. Это независимое подтверждение, что остаток
нужно искать в Sputnik/runtime, а не менять модель ради меньшего числа kernels.

Дополнительно снят CPU operator profile Python на десяти MPS train steps:
60 conv2d, 60 BatchNorm, 50 ReLU, 10 adaptive average pooling, 10 matmul,
10 cross entropy, 60 convolution backward, 10 gradient cat и по десять
foreach mul/sub. Это соответствует шести conv/BN, пяти ReLU и одному
pooling/matmul на шаг модели, а также packed optimizer. Fused nn.Linear
в Python не используется: reference выполняет такой же matmul + add,
как Sputnik. Operator counts служат проверкой состава работы; CPU profiler
тоже добавляет overhead. Данные: `python_ops.json`.

## Throughput до возврата атрибутов и исправления materialization

Свежая full-native сборка и пять отдельных процессов каждой lane, один
процесс-прогрев, вращающийся порядок; train/test 2048/512, batch 32,
три эпохи, четыре threads, packed optimizer в Python, run scope в Sputnik.
Все epoch losses, accuracy и checkpoint round trips совпали.

| Устройство | VM, с | native, с | Python, с | VM wins | native wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| MPS | 0.941391 | 0.783188 | 0.793538 | 0/5 | 4/5 |
| CPU | 3.025000 | 2.918144 | 2.943677 | 1/5 | 3/5 |

Медианы native немного лучше Python, но условие выигрыша в каждой паре
не прошло на обоих устройствах. VM на CPU отстаёт по медианам на 2,8%, на
MPS — на 18,6%. Process wall быстрее Python на обоих устройствах за счёт
меньшего startup, однако эта фаза не заменяет сравнение вычислительного цикла.
Сырые результаты и provenance: `final-mps-32/`, `final-cpu-32/`.

Atomic costs до возврата attr from и исправления materialization: `atoms-final-mps/`.
Например, dataset batch: 15,946 мкс VM, 5,897 мкс native, 5,178 мкс Python;
data epoch: 1903,044 / 1153,729 / 455,500 мкс. Относительно defaults A/B
это уже включает CALL cache и lazy error diagnostics. Новый батч VM
быстрее исходных 84 мкс в 5,3 раза; сравнивать эту величину с общим
throughput speedup следует с учётом доли batching в полном цикле.

## Throughput с возвращёнными attr from и новым materialization

Свежая full-native сборка, `execution_scope=run`, тот же split 2048/512,
batch 32, три эпохи, packed Python optimizer и четыре threads. Один процесс
прогрева и пять отдельных измерений каждой lane, вращающийся порядок.
Epoch losses, accuracy и checkpoint round trips совпали.

| Устройство | VM, с | native, с | Python, с | VM wins | native wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| MPS | 0,911028 | 0,774299 | 0,786188 | 0/5 | 5/5 |
| CPU | 2,939839 | 2,844607 | 2,859082 | 0/5 | 4/5 |

Native прошёл проверку преимущества в каждой MPS паре; минимальный выигрыш
0,77%, медиана на 1,5% быстрее Python. VM ещё медленнее Python на 15,9%
по медианам. На CPU отставание VM — 2,8%; native быстрее по медиане на 0,5%,
но одна пара медленнее Python на 0,15%. Runner с
`--require-faster-than-python` вернул код 2 в обоих запусках: на MPS из-за
VM, на CPU из-за VM и одной native пары. Численная проверка прошла, отчёт и все пары сохранены в
`attrs-array-mps-32/` и `attrs-array-cpu-32/`. Маленький минимальный запас native требует дальнейших
повторов; эти пять побед не доказывают выигрыш на любом workload.

Свежий full-native atomic driver содержит 16 случаев и проходит checksum
проверки во всех трёх lanes (`attrs-array-atoms-mps/`). Оставшиеся расходы:

| Случай | VM, мкс | native, мкс | Python, мкс |
| --- | ---: | ---: | ---: |
| sampler to_a, 2048 indices | 33,604 | 21,775 | 21,382 |
| preloaded train step | 2872,148 | 2632,490 | 2672,432 |
| optimizer step | 256,779 | 198,416 | 194,642 |
| core convolution | 25,624 | 24,666 | 24,192 |
| module convolution | 31,655 | 25,695 | 24,613 |
| immutable model eval | 814,659 | 515,497 | 509,045 |
| dataset batch | 15,765 | 5,944 | 5,267 |
| DataLoader epoch | 1357,704 | 606,633 | 439,442 |

После исправления materialization разрыв VM остаётся в Sputnik dispatch,
frames, коллекционных callbacks, упаковке/передаче аргументов и рекурсивной
stateful-проверке. Например, module convolution добавляет к core около
6 мкс в VM и 1 мкс в native; полный eval сохраняет около 306 мкс разрыва
VM/Python. Эти цифры локализуют следующую работу, но не доказывают, что
одна из этих частей отдельно объясняет весь остаток.

## Воспроизведение

Python используется только для опционального сравнения и анализа. Пакет,
сборка и тесты Sputniktorch его не требуют.

```sh
./tools/build bench/mnist/atoms.s
/opt/homebrew/bin/python3 bench/mnist/profile.py --atoms --device mps \
  --output build/bench/mnist/my-atoms

# C ABI counts and exclusive times; overhead makes this a diagnostic run.
/opt/homebrew/bin/python3 bench/mnist/profile.py --atoms --boundary --device cpu \
  --output build/bench/mnist/my-boundary

./tools/build bench/mnist/benchmark.s
/opt/homebrew/bin/python3 bench/mnist/profile.py --sample --device mps --epochs 80 \
  --output build/bench/mnist/my-stacks

# Isolated Sputnik stack. Build full native too if comparing its executable.
./tools/build bench/mnist/hotspot.s
../sputnik-lang/build/sputnik run build/sputniktorch.mnist_hotspot.build.json \
  --grant ffi --grant fs.read -- --case forward --device mps --iterations 20000
# In another terminal: sample <pid> 6 1 -file /tmp/mnist-forward.sample.txt
# The same driver: --case array or --case epoch --device cpu.

# CPU/MPS materialization validation, independent of Python.
./tools/build tests/materialization.s
./build/sputnik/sputniktorch.materialization_test
PYTORCH_ENABLE_MPS_FALLBACK=0 ./build/sputnik/sputniktorch.materialization_test --device mps

# Uninstrumented, numerically checked three-lane throughput.
/opt/homebrew/bin/python3 bench/mnist/run.py --device mps --epochs 3 \
  --batch-size 32 --train-limit 2048 --test-limit 512 --repeats 5 --warmups 1 \
  --python /opt/homebrew/opt/pytorch/libexec/bin/python --python-optimizer packed
```

`profile.py --before-vm /path/to/sputnik` добавляет старую VM в atomic A/B.
Она должна поддерживать ту же bytecode/FFI ABI. Profiler сам не пересобирает
native: соответствующий driver нужно построить заранее. Логи фиксируют
hashes исполняемых файлов и текущих sources; throughput runner дополнительно
проверяет build provenance и численное совпадение каждой эпохи/checkpoint.

Разложение runtime/FFI/kernel на 31 контрольный случай, новые CPU/MPS
матрицы и samples отдельных calls описаны в
[профилировании FFI 2026-10-05](ffi-call-profile.md).
