# Native dispatch и MNIST, 2026-10-07

Native Sputnik всё ещё имеет существенные расходы динамического runtime.
Полное native-покрытие убирает исполнение bytecode, но оставляет boxed values,
dispatch по code/class id, shaping аргументов, кадры, callbacks и управление
временем жизни. Короткий public ReLU на CPU по-прежнему медленнее Python.
В полном MNIST native близок к Python; стабильного преимущества во всех парах
пока нет. Это разные, отдельно измеренные результаты.

Этот этап продолжает [профиль вызовов](native-call-overhead-profile.md) и
[оптимизацию VM](vm-value-dispatch-profile.md). Sputnik-модель, optimizer,
данные и количество LibTorch kernels сохранены. Алгоритмы не перенесены в C++.
Python нужен только для опционального reference/анализа.

## Найденные и исправленные проблемы native

### 1. Гонка между code id и flags в method cache

`NativeUserCallSiteCache` публиковал flags отдельным relaxed atomic, затем
class/key + code id через release atomic. Reader делал acquire packed, после
чего отдельно читал flags. Acquire не защищает от следующей записи другим
потоком: reader мог получить code одного метода и flags другого.

Регрессия использует один explicit-call site для обычного `value()` и
read-only `attr value from @value`, восемь tasks и 50000 вызовов каждого.
Первый вызов должен вернуть 42, второй — поднять TypeError. Старый native
воспроизвёл пять ошибок dispatch; исправленный — ноль.

Теперь одна atomic запись хранит receiver key и индекс неизменяемой таблицы
`kNativeUserMethods`. Code id и flags извлекаются из одной записи таблицы.
Отдельной публикации flags больше нет. Кэш корректен для закрытого native
графа; динамическую замену методов этим изменением не вводили.

Исходники регрессии: `sputnik-lang/tests/fixtures/native_dispatch_cache/`.
Логи до/после: `sputnik-lang/build/native_dispatch_cache/{before,after}.log`.
Этот fixture проверяет native. Общие VM-тесты выполняются отдельно.

Отдельно обнаружено различие VM/native: invalid property call в VM использует
`set_fault`, поэтому его TypeError обходит language `rescue`. Это воспроизведено
даже в последовательном driver и в сохранённой VM до этого этапа; исходная
гонка двух native atomics этим поведением не объясняется. Семантика VM faults
в данном patch не менялась. Диагностики сохранены в
`build/bench/native-stage4/vm-dispatch-sequential{,-before}.log`.

### 2. Attr reader исполнялся как полноценный метод

Native создавал invocation и frame даже для прямого чтения instance field.
Для property также создавался `NativeNoSuspendScope` со строкой диагностики.
VM уже имела guarded direct-ivar путь; native его не имел.

Generator теперь доказывает форму reader: флаг compiler-generated attr,
нулевая сигнатура, отсутствие captures/defaults/clauses/handlers и ровно
`LoadSelf → LoadIvar → CloseUpvalues → Return` с согласованными регистрами.
После проверок синтаксиса property, receiver, arguments и block возвращается
`native_load_ivar`. Override и вычисляемый getter остаются обычными методами.

Standalone A/B, пять процессов и пять внутренних повторов:

| Случай, нс/итерацию | До | После |
| --- | ---: | ---: |
| Пустой captured block | 24,30 | 25,07 |
| Attr reader | 96,30 | 33,87 |
| Обычный method reader | 51,78 | 53,94 |
| Keyword default | 90,51 | 81,69 |
| Keyword override | 75,92 | 65,34 |
| Rest + keywords | 250,20 | 242,58 |

Ускорение reader — 2,84×, а не ускорение всех методов. Этот ранний A/B
предшествовал уточнению фоновой нагрузки; контрольные строки и каждый
повтор сохранены в [dispatch-ab](../build/bench/native-stage4/dispatch-ab/results.json).

### 3. Heap allocation для коротких shaped calls

`native_shape_method_args` всегда создавал `std::vector<NativeValue>`.
Добавлен `NativeShapedArgs`: восемь inline slots и vector fallback для больших
сигнатур. Позиционные, keyword/rest/block параметры, зависимые defaults и
исключения сохраняют прежнюю семантику. Заполняется прежнее число параметров.
Inline array пока конструирует все восемь Null values; это оставшийся расход,
а не доказанная следующая оптимизация.

### 4. Лишние list access и многократный рост ABI arena

Generated `sputnik_native_ext_list_at` дважды вызывал `as_list` для одного
элемента. `as_list` вызывал out-of-line materialization даже для обычного,
уже materialized List. Теперь items читаются один раз, а inline guard вызывает
отдельную cold функцию только для lazy chars.

При первом переполнении 16 inline ABI values arena раньше росла через
capacities 1/2/4/8/16/32. Один List[32] вызывал несколько reallocations и
перемещений за один короткий FFI call. Теперь первый spill резервирует
32 slots, не создавая дополнительных values и не перемещая inline entries.
Последующий большой рост по-прежнему поддерживается. Общий arena помогает
также VM.

### 5. Default thunk заново копировал всё окружение

В `native_shape_method_args` оставался `NativeClosure thunk = *invocation`.
Copy constructor материализует даже borrowed captures, копирует vector и
делает retain/release каждого cell. Скалярный default может наследовать пять
imports, используемых только телом функции, и оплачивать их все.

Добавлен `native_literal_default_value`: generator проверяет DefaultThunk,
нулевые flags, отсутствие handlers и ровно load/close/return с согласованным
регистром. Только Null/Bool/Int/Float/StringRef/SymbolRef возвращаются напрямую.
Неиспользуемый capture layout не мешает этому доказанному пути. Mutable
defaults и произвольные вычисления продолжают исполнять thunk каждый раз.
Для них временный thunk заимствует capture view invocation; self, block и
nonlocal-return target сохраняются. Escaping closures продолжают владеть
своими captures через существующие copy/move/materialization правила.

Отдельный generated C++ A/B использует один runtime/extension набор, четыре
variants: исходный default path, только literal fix, только borrowed computed
thunks и оба изменения. Пять процессов, rotating order, один process warmup,
500000 iterations и пять timed repeats. Все восемь checksums равны 42.
Generated source подтверждает пять captures у двух целевых функций.

| Случай, нс/итерацию | До | Только literal | Только borrow | Оба |
| --- | ---: | ---: | ---: | ---: |
| Empty block | 25,72 | 25,77 | 25,81 | 25,88 |
| Attr reader | 36,80 | 36,94 | 36,97 | 37,33 |
| Method reader | 55,63 | 54,73 | 55,01 | 54,34 |
| Keyword override, без default | 65,97 | 66,47 | 65,65 | 66,90 |
| Простой scalar default | 78,97 | 62,60 | 81,66 | 63,63 |
| Literal с 5 неиспользуемыми captures | 102,02 | 69,22 | 88,02 | 70,87 |
| Вычисляемый default с 5 captures | 140,78 | 142,06 | 122,40 | 123,45 |

В целевых случаях улучшение порядка 19%, 31% и 12%: after/before paired medians
= 0,8054 / 0,6927 / 0,8770, в каждом случае выигрыш 5/5. Keyword override,
где default не исполняется, = 1,0152; empty block = 0,9989. Контрольный путь
не ускорился вместе с defaults. Literal-only и borrow-only разделяют две
причины. Этот A/B выполнялся после
остальных native исправлений: его baseline уже содержит inline shaped args
и attr fix.
[Повторы и hashes](../build/bench/native-stage4/defaults-ab/results.json),
[source variants и команды](../build/bench/native-stage4/defaults-isolated/provenance.json),
[Sputnik driver](../build/bench/native-stage4/defaults.s).

## Контрольные сборки без отдельных оптимизаций

После устранения конкурирующей задачи собраны шесть generated C++ variants
с одинаковыми production `-O2 -DNDEBUG`, extension objects и runtime archive.
`before` использует исходный arena header. В `no_attr`, `no_shape`, `no_list`
и `no_arena` отключена ровно соответствующая часть нового generated runtime.
Пять процессов каждой lane, вращение порядка, прогрев, пять timed повторов,
500000 host iterations, 10000 kernel iterations, CPU, size 1.
Все 31 Sputnik и 15 Python cases прошли checksum проверку.

| Случай, нс/итерацию | До | Все исправления | Без list fix | Без arena fix | Python |
| --- | ---: | ---: | ---: | ---: | ---: |
| ABI List[4] conversion | 112,36 | 103,44 | 111,93 | 102,69 | — |
| ABI List[32] conversion | 507,04 | 314,91 | 387,23 | 437,66 | — |
| Empty block | 28,34 | 28,84 | 29,07 | 28,98 | 15,37 |
| Method constant | 54,61 | 55,39 | 55,28 | 54,92 | 29,50 |
| Core numel method | 72,92 | 70,65 | 72,66 | 71,81 | 41,53 |
| Tensor numel | 105,62 | 104,59 | 106,02 | 105,66 | 41,66 |
| Specialized ReLU | 474,94 | 474,24 | 470,61 | 469,49 | 486,38 |
| Generic core ReLU | 491,65 | 495,20 | 501,11 | 495,53 | 483,05 |
| Generic + Tensor ReLU | 576,11 | 584,00 | 584,34 | 569,14 | 484,86 |
| Public ReLU | 692,40 | 701,89 | 694,50 | 687,70 | 409,82 |
| Raw C++ ReLU loop | 268,75 | 273,26 | 265,54 | 267,52 | — |

List[32] after/before paired median = 0,6159, выигрыш 5/5; after/no_list =
0,8143 и after/no_arena = 0,7202, также 5/5. Две независимые проблемы
подтверждены отключением соответствующих исправлений. Их эффекты нельзя
складывать как независимые времена.

Для ReLU общего выигрыша нет: public after/before paired median = 1,0174,
выигрыш только 2/5. Разброс raw C++ kernel также присутствует. Небольшие
различия здесь не выдаются за ускорение или доказанную локальную регрессию.

[Результаты и повторы](../build/bench/native-stage4/quiet-isolated-cpu/results.json),
[варианты, команды и hashes](../build/bench/native-stage4/isolated/provenance.json),
[script сборки](../build/bench/native-stage4/make_variants.py).

## Почему native может проигрывать Python

1. Оба языка вызывают один LibTorch. Ускорение языка действует на обвязку,
   а свёртка, autograd, optimizer kernels и MPS queue исполняют тот же код.
   Поэтому выигрыш на коротком reader не превращается в 2,84× на MNIST.
2. Generated native всё ещё использует динамический протокол вызова:
   `sputnik_native_call_value → call_code → NativeFrame`, metadata checks,
   boxed args и atomic retain/release. Full-native означает отсутствие VM
   fallback, но не прямую C++ сигнатуру каждого Sputnik вызова. Empty block и
   method constant действительно уступают Python примерно вдвое.
3. Python здесь — CPython 3.14.6 с adaptive specialization, без JIT и без
   free-threading. Локальный `dis(..., adaptive=True)` после прогрева показал
   `LOAD_ATTR_METHOD_WITH_VALUES` и `CALL_PY_EXACT_ARGS`. Название «Python VM»
   само по себе не задаёт нижнюю границу производительности вызова.
4. Native Tensor result проходит дополнительные объекты: `TensorCore` в
   adapter `wrap`, shared lifetime `RuntimeForeignHandle`, `NativeForeignHandle`
   и Sputnik `Tensor` instance. У последних трёх есть отдельные allocations;
   это видно в generated code, а allocator заметен в sample. Их индивидуальный
   вклад по отдельным allocation counters ещё не измерен.
5. Public API читает logical-task autograd mode и вызывает несколько Sputnik
   wrappers. Это реальные semantics (task inheritance, tracing, lifetime),
   которые требуется сохранять при специализации. ReLU controls показывают,
   что specialized core уже близок к `torch.relu`, а public `x.relu` остаётся
   существенно дороже Python `x.relu()`.

CPU sample длинного public ReLU до этого этапа: активный FFI worker имел
3358 samples, из них 1440 LibTorch, 1778 Sputnik runtime, 137 bridge и 3 waits.
Exclusive горячие места: NativeFrame constructor 98, destructor 68,
call_value 69, call_code 57, lookup_user_method 47; заметны malloc/free.
Это профиль маленького ReLU, а не доли времени полного MNIST.
Ожидание main thread и других workers не прибавляется к активному потоку.
[Полный sample](../build/bench/native-stage4/sample-before-relu_public/native.sample.txt).

## Отброшенный эксперимент

Проверен дополнительный inline thread-local cache `init`/`after_init!`,
заменяющий уже существующий string-key cache для конструкторов.
Пять процессов, общий runtime и extensions, та же проверка всех results:
Generic+Tensor ReLU paired ratio = 1,0094, public ReLU = 1,0033, выигрыши
только 2/5. Практического ускорения не обнаружено; эксперимент не добавлен
в compiler. Нельзя считать текущий constructor lookup доказанной основной
причиной проигрыша.
[Данные](../build/bench/native-stage4/quiet-constructor-cpu/results.json),
[prototype source](../build/bench/native-stage4/isolated/constructor.native.cpp).

## MNIST в спокойных условиях

SmallResNet 19706 параметров; сохранённые IDX 2048/512, seed 7, три эпохи,
четыре intra-op threads, одинаковые initialization/normalization/batches,
packed AdamW Python и `execution_scope=run` Sputnik. MPS CPU fallback = 0.
Один полный process warmup, затем отдельные процессы с вращением lanes.
В таблице время train+eval, без startup/checkpoint.

| Устройство / batch | Native до, с | Native после, с | VM, с | Python, с | Native / Python paired median | Победы |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU / 32, 5 процессов | 2,984973 | 3,035645 | 3,099622 | 3,038178 | 0,99977 | 3/5 |
| MPS / 32, 7 процессов | 0,814719 | 0,820477 | 0,890354 | 0,840621 | 0,98430 | 4/7 |

Native after/before paired median: CPU 1,01874 (1/5 выигрышей),
MPS 1,00304 (3/7). Значит ускорение полного обучения этим набором исправлений
не доказано. На CPU возможную небольшую регрессию нельзя просто отбросить
как шум. Межпроцессный CV lanes составляет примерно 1–2,4%.
Native/Python практически равны на CPU; native слегка быстрее на MPS,
но запаса недостаточно для выигрыша в каждой паре.

[CPU повторы, numerical checks и provenance](../build/bench/native-stage4/quiet-mnist-cpu-32/results.json),
[MPS повторы](../build/bench/native-stage4/quiet-mnist-mps-32/results.json).
Предыдущие дневные `mnist-{cpu,mps}-*` имели конкурирующую задачу и сильные
скачки; они сохранены, но не используются как оценка speedup.

### Финальная сборка с default fix

После последнего изменения вновь собран полный native benchmark. Baseline
этого A/B — сохранённый `mnist-predefaults` с теми же предыдущими native fixes;
сравнение изолирует последнюю итерацию defaults. CPU — пять процессов,
MPS — семь, process warmup и тот же вращающийся порядок четырёх lanes.
Все epoch losses/accuracy/checkpoint проверки прошли.

| Устройство / batch | Native до defaults, с | Финальный native, с | VM, с | Python, с | Native / Python paired median | Победы |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU / 32 | 2,838788 | 2,851548 | 2,912880 | 2,858361 | 0,99614 | 5/5 |
| CPU / 128 | 2,869830 | 2,856911 | 2,897506 | 2,887026 | 0,99134 | 5/5 |
| MPS / 32 | 0,770687 | 0,771376 | 0,861059 | 0,791823 | 0,97504 | 7/7 |
| MPS / 128 | 0,564691 | 0,567253 | 0,574109 | 0,568962 | 0,99653 | 4/7 |

Победы CPU/32 имеют очень маленький запас: худшая native/Python пара =
0,999844. На MPS/32 запас больше, худшая пара = 0,980206. На MPS/128
худшая пара = 1,011832; универсального преимущества native пока нет.

After/before defaults paired medians: CPU/32 1,00322, CPU/128 0,99799,
MPS/32 1,00070, MPS/128 1,00416. На полном MNIST default fix не дал
подтверждённого ускорения; небольшой проигрыш MPS/128 следует сохранить
в результатах, даже при больших улучшениях целевых host calls.
CV большинства финальных lanes около 0,3–1,1%; большой CPU/128 Python
outlier также сохранён. Медианы пар не обязаны равняться отношению медиан.

VM/Python paired medians: CPU/32 1,01650 (0/5), CPU/128 0,99828 (3/5),
MPS/32 1,08231 (0/7), MPS/128 1,00905 (1/7). С batch 128 VM близка к
Python; с batch 32 остаётся заметный MPS разрыв.

[CPU/32](../build/bench/native-stage4/defaults-mnist-cpu-32/results.json),
[CPU/128](../build/bench/native-stage4/defaults-mnist-cpu-128/results.json),
[MPS/32](../build/bench/native-stage4/defaults-mnist-mps-32/results.json),
[MPS/128](../build/bench/native-stage4/defaults-mnist-mps-128/results.json),
[full-native build log](../build/bench/native-stage4/defaults-build-control/build.log).
Финальные compiler/native/source hashes находятся в
[provenance](../build/bench/native-stage4/provenance.json).

## Следующая работа и проверки

Приоритет — измерить и уменьшить число allocations/ownership transitions
Tensor result, затем специализировать доказанные короткие calls и metadata
reads. Отдельно нужен A/B размерности code/frame dispatch: небольшой
статический C++ вызов должен обходить общие invocation/frame checks только
при доказанных semantics. Большие kernels сами по себе не объясняют
2× проигрыш metadata/host controls.

Проверки этого этапа: общие VM/native/extension tests, full-native language
idioms, property override, default/rest/keyword/block forwarding, девять
shaped slots, рост ABI arena более 2048 values, escaping/clearing closures,
все виды scalar literals, свежие mutable defaults и escaping default closure,
вложенные callbacks, exceptions, collected handle aliases/tombstone и lazy
Unicode chars через C ABI. Native cache regression проверяется отдельным
`make test-native-dispatch-cache`. Optional runner отклоняет partial/stale
saved native; его tests не импортируют Torch и не нужны для сборки пакета.
После финальной сборки также прошли все 31 FFI Sputnik и 15 Python cases
на CPU и реальном MPS, включая saved-native lane. Короткие
`final-ffi-smoke-{cpu,mps}` проверяют checksum/provenance, их времена не
используются для оценки производительности.
