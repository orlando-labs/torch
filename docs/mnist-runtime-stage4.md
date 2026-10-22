# MNIST: runtime investigation, 2026-10-10

Продолжение [VM Value/dispatch](vm-value-dispatch-profile.md),
[direct method entry](call-fast-profile.md) и
[typed resource results](typed-resource-result-profile.md).
Основной benchmark: `bench/mnist/benchmark.s`, внешний Python reference:
`bench/mnist/python_mnist.py`, runner: `bench/mnist/run.py`.

После переноса в текущий проект MPS batch=32: native быстрее Python по
медиане пар на 2.6% (4/5 побед), VM медленнее на 8.8% (0/5). Приняты две
небольшие runtime/compiler оптимизации; два других варианта отклонены после
проверки на полной модели. Цель «обе lanes быстрее Python» и строгий speed
gate ещё не достигнуты. Изолированные A/B и итоговый рабочий запуск приведены
раздельно ниже.

## Исходное состояние

Apple M4, macOS 26.6.2, LibTorch и PyTorch **2.14.0**, четыре threads.
SmallResNet, 19706 параметров, MNIST 2048 train / 512 test, batch=32,
три эпохи, seed=7, run execution scope, Python packed optimizer.
MPS fallback выключен. Пакет, сборка и language tests не требуют Python;
Python здесь используется только как optional benchmark/reference.

Начальная серия: один process warmup, три measured repeats. Медианы, секунды:

| Фаза | VM | Native | Python |
| --- | ---: | ---: | ---: |
| Process wall | 1.404136 | 1.032310 | 1.523873 |
| IDX + normalization | 0.033034 | 0.024468 | 0.016163 |
| Model + optimizer setup | 0.077959 | 0.077749 | 0.072864 |
| Train | 0.746586 | 0.663513 | 0.689028 |
| Evaluation | 0.085674 | 0.075596 | 0.074006 |
| Train + evaluation | 0.832923 | 0.739285 | 0.765059 |
| Model recreation + checkpoint load | 0.015363 | 0.013097 | 0.011302 |

VM проиграла Python во всех трёх train+evaluation парах, худшее отношение
1.113×. Native выиграл две из трёх; худшее отношение 1.010×. Меньший process
wall связан также со стоимостью импорта и запуска Python и не означает
победы в скорости обучения. Loss/accuracy всех эпох и checkpoint round trip
совпали; final loss 0.97007659, accuracy 80.66%.

[Начальные результаты](../build/bench/mnist-stage4/torch/build/baseline-mps32/results.json)
и [таблица](../build/bench/mnist-stage4/torch/build/baseline-mps32/report.md).

## Где тратится время

В отдельном длинном MPS запуске сняты восьмисекундные macOS samples после
четырёхсекундной задержки. Это диагностические запуски, не throughput.
Категории получены по exclusive stack weights для каждого потока.

| Main thread | VM samples | Python samples |
| --- | ---: | ---: |
| Active language runtime | 627 | 201 |
| Active LibTorch | 497 | 562 |
| Active Metal/MPS | 1062 | 1269 |
| Wait LibTorch | 1216 | 1415 |
| Wait Metal/MPS | 2513 | 2491 |
| Total | 5924 | 5938 |

У VM есть ещё 9 active bridge samples. Суммы являются sample weights,
не замером чистой стоимости каждого слоя. Ожидания и работу разных потоков
нельзя складывать в процент wall time. У native исполняющий поток отличается
от main: в его sample есть 174 active runtime weights, причём 121 без имени
функции. Этого недостаточно для точной функции-лидера native.

У VM видны `Vm::step`, чтение/запись registers, retain/release, выдача и
очистка frames, argument shaping и heap operations. Отдельный обнаруженный
излишек: `acquire_block_vm → sync_runtime_names_to` сравнивает полные vectors
строк и символов даже при повторном использовании неизменившегося child VM.
На этом стеке было 34 active sample weights — около 5% active VM runtime,
но лишь около 0.6% всех samples основного потока. Это локальный участок;
его устранение само по себе не объясняет и не закрывает весь разрыв с Python.

[Raw samples и summaries](../build/bench/mnist-stage4/torch/build/profile-before/results.json),
[per-thread analysis](../build/bench/mnist-stage4/torch/build/profile-before/analysis.log).

## Эксперименты

Все экспериментальные сборки изолированы в `build/bench/mnist-stage4`.
Sputnik sources и Torch sources сняты с текущих dirty working trees;
посторонние изменения сохранены. Для VM A/B отдельно пересобран исходный
compiler теми же flags (`-O2 -DNDEBUG -gline-tables-only`). Saved binaries,
build logs, исходные patch snapshots и hashes сохранены рядом с результатами.
Сборки, correctness tests, profiler и timing runs выполняются последовательно.

### Идентичность private name tables

Parent и pooled child разделяют неизменяемый token только после проверки
содержимого таблиц. Изменение private table меняет token; одинаковый token
разрешает O(1) проверку при следующем входе. Сравнение одного размера
недостаточно: child и parent могут независимо добавить разные symbols.
Shared ownership token исключает повторное использование адреса, пока старую
версию хранит child. Канонические shared name pools сохраняют свой прежний путь.

В отдельном микробенчмарке с 2000 runtime symbols и неконстантным default
argument стоимость вызова упала примерно с 6.39 до 2.51 мкс. Пять процессов,
пять внутренних повторов, 20000 вызовов на повтор; результат проверяется.
Это проверка алгоритмического эффекта на нужном пути, не оценка ускорения
MNIST. Контроль через `Mutex.synchronize` не воспроизводил этот путь и
ускорения не дал.

[Default-argument samples](../build/bench/mnist-stage4/names-default.json),
[source](../build/bench/mnist-stage4/names-default.s),
[Mutex control](../build/bench/mnist-stage4/names-mutex.json).

Ранний MNIST A/B с сохранённым старым binary был отрицательным (1/5 побед).
С заново собранным control — 4/5 побед, median paired after/before 0.962516,
но одна пара замедлилась на 14.7%. Эти серии имеют существенный шум; их нельзя
выдавать за устойчивое ускорение обучения. Финальный кандидат хранит token
в конце объекта VM, сохраняя offsets существующих полей.

[Первый A/B](../build/bench/mnist-stage4/names-mps32/results.json),
[A/B с rebuilt control](../build/bench/mnist-stage4/names-control-mps32/results.json).

### Прямой typed SEND в VM

Эксперимент вызывал совместимый synchronous leaf из разобранного SEND,
передавая один stack argument без vector. Live descriptor, type/lifetime
checks и обычный fallback сохранялись. Language VM/ABI и full-native
call-buffer/trivial-method fixtures прошли.

Однако MNIST проиграл все пять пар: median paired after/before 1.015329;
медианы control 0.906701 s и candidate 0.920599 s. Одна пара дала +23.5%.
Эта ветка исключена из финального кандидата. Устранение операций в одном
месте нельзя считать ускорением всего интерпретатора: код и его размещение
в hot dispatch также меняются. Точная причина общего замедления не изолирована.

[Raw A/B](../build/bench/mnist-stage4/leaf-control-mps32/results.json).

CPU FFI ablation (size=1, 300000 host / 30000 kernel iterations, пять
внутренних повторов и пять процессов) объясняет, почему одного microbench
недостаточно. Здесь after VM — именно отвергнутый typed SEND binary:

| VM case | До, ns/op | После, ns/op | Paired after/before | Победы |
| --- | ---: | ---: | ---: | ---: |
| Loop + captured block | 248.48 | 250.54 | 1.00849 | 0/5 |
| Core numel | 380.40 | 345.01 | 0.90576 | 5/5 |
| Public numel | 417.64 | 426.01 | 1.02034 | 1/5 |
| Execution mode | 518.72 | 523.95 | 1.01009 | 0/5 |
| Public relu | 1884.56 | 1809.15 | 0.96038 | 5/5 |

Public numel уже использует forwarding summary; новый Core fast path
не помогает ему, а общий контрольный цикл стал дороже. Python public relu
в этой серии — 406.45 ns/op. Все строки включают общий loop/block harness.

[FFI results](../build/bench/mnist-stage4/torch/build/ffi-cpu-leaf/results.json)
и [таблица всех controls](../build/bench/mnist-stage4/torch/build/ffi-cpu-leaf/report.md).

### Plain entry для native declarations

Кандидат разрешает существующий exact-positional plain entry для простых
native declarations с type hooks. Декларация по-прежнему вызывает extension
adapter; его проверки остаются. Исчезают временная invocation closure и
общий code-id dispatcher. Sputnik bodies с реальными type hooks продолжают
полное выполнение. Defaults, keywords, rest и clauses сохраняют binder.

Fixture проверяет один call site с обычным объектом и foreign handle,
неподходящий тип аргумента, уничтоженный handle и освобождение результата.

В той же FFI серии native public relu: 568.43 → 568.21 ns/op, median paired
after/before 0.99521, 4/5 побед. Это маленький эффект. Execution-mode control,
алгоритм которого не менялся, дал 1.01463 (1/5 побед), loop control — 1.00954.
Нельзя объявлять этот вариант общим ускорением native по одной выбранной строке.
Generated FFI C++ вырос с 4877221 до 4885468 bytes (0.17%).

## Проверка кандидата на полной модели

Кандидат объединяет version token в конце VM и plain native entries;
typed SEND исключён. Порядок запуска lanes вращается, каждая серия имеет
отдельный process warmup. Время — сумма train и evaluation за три эпохи,
без сборки и startup. Медианы, секунды:

| Режим | VM до | VM кандидат | Native до | Native кандидат | Python |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU, batch=32, 5 пар | 3.002094 | 2.997650 | 2.936147 | 2.920744 | 2.953042 |
| MPS, batch=32, 7 пар | 0.896458 | 0.877432 | 0.778261 | 0.775583 | 0.799065 |
| MPS, batch=128, 5 пар | 0.625441 | 0.606544 | 0.589416 | 0.584113 | 0.585242 |

Парные отношения считаются внутри каждого repeat, поэтому медиана отношений
не обязана совпадать с отношением медиан предыдущей таблицы:

| Режим | VM после/до; победы | Native после/до; победы | VM/Python; победы | Native/Python; победы |
| --- | ---: | ---: | ---: | ---: |
| CPU, batch=32 | 0.998870; 3/5 | 0.993894; 4/5 | 1.020133; 1/5 | 0.988448; 5/5 |
| MPS, batch=32 | 0.993775; 5/7 | 0.992938; 6/7 | 1.084973; 0/7 | 0.956607; 7/7 |
| MPS, batch=128 | 0.942173; 3/5 | 0.991002; 4/5 | 1.029978; 0/5 | 0.997540; 3/5 |

Native устойчиво выиграл у Python в этих сериях с batch=32, но batch=128
близок к паритету. Основная часть преимущества native уже была до этого
изменения: прибавка нового plain entry около 0.6–0.9%, а не весь отрыв от
Python. VM пока не достигла цели: +8.5% на MPS batch=32, +3.0% на MPS
batch=128 и +2.0% на CPU. Для VM batch=128 разброс after/before
0.884–1.054 слишком велик, чтобы трактовать медиану как устойчивые −5.8%.

Численные проверки прошли для каждой эпохи и checkpoint каждой lane.
Batch=32: CPU final test loss 0.96657429, accuracy 81.25%; MPS 0.97007659,
80.66%. Batch=128 MPS: 1.72057793, 49.22%; это другой training workload,
его accuracy нельзя сравнивать с batch=32 как regression.

Raw results: [CPU 32](../build/bench/mnist-stage4/torch/build/candidate-ab-cpu32/results.json),
[MPS 32](../build/bench/mnist-stage4/torch/build/candidate-ab-mps32/results.json),
[MPS 128](../build/bench/mnist-stage4/torch/build/candidate-ab-mps128/results.json).

### Разложение оставшегося разрыва

Отдельный MPS batch=32 diagnostic, три measured repeats. После каждого
этапа выполняется synchronize, поэтому эти суммы нельзя вычитать из обычного
throughput как независимые additive costs. Медианы за три эпохи, секунды:

| Training stage | VM | Native | Python |
| --- | ---: | ---: | ---: |
| Batch selection | 0.010880 | 0.006821 | 0.005560 |
| Transfer | 0.065859 | 0.062674 | 0.063227 |
| Zero grad | 0.002171 | 0.001152 | 0.001447 |
| Forward + loss | 0.319837 | 0.288519 | 0.294628 |
| Backward | 0.366867 | 0.363070 | 0.368090 |
| Optimizer | 0.090478 | 0.080070 | 0.081704 |
| Metrics | 0.033983 | 0.031575 | 0.032767 |

Главный observable excess VM — forward (+25 ms), затем optimizer (+9 ms)
и batch selection (+5 ms). Backward близок к Python. Evaluation VM занимает
0.088518 s против 0.075821 s Python. Это согласуется с расходом интерпретатора
на множество мелких вызовов вокруг одинаковых kernels, а не с общим
замедлением LibTorch backward.

[Stage report](../build/bench/mnist-stage4/torch/build/candidate-profile-mps32/report.md).

Повторный stack sample подтвердил устранение выбранного name-table обхода:
на `sync_runtime_names_to` приходилось 41 active runtime weight из 6525
samples контрольного main thread и 1 из 4679 у кандидата. Sample counts
различаются, поэтому абсолютное уменьшение общего runtime здесь не является
доказательством throughput gain. Общий binder и очистка frames остаются в
профиле. [Повторные samples](../build/bench/mnist-stage4/torch/build/profile-candidate-complete/results.json).

### Дополнительный binder experiment — исключён

Проверен вариант без временного vector позиционных индексов и bitmap
использованных keywords в `shape_method_call`. Существующие VM/extension tests
прошли. В отдельном Sputnik микробенчмарке (100000 calls, пять внутренних
повторов после warmup, пять чередующихся процессов) медианы:

| Вызов | До, ns/op | После, ns/op | Paired after/before; победы |
| --- | ---: | ---: | ---: |
| Обычный positional control | 426.82 | 420.75 | 0.98929; 5/5 |
| Keyword | 743.54 | 615.15 | 0.82174; 5/5 |
| Rest + keyword rest | 1778.18 | 1596.11 | 0.90036; 5/5 |

Но MPS MNIST batch=32 выиграл лишь одну из пяти пар: median paired
after/before **1.012674**, медианы 0.877127 → 0.886334 s. Такой результат
не обосновывает перенос в рабочую VM. Вариант сохранён как эксперимент;
финальные исходники содержат прежний binder. Причина различия между micro
и полной моделью пока не изолирована.

[Micro](../build/bench/mnist-stage4/binding-micro.json),
[MNIST](../build/bench/mnist-stage4/binding-mps32/results.json).

## Перенесённые изменения

- [`runtime/vm.cpp`](../../sputnik-lang/runtime/vm.cpp): identity/version
  private name tables, инвалидируемая до изменения и разделяемая с pooled
  child только после точной синхронизации. Добавлен regression на расходящиеся
  runtime symbols одинакового размера и повторное использование child.
- [`tools/sputnik/main.cpp`](../../sputnik-lang/tools/sputnik/main.cpp):
  plain method entry для подходящих native declarations с type hooks.
  Fixture `native_call_buffers` дополнен смешанными Sputnik/foreign receivers
  и проверками type/lifetime/reclaim на том же call site.
- [`bench/heap/default_name_sync.s`](../../sputnik-lang/bench/heap/default_name_sync.s):
  самостоятельный воспроизводимый microbench без Torch и Python.

Typed SEND и изменённый binder в рабочие исходники не вошли. Алгоритмы
SmallResNet и optimizer, batch construction и математические kernels не менялись.
Сохранён [изолированный patch принятых изменений](../build/bench/mnist-stage4/accepted.patch)
поверх исходного dirty workspace; посторонние изменения не откатывались.

Сборка и language checks после переноса:

```sh
cd ../sputnik-lang
make -j3 build/sputnik build/vm_tests build/sputnik_ext_tests \
  CXXFLAGS='-std=c++17 -Wall -Wextra -Wpedantic -O2 -DNDEBUG -gline-tables-only'
build/vm_tests
build/sputnik_ext_tests
make test-native-call-buffers
```

В изолированном этапе также прошли `test-native-trivial-methods` с проверкой
совпадения VM/native diagnostics. Сохранённые builds используют full native
coverage, без bytecode fallback.

### Итоговая проверка текущего workspace

После переноса прошли `vm_tests`, `sputnik_ext_tests` и
`test-native-call-buffers` в VM и full-native. Затем заново собран и измерен
MNIST из текущего `torch`: в нём за время исследования появились
дополнительные NN/loss exports и kernels. Это end-to-end проверка актуального
workspace, а не чистый A/B одной оптимизации.

```sh
"$PYTHON" bench/mnist/run.py --device mps --epochs 3 --batch-size 32 \
  --train-limit 2048 --test-limit 512 --repeats 5 --warmups 1 \
  --python-optimizer packed --output build/bench/mnist-stage4/current-mps32
```

| Текущие исходники, MPS 32 | VM | Native | Python |
| --- | ---: | ---: | ---: |
| Train + evaluation median, s | 0.876486 | 0.787439 | 0.808653 |
| Median paired time / Python | 1.087522 | 0.973818 | 1.000000 |
| Победы над Python | 0/5 | 4/5 | — |
| Худшее отношение к Python | 1.119966 | 1.024074 | — |

Correctness PASS: loss/accuracy каждой эпохи и checkpoint round trip
совпали, final loss 0.97007659, accuracy 80.66%. Full native coverage PASS,
bytecode fallback и MPS CPU fallback выключены. Строгий speed gate **FAIL**:
обе lanes пока не быстрее Python в каждой паре. Общий exit code runner равен
нулю, поскольку `--require-faster-than-python` не задан; это не performance PASS.

[Итоговый report](../build/bench/mnist-stage4/current-mps32/report.md),
[raw results и hashes](../build/bench/mnist-stage4/current-mps32/results.json),
[native build log](../build/bench/mnist-stage4/current-mps32/build.log).

## Дополнительно найденная ошибка — исправлена

`def intern(text, symbol = text.symbol): symbol` возвращал ID символа,
которого ещё не было в private parent table. Следующее `symbol.str`
заканчивалось `VMError: symbol ref is invalid`. Это воспроизводилось и на
rebuilt исходной VM, и на кандидате; не было вызвано version token.

Исправлено отдельным correctness изменением 2026-10-10: после выполнения
default thunk имена переносятся из child VM в parent **до** обработки
успеха или fault. Это сохраняет и Symbol, опубликованный через изменяемый
объект перед ошибкой. Такой же перенос добавлен на границах module init,
clause и prepared protocol execution, иначе вложенный default мог снова
потерять свои names при выходе через эти helpers. Используется существующий
`merge_runtime_names_from`; проверка version token сохраняется.

Regression [default_runtime_names](../../sputnik-lang/corpus/run/default_runtime_names/source.s)
проверяет positional/keyword/constructor defaults, зависимые и вложенные
defaults, properties, clauses, keyword-spread protocol и стабильность
возвращённых Symbol после повторного использования child VM.
В [`vm_tests.cpp`](../../sputnik-lang/tests/vm_tests.cpp) дополнительно
проверяются fault-side effects, последующий вызов в том же RuntimeWorld и
names, созданные при автоматической инициализации модуля для direct entry.
До исправления новый VM regression падал с `symbol ref is invalid`;
после него полный `vm_tests` и `sputnik_ext_tests` проходят, исходный
воспроизводящий пример возвращает `true`. Corpus-сценарий также проходит
в VM и после сборки с `--require-full-native`; build report подтверждает
`native_full_coverage: true` и `native_vm_independent: true`, вывод обеих
версий совпадает: `PASS default runtime names`, затем `0`.

Приведённые выше performance series выполнены до этого correctness fix;
их executable hashes сохранены и не переименованы в результаты новой VM.

[Минимальный воспроизводящий пример](../build/bench/mnist-stage4/names-regression-default.s).

## Направление дальнейшей работы

1. Сначала уменьшать общий расход VM на instructions/frames/Value, проверяя
   loop, ordinary calls и Tensor wrappers одновременно. Добавление очередного
   случая в большой SEND может ускорить выбранный leaf и замедлить остальной
   dispatch. Нужны отдельные варианты размещения/выделения hot paths и
   измерение на полной модели.
2. Для native отдельно измерить representation/lifetime результата: ресурс
   расширения, tag, общий ownership и Sputnik `Tensor`. В CPU controls разница
   generic Core relu → generic Tensor составляет примерно 60 ns у native и
   550 ns у VM; это controlled contrast всей обёртки, не чистая allocation.
3. Сохранять packed Python reference, равную математику и численные проверки.
   Проверять batch=32 и более крупный batch; преимущество на одном режиме
   не является свойством всех программ на языке.
4. Экспорт runtime names из default thunk исправлен отдельным correctness
   изменением; дальнейшие performance A/B должны использовать свежие hashes
   и заново собранный native executable.

