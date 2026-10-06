# MNIST benchmark: Apple M4, 2026-10-03–04

Исходный замер 3 октября: полный MNIST на MPS показывал отставание VM на
4,8%, native — на 3,4%. На batch 32 отставание достигало 63–66%.
Оптимизация tensor-list операций и execution scopes сократила этот разрыв:
в замере 4 октября native быстрее исходной Python-программы на batch 32
во всех пяти парах. VM на этом workload всё ещё медленнее Python.

Подробные stack samples, 16 атомарных случаев и исправление Tensor → Array
с результатами поздних прогонов описаны в
[отдельном отчёте профилирования](mnist-deep-profile.md).

## Изменения 4 октября

Adam/AdamW остаются алгоритмами на Sputnik. Generic ATen foreach primitives
сокращают число ABI-вызовов; moments и gradients объединяются в flat tensors
при одинаковых dtype/device и counters, наличии всех gradients и независимых
storage. Отсутствующие gradients, mixed dtype и разные counters сохраняют
общий путь; shared storage — последовательный порядок updates. Updates
меняют исходные параметры, сохраняя views, TensorImpl identity и version checks.

Bulk zero_grad освобождает buffers одним вызовом; metadata TensorCore читается
без blocking dispatch; CPU TensorDataset выбирает все tensors одним ABI-вызовом.
Grad/inference/trace mode хранится в одном task-local значении. C ABI больше
не выделяет пустой deque строк в каждом call context и резервирует arena
аргументов; VM пропускает лишние dispatch checks для native handles и inline FFI.
Общий `execution_batch` переносит native train/eval эпоху на один worker,
сохраняя Sputnik callback, task locals и исключения.

`vm_tests`, `sputnik_ext_tests`, optional runner tests (10), комплексные VM и
full-native проверки optimizer/Tensor/NN/ONNX на CPU и MPS прошли. Full-native
проверки запускались с `PATH=/nonexistent`, без Python dylibs и MPS CPU fallback.
В том числе проверены partial gradients, snapshots/resume, nested scopes,
mixed dtype, shared storage и autograd mutation checks.

Пять измерений после одного прогрева, median train + evaluation. Эти строки
относятся к состоянию до последующего ускорения литеральных defaults VM:

| Workload; Python optimizer | sputnik-vm, с | sputnik-built, с | Python, с | VM / Python | built / Python |
| --- | ---: | ---: | ---: | ---: | ---: |
| MPS; 2 048/512; batch 32; 3 эпохи; matched | 1.039 | 0.788 | 0.834 | 1.246× | 0.946× |
| CPU; 4 096/1 024; batch 128; 3 эпохи; matched | 5.924 | 5.839 | 5.870 | 1.009× | 0.995× |
| MPS; 60 000/10 000; batch 128; 5 эпох; matched | 23.877 | 25.437 | 23.614 | 1.011× | 1.077× |
| MPS; 2 048/512; batch 32; 3 эпохи; packed | 1.045 | 0.796 | 0.797 | 1.311× | 0.999× |

На batch 32 native выиграл 5/5 пар, худшее отношение времени к Python —
0.972×. VM выиграла 0/5, худшее отношение — 1.253×. Поэтому общий
`--require-faster-than-python` завершился кодом 2 после сохранения отчёта.
Native преимущество в медиане составляет около 5,4%, но запас в каждой
паре меньше 5%. Все epoch/checkpoint метрики matched-прогона совпали точно.

На CPU native выиграл 5/5 пар, но худшее отношение — 0.999×, то есть
запас меньше 0,1%. VM выиграла 0/5, худшее отношение — 1.021×.
CPU результат близок к паритету и не подтверждает устойчивый запас скорости.

В полном MPS прогоне native выиграл 2/5 пар, VM — 0/5. Время заметно
изменялось по ходу серии: native 23.245–28.153 с (CV 8,0%), VM и Python
CV около 4%. Outliers сохранены; этот прогон не подтверждает устойчивое
преимущество native на полном MNIST. Метрики всех эпох совпали точно,
финальная accuracy 98,69%. Причина разброса отдельно не установлена.

Matched Python сохраняет исходную скалярную последовательность AdamW.
Оптимизированный Sputnik применяет ту же формулу в другом layout и через
tensor-list primitives; это сравнение программ, включая оптимизации пакета.
Штатный `torch.optim.AdamW(foreach=False, fused=False)` тоже проверен на
этом workload, но не прошёл численный допуск: epoch 1 train loss отличается
на 0.0000319, epoch 2 test loss — примерно на 0.0283, accuracy — на 3.125
процентного пункта. Runner остановился на прогреве и не создал speed report.
Результаты выше не доказывают преимущество над штатным PyTorch optimizer.

Дополнительный `--python-optimizer packed` использует те же flat moments,
gradient packing и foreach updates в Python. Метрики всех эпох и checkpoint
совпали точно. На batch 32 native выиграл только 3/5 пар, VM — 0/5.
Это отделяет ускорение алгоритма от оставшихся расходов Sputnik: при одинаковом
packed layout преимущество native над Python находится в пределах разброса,
VM тратит примерно на 31% больше времени.

Следующий отдельный A/B меняет только вычисление литеральных default arguments
в VM: точные `LOADK/LOADNULL/LOADBOOL`, `CLOSE_UPVALUES`, `RETURN` получают
immutable scalar прямо, без lease вложенной VM. Defaults с вычислениями,
side effects и mutable collections продолжают исполняться обычным путём.
VM tests и CPU/MPS/ONNX suite прошли; порядок и эффекты defaults сохранены.
На batch 32 с packed Python пять чередующихся пар дали median VM
1.033902 → 1.015734 с; after/before во всех парах меньше 1, median парного
отношения 0.98693×. Python median — 0.794639 с. Метрики совпали точно.
Это небольшой выигрыш, он не устраняет отставание VM.

Итоговый full-native executable пересобран с этим компилятором, coverage
проверено повторно. На том же MPS batch 32 с packed Python: median loop
VM **1.003352 с**, native **0.787855 с**, Python **0.799783 с**.
VM проиграла 5/5 пар (worst 1.265×), native выиграл 5/5 (worst 0.999×).
CV 1.3% / 0.6% / 0.5%; численные метрики всех эпох совпали точно.
Native запас меньше 0,1% в худшей паре: устойчивое преимущество с запасом
в 5% пока не достигнуто. VM по медиане требует на 25,5% больше времени.

Отдельный итоговый профиль MPS использует packed Python и synchronization
после каждой стадии; его времена нельзя подставлять в throughput-таблицу:

| Стадия; 3 train-эпохи | VM, с | native, с | Python packed, с |
| --- | ---: | ---: | ---: |
| Batch selection | 0.027591 | 0.008894 | 0.005761 |
| Transfer | 0.074633 | 0.065965 | 0.064998 |
| zero_grad | 0.003064 | 0.001493 | 0.001482 |
| Forward + loss | 0.397919 | 0.307370 | 0.339164 |
| Backward | 0.506131 | 0.383978 | 0.434099 |
| Optimizer | 0.116810 | 0.086557 | 0.087734 |
| Metrics | 0.039403 | 0.033858 | 0.035506 |

AdamW больше не доминирует в дополнительном времени. В этом профиле VM
отстаёт также в forward, backward и batch selection. Backward внутри обоих
путей вызывает один и тот же LibTorch kernel, поэтому разницу времени стадии
нельзя целиком приписать вычислению градиентов: в неё входят synchronization
и runtime overhead. Причинность остаточного разрыва требует отдельного A/B.
Разброс profile loop CV до 5,5% сохранён в отчёте.

Локальные артефакты:
[batch 32 report](../build/bench/mnist/opt-epoch-mps-32/report.md),
[полный envelope](../build/bench/mnist/opt-epoch-mps-32/results.json),
[CPU report](../build/bench/mnist/opt-epoch-cpu-4096/report.md),
[CPU envelope](../build/bench/mnist/opt-epoch-cpu-4096/results.json),
[полный MPS report](../build/bench/mnist/opt-epoch-mps-full/report.md),
[полный MPS envelope](../build/bench/mnist/opt-epoch-mps-full/results.json),
[packed Python report](../build/bench/mnist/opt-epoch-mps-32-packed/report.md),
[packed Python envelope](../build/bench/mnist/opt-epoch-mps-32-packed/results.json),
[defaults A/B, hashes и raw samples](../build/bench/mnist/opt-literal-defaults-ab/raw.json),
[итоговый packed report](../build/bench/mnist/opt-literal-mps-32-packed/report.md),
[итоговый packed envelope](../build/bench/mnist/opt-literal-mps-32-packed/results.json),
[итоговый stage profile](../build/bench/mnist/opt-literal-mps-profile-32-packed/report.md),
[отклонённый torch-optimizer прогон](../build/bench/mnist/opt-epoch-mps-32-torch/raw.json).

## Исходные условия и результаты 3 октября

### Условия

Apple M4, 10 CPU cores, arm64, macOS 26.6.2; Homebrew PyTorch/LibTorch
2.14.0, Python 3.14.6. Во всех процессах четыре intra-op потока и десять
interop-потоков. MPS CPU fallback отключён. Native имеет полное покрытие
графа и не использует bytecode fallback.

SmallResNet из `examples/showcase.s`: 19 706 параметров, одинаковые
инициализация слоёв, нормализация, sampler, AdamW и cosine schedule, seed 7.
Python использует `MatchedAdamW`, повторяющий последовательность tensor
операций Sputnik. Стандартный `torch.optim.AdamW` в этих таблицах не измерен.

Для каждого workload выполнен один прогрев всех трёх отдельных процессов
и три измерения с циклической перестановкой порядка. Сборка и подготовка
VM caches исключены. Таблицы показывают медианы **train + evaluation**
за все эпохи, с MPS synchronization на границах фаз. Загрузка данных,
импорты и checkpoint в это время не входят; они измеряются отдельно в
полных отчётах. CV времени цикла в обычных прогонах не превышает 0,65%.

Train/test loss и accuracy каждой эпохи совпали точно между тремя
реализациями и повторами. Checkpoint round trip сохранил метрики.
В полном MPS-прогоне итоговые test loss = 0.04451079100072384,
accuracy = 98,69%.

## Обычные прогоны

| Устройство; train/test; batch; эпохи | sputnik-vm, с | sputnik-built, с | Python, с | VM / Python | built / Python |
| --- | ---: | ---: | ---: | ---: | ---: |
| MPS; 60 000/10 000; 128; 5 | 24.289 | 23.972 | 23.182 | 1.048× | 1.034× |
| CPU; 4 096/1 024; 128; 3 | 6.076 | 6.085 | 5.765 | 1.054× | 1.055× |
| MPS; 2 048/512; 32; 3 | 1.337 | 1.361 | 0.821 | 1.628× | 1.659× |

У короткого MPS workload средняя train-эпоха после первой занимает
0.375 с в VM, 0.390 с в native и 0.209 с в Python. Разрыв сохраняется
после первой эпохи. Native не устраняет его, поэтому смотреть следует
также на общую границу Sputnik/LibTorch и состав операций optimizer.

## Профиль batch 32

Отдельные прогоны используют 2 048 train / 512 test, три эпохи и
synchronization после каждой стадии. Инструментация меняет исполнение
MPS; эти числа служат локализации затрат и не заменяют throughput выше.
В таблице — суммарное время стадии за три train-эпохи.

| Устройство; стадия | sputnik-vm, с | sputnik-built, с | Python, с | VM / Python | built / Python |
| --- | ---: | ---: | ---: | ---: | ---: |
| MPS; forward + loss | 0.346 | 0.345 | 0.292 | 1.18× | 1.18× |
| MPS; backward | 0.370 | 0.362 | 0.363 | 1.02× | 1.00× |
| MPS; AdamW | 0.621 | 0.684 | 0.205 | 3.03× | 3.34× |
| CPU; forward + loss | 0.920 | 0.894 | 0.841 | 1.09× | 1.06× |
| CPU; backward | 1.801 | 1.793 | 1.796 | 1.00× | 1.00× |
| CPU; AdamW | 0.439 | 0.522 | 0.084 | 5.23× | 6.22× |

В MPS-профиле дополнительное время AdamW объясняет примерно 78% разницы
train между VM и Python и 83% между native и Python. На CPU backward
также практически одинаков, а optimizer заметно дороже в Sputnik.
Batch selection и zero_grad тоже имеют дополнительную стоимость;
детальные таблицы находятся в отчётах.

## Следующий эксперимент

Первый кандидат — стоимость blocking FFI dispatch для мелких tensor
операций. В [BuildSputnik.cmake](../cmake/BuildSputnik.cmake) `amt_binary`,
`amt_unary`, `amt_mutate`, `amt_grad`, `amt_meta` и другие символы
помечены blocking. Native runtime для такого вызова копирует аргументы,
ставит задачу в `RuntimeBlockingFfiExecutor` и ждёт `future.get()`.
AdamW делает много таких вызовов на каждый параметр; backward передаёт
более крупную работу одним вызовом LibTorch.

Это гипотеза по профилю и коду, причинность ещё не проверена. Полезный
следующий замер — число и стоимость worker handoffs на optimizer step
и изолированное A/B для безопасного пути мелких операций. Также стоит
измерить boxing, создание tensor handles и проверки optimizer.
Изменение execution policy должно сохранить scheduler, task context и
autograd semantics. Алгоритм optimizer остаётся в Sputnik.

Уточнение при следующем эксперименте: lexical native-function calls в VM
попадают в inline ABI path. Наличие `blocking_symbols` само по себе не
доказывает worker handoff на этом пути. Поэтому исходная гипотеза о
handoff относится прежде всего к native; для VM измеряется стоимость
Sputnik wrappers и количества вызовов на границе. Первые grouped tensor
примитивы ускорили VM-стадию AdamW без изменения её формулы и метрик.

## Воспроизведение и артефакты

Из корня sputniktorch, после подготовки MNIST через `./tools/fetch-mnist`:

```sh
PYTHON="$(brew --prefix pytorch)/libexec/bin/python"

"$PYTHON" bench/mnist/run.py --device mps --epochs 5 --repeats 3

"$PYTHON" bench/mnist/run.py --skip-build --device cpu --epochs 3 \
  --train-limit 4096 --test-limit 1024 --repeats 3

"$PYTHON" bench/mnist/run.py --skip-build --device mps --epochs 3 \
  --train-limit 2048 --test-limit 512 --batch-size 32 --repeats 3

# С тем же workload добавить --profile для профиля MPS или CPU.
```

Сохранённые локальные отчёты и полные envelopes:

| Прогон | Отчёт | JSON с provenance и raw samples |
| --- | --- | --- |
| Полный MPS | [report.md](../build/bench/mnist/mps-full/report.md) | [results.json](../build/bench/mnist/mps-full/results.json) |
| CPU batch 128 | [report.md](../build/bench/mnist/cpu-4096/report.md) | [results.json](../build/bench/mnist/cpu-4096/results.json) |
| MPS batch 32 | [report.md](../build/bench/mnist/mps-batch-32/report.md) | [results.json](../build/bench/mnist/mps-batch-32/results.json) |
| MPS profile | [report.md](../build/bench/mnist/mps-profile-32/report.md) | [results.json](../build/bench/mnist/mps-profile-32/results.json) |
| CPU profile | [report.md](../build/bench/mnist/cpu-profile-32/report.md) | [results.json](../build/bench/mnist/cpu-profile-32/results.json) |

В JSON сохранены hashes исходников, compiler/native/Python executables,
manifest, MNIST и библиотек LibTorch, настройки сборки, backend, окружение
и порядок процессов. Каталоги `build/` — локальные артефакты этих запусков.
Для будущих сравнений runner принимает `--baseline PATH/results.json`;
`--fail-on-regression` возвращает код 2 при превышении заданного порога.
Полные опции и ограничения сравнения описаны в
[bench/mnist/README.md](../bench/mnist/README.md).
