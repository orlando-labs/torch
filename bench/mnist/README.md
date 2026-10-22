# MNIST: sputnik-vm / sputnik-built / Python

Опциональный бенчмарк `examples/showcase.s`. Python используется как внешний
эталон и инструмент замеров; пакет, `tools/build` и `tools/test` по-прежнему
не зависят от Python. Torchvision и NumPy не нужны.

`python_mnist.py` повторяет SmallResNet, нормализацию, CPU-хранение splits,
`index_select` batches, seed каждого слоя и sampler `seed + epoch`, AdamW,
cosine schedule, sample-weighted loss, test accuracy и checkpoint round trip.
Сеть имеет 19 706 параметров. Общий Sputnik train/eval-модуль используют
showcase и benchmark, поэтому изменения цикла примера попадают в замеры.

Первый замер на Apple M4 и найденный провал AdamW описаны в
[результатах и оптимизациях 2026-10-03–04](../../docs/mnist-benchmark-results.md).
Подробные стеки, атомарные A/B и новые исправления описаны в
[профилировании 2026-10-04](../../docs/mnist-deep-profile.md).

## Запуск

MNIST готовится отдельно: `./tools/fetch-mnist`. Из корня torch:

```sh
# Python от того же Homebrew PyTorch, что и используемый LibTorch.
PYTHON="$(brew --prefix pytorch)/libexec/bin/python"

# Полный MNIST, MPS, 5 эпох, 5 измерений и один прогрев каждого процесса.
"$PYTHON" bench/mnist/run.py --device mps

# CPU: ограниченный split для быстрого отслеживания регрессий.
"$PYTHON" bench/mnist/run.py --device cpu --epochs 3 \
  --train-limit 4096 --test-limit 1024 --repeats 3

# Самостоятельная Python-программа с теми же опциями, что и showcase.
"$PYTHON" bench/mnist/python_mnist.py --device mps --epochs 5
```

Если runner запускается другим Python, передайте `--python /path/to/python`
с установленным PyTorch той же версии. MPS требует доступа Metal.
Автоматический CPU fallback отключён; недоступность MPS завершает запуск.

Runner последовательно собирает native с `--require-full-native`, проверяет
отсутствие VM fallback, исполняет короткий запуск всех трёх реализаций и
подготавливает VM graph/extension caches. Затем запускает отдельные процессы
с прогревом и циклическим изменением порядка. Сборка и подготовка caches
не входят в измерения. `--skip-build` допускается при совпадении hashes
Sputnik/native исходников, компилятора и native executable.
По умолчанию `--execution-scope run` держит весь Sputnik job на одном
blocking worker, чтобы LibTorch не создавал OpenMP pools для каждой эпохи.
`--execution-scope epoch` сохраняет прежний режим для A/B того же бинарника.
Scope записывается в результат и должен совпадать при сравнении baseline.

По умолчанию результат находится в `build/bench/mnist/<UTC timestamp>/`;
`--output PATH` задаёт новый каталог. В нём:

- `report.md`: median по процессам, ratios к Python, min/max и CV;
- `results.json`: все измерения, порядок, настройки, версии, hashes и статистика;
- `raw.json`: промежуточные результаты, сохраняемые и при последующей ошибке;
- `*.log`, `*.pt`: полные логи и checkpoint каждого процесса;
- `build.log`: подтверждение native coverage при сборке.

Runner проверяет размеры splits, параметры сети, batch, seed, число эпох,
потоки, версии LibTorch/PyTorch и метрики каждой эпохи. По умолчанию допустима
абсолютная разница loss до `1e-5`; accuracy должна совпадать. Границы можно
явно изменить через `--loss-atol` и `--accuracy-atol` (accuracy — доля,
не проценты). Несовпадение завершает запуск с ошибкой: итоговая таблица
скорости для непроверенных результатов не создаётся.

## Что измеряется

Основная строка — **Train + evaluation**: цикл примера без импортов,
подготовки данных и checkpoint. Дополнительно показаны train и eval,
первая эпоха и средняя последующих эпох внутри каждого процесса,
загрузка/нормализация IDX, создание модели/optimizer, сохранение,
создание новой модели/загрузка и повторная оценка checkpoint.

Время GPU ограничено `synchronize` на границах фаз. Обычный цикл сохраняет
`loss.item` на каждом batch, как в showcase. **Process wall** включает
Python import, cached VM manifest/extension checks, загрузку библиотек,
вывод и завершение. Разница wall и времени внутри `main` показана как
Startup + teardown. VM запускается через `sputnik run manifest`; этот путь
проверяет caches при каждом старте, поэтому wall нельзя считать чистым
временем интерпретатора.

По умолчанию `--python-optimizer matched` сохраняет исходную поэлементную
формулу Sputnik AdamW, включая выделение новых moments. Sputnik теперь применяет
её через tensor-list primitives и packed moments; формула остаётся той же,
но число вызовов и layout временных tensors различаются. Поэтому это
сравнение эквивалентных программ с оптимизациями пакета, а не изолированный
замер стоимости интерпретаторов. Для сравнения с библиотечной реализацией
PyTorch используйте отдельный запуск:

```sh
"$PYTHON" bench/mnist/run.py --device mps --python-optimizer torch
```

Этот вариант использует `torch.optim.AdamW(foreach=False, fused=False)`
с теми же гиперпараметрами. PyTorch оптимизирует последовательность
обновлений; численные различия допустимы лишь в указанных пределах.
Baseline с другим `python_optimizer` runner отклоняет.

`--python-optimizer packed` применяет те же flat moments, gradient packing
и foreach parameter updates в Python. Это дополнительный контроль стоимости
языка/bridge после оптимизации Sputnik. Reference ограничен MNIST, где каждый
параметр получает gradient на каждом шаге и имеет одинаковые dtype/device;
отсутствующий gradient завершает этот вариант с ошибкой. Он также проходит
обычные epoch/checkpoint проверки. Python-tooling остаётся опциональным.

```sh
"$PYTHON" bench/mnist/run.py --device mps --epochs 3 --batch-size 32 \
  --train-limit 2048 --test-limit 512 --repeats 5 --python-optimizer packed
```

## Поиск провалов

```sh
# Малый batch усиливает относительную стоимость dispatch и optimizer.
"$PYTHON" bench/mnist/run.py --device mps --epochs 3 --batch-size 32 \
  --train-limit 2048 --test-limit 512 --repeats 3 --profile

# CPU profile не требует GPU.
"$PYTHON" bench/mnist/run.py --device cpu --epochs 3 --batch-size 32 \
  --train-limit 2048 --test-limit 512 --repeats 3 --profile

# Сравнение с предыдущим измерением того же workload.
"$PYTHON" bench/mnist/run.py --device cpu --epochs 3 \
  --train-limit 4096 --test-limit 1024 --repeats 3 \
  --baseline build/bench/mnist/PREVIOUS/results.json \
  --regression-threshold 0.15 --fail-on-regression
```

`--profile` отдельно измеряет batch selection/sampler, transfer, zero_grad,
forward + loss, backward, optimizer и получение scalar metrics. На MPS
каждая граница делает synchronization, что меняет overlap и скорость.
Profile служит локализации затрат; его таблицу нельзя смешивать с обычными
throughput-замерами. Часть loop overhead остаётся вне стадий.

Для стоимости отдельных операций и macOS stack samples есть отдельные drivers:

```sh
./tools/build bench/mnist/atoms.s
"$PYTHON" bench/mnist/profile.py --atoms --device mps \
  --output build/bench/mnist/atoms
"$PYTHON" bench/mnist/profile.py --atoms --boundary --device cpu \
  --output build/bench/mnist/boundary
./tools/build bench/mnist/benchmark.s
"$PYTHON" bench/mnist/profile.py --sample --device mps --epochs 80 \
  --output build/bench/mnist/stacks
```

Atomic driver включает loop, dispatch, metadata, preloaded training step,
optimizer, kernel/module forward, преобразование sampler Tensor → Array
и data pipeline. Один прогрев, пять
in-process повторов и checksums; старая VM добавляется через `--before-vm`.
`--boundary` включает `AMT_BOUNDARY_PROFILE=1`: C ABI counts и exclusive
wall/CPU time вызывающего потока. Это диагностический режим с overhead;
CPU время LibTorch workers не входит в calling-thread CPU time.
`--sample` снимает все потоки и создаёт summaries exclusive stack weights
через `sample_report.py`. Ожидания разных потоков нельзя суммировать в
процент wall time. Запускайте профилирование, throughput и сборки раздельно.

`hotspot.s --case array` изолирует `to_a` для 2048 int64 индексов,
а `--case epoch` — полный DataLoader с повторной генерацией порядка. После
`./tools/build bench/mnist/hotspot.s` можно запускать native executable или
его VM manifest, задавая `--iterations` для достаточно длинного `sample`.

Baseline сравнивает median каждой фазы и throughput. Он требует одинаковых
workload, режима profile, optimizer, hardware/backend и файлов MNIST.
Срабатывание порога означает кандидата на регрессию, которого следует
перепроверить с учётом разброса, фоновой нагрузки и состояния GPU.
`--fail-on-regression` возвращает код 2 после сохранения отчёта; ошибки
исполнения/валидации возвращают 1. Без этого флага кандидаты лишь отмечаются.

Чтобы проверять преимущество над Python в каждом измеренном повторе:

```sh
"$PYTHON" bench/mnist/run.py --device mps --epochs 3 --batch-size 32 \
  --train-limit 2048 --test-limit 512 --repeats 5 \
  --require-faster-than-python --minimum-speedup 1.05
```

Проверка сравнивает train + evaluation каждой Sputnik lane с Python того же
повтора, исключая прогревы. Каждый запуск должен быть быстрее более чем
в `minimum-speedup` раз; равенство тоже считается провалом. При провале
runner сохраняет численные результаты, отчёт и медленные пары, затем
возвращает код 2. Эта проверка доступна только без `--profile`.
По умолчанию запас равен 1.0; победа во всех измеренных парах не доказывает
преимущество на других программах, устройствах и размерах batch.

Для исследования изменения обвязки есть optional `compare_calls.py`: он
добавляет сохранённые native и VM binaries к текущим VM/native/Python, проверяет
full-native build stamp и saved build log, numeric metrics/checkpoint, hashes
исходников, executables и MNIST. По умолчанию: три epochs, batch=32,
train=2048/test=512, seed=7, четыре threads, run scope и packed Python optimizer.
Сборку текущего benchmark через `run.py` выполняют перед A/B; snapshots binary
и его build log сохраняют до изменения compiler/runtime.

```sh
"$PYTHON" bench/mnist/compare_calls.py --device mps --python "$PYTHON" \
  --before-native /path/to/saved/mnist --before-build-log /path/to/saved/build.log \
  --before-vm /path/to/saved/sputnik --processes 5 \
  --output build/bench/mnist/call-overhead-ab
```

Для VM-оптимизаций достаточно `--before-vm`; сохранённый native и его log
передаются только вместе, если нужен также native A/B. Текущий full-native
и Python остаются контрольными lanes. `--batch-size` меняет размер батча
во всех lanes и в проверке численного совпадения:

```sh
"$PYTHON" bench/mnist/compare_calls.py --device mps --python "$PYTHON" \
  --before-vm /path/to/saved/sputnik --batch-size 128 --processes 5 \
  --output build/bench/mnist/vm-ab-128
```

Сохранённая VM читает текущий manifest и adapter; native snapshot содержит
свой adapter. Compilation/cache warmup, samples и другие нагрузки должны
закончиться до measured runs. Результаты этой оптимизации:
[обвязка 2026-10-06](../../docs/native-call-overhead-profile.md).

Новая VM-итерация с constructor hook cache и inline tagged Value, её
атомарные A/B, полные CPU/MPS серии и оставшийся разрыв:
[Value и dispatch VM 2026-10-06](../../docs/vm-value-dispatch-profile.md).

Причины медленного `numel`, исправления native frame/metadata, приоритеты
следующей архитектурной работы и свежие CPU/MPS MNIST A/B:
[numel и native 2026-10-08](../../docs/native-numel-profile.md).
Первый typed ABI для metadata calls и отдельный FFI A/B:
[typed leaf 2026-10-08](../../docs/typed-leaf-abi-profile.md).
`numel` используется в setup и метриках батчей; этот FFI результат не доказывает
ускорения train/eval.
TaskLocal и autograd carrier state, проверенные CPU/MPS FFI A/B:
[execution mode 2026-10-08](../../docs/execution-mode-profile.md).

Свежая диагностика проигрыша VM, проверенные и отвергнутые оптимизации,
CPU/MPS A/B с batch=32/128 и разложение времени по этапам:
[runtime investigation 2026-10-10](../../docs/mnist-runtime-stage4.md).

Продолжение: динамический состав работы VM, отбракованные runtime/compiler
эксперименты и проверка PGO для VM и built target:
[runtime investigation 2026-10-11](../../docs/mnist-runtime-stage5.md).
Вызовы, кадры, временные аргументы и повторные измерения после фоновой нагрузки:
[runtime investigation 2026-10-12](../../docs/mnist-runtime-stage6.md).

Проверка heap registry и передачи возвращаемых объектов, включая отдельные
PGO control/candidate builds:
[stage 7, 2026-10-12](../../docs/mnist-runtime-stage7.md).
Воспроизводимая профильная сборка без Python:
[`tools/profile-build`](../../../sputnik-lang/docs/engineering/profile-builds.md).

Проверки optional runner:

```sh
"$PYTHON" -m unittest discover -s bench/mnist -p 'test_*.py'
```
