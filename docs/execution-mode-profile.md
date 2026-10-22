# Autograd context overhead, 2026-10-08

Python/PyTorch и Sputnik используют обычную проверку autograd в LibTorch.
Дополнительная стоимость Sputnik — доставка режима логической задачи на
carrier thread и повторная установка grad/inference/tracing перед kernels.
Режим нельзя просто заменить глобальным TLS cache: задачи мигрируют между
потоками, scopes вкладываются и callbacks могут менять состояние потока.

## Реализация

- Общий `RuntimeTaskContext` сохраняет immutable COW map и прежние GC roots.
  Atomic hint для пустого active map позволяет вернуть default без atomic
  shared_ptr snapshot и hash lookup. Hint публикуется вокруг map mutation;
  inherited contexts и восстановление последнего scoped binding обновляют его.
  Thread-local кэш Values не используется: старые значения не удерживаются
  после overwrite, clear, task completion или cancellation.
- Immediate TaskLocal operations заимствуют указатель у установленного task
  context. `with` в native удерживает отдельный owning context на всё время
  callback и использует его при unwind, включая перенос исполнения.
- VM входит в зарегистрированный TaskLocal handler до цепочки посторонних
  builtin types. Intrinsic path тоже расположен раньше; `get(default:)`
  валидирует и читает keyword за один проход. Embedding handler overrides и
  обычное поведение ошибочных аргументов сохраняются.
- `ExecutionModeGuard` сверяет настоящее C++ TLS и dispatch keys. На совпадении
  не пишет state; одиночный grad toggle не переустанавливает inference state.
  При выходе сравнивает и восстанавливает исходное состояние, даже если его
  поменяло тело вызова. Fast path проверен для LibTorch 2.14; другие версии
  используют прежние полные guards. Bootstrap сейчас принимает 2.14.0/2.14.1;
  фактически проверена установленная 2.14.0.
- `execution_batch` передаёт inherited logical mode и устанавливает grad и
  inference на время синхронного callback. Вложенные kernels при совпадении
  переиспользуют состояние. Trace attachment остаётся per kernel: его mutex
  нельзя держать во время Sputnik callback или ожидания child task. Старый
  one-argument native thunk продолжает приниматься для уже собранных modules.
- При token=0 `ExecutionGuard` не конструирует out-of-line TraceKernelGuard.

Это общие runtime/boundary изменения для VM и native. Compiler не знает ни
Torch mode, ни selectors tensor kernels; модели и algorithms остаются Sputnik.
Public unary по-прежнему читает логический mode: оптимизация не подменяет его
константой и не удаляет task isolation.

## Проверки

C++ test сравнивает новый state с `InferenceMode + AutoGradMode` для всех
комбинаций grad/inference/fw-grad/multithreading/view-replay/layout flags,
частично исключённых Autograd dispatch keys, вложенных режимов, exceptions и
изменения TLS внутри тела. Он проверяет и второй physical thread.
Language task suite покрывает forced worker migration, inheritance/isolation,
dynamic scopes, lifecycle cleanup/cancellation и GC retention. Добавлены empty
map transitions при shadowed scopes и invalid get call shapes.

Torch idioms покрывают no-grad batch, вложенные with-grad/inference batches,
kernel fault, caller-state restoration и отсутствие trace lock на callback.
ONNX batch regression использует настоящий traced kernel.

Пройдены `stdlib_task_tests`, `vm_tests`, `sputnik_ext_tests`, standalone C++
guard test, optional runner tests (5), Torch CPU, idioms, ONNX operators, MPS
и training/data MPS. Torch suites запускались в VM и full-native; настоящее
MPS — с `PYTORCH_ENABLE_MPS_FALLBACK=0`. CPU native binary собран до добавления
совместимости с прежним one-argument batch thunk; сам thunk проверен позднее
в финальных VM/native idioms. Логи лежат в
[`build/bench/context-mode`](../build/bench/context-mode/) и
[`sputnik-lang/build`](../../sputnik-lang/build/).

## Методика A/B

До изменений сохранены compiler/native executable, generated C++, все Sputnik
module и C++ adapter inputs. `--before-vm-manifest` запускает старую VM именно
с сохранёнными module/adapter sources; это сравнение всего context пути.
Без отдельного manifest старый VM executable использовал бы новый adapter и
не показывал бы эффект изменения C++ guard.

Все процессы выполняются последовательно с warmup, rotating lane order и
проверкой checksums, full-native coverage и input/executable hashes. Сборки,
unit tests и samples завершаются до timing series. Python остаётся optional
benchmark/reference, без package/build/test dependency.

## Результаты

Apple M4, arm64, Apple clang 21, LibTorch/PyTorch **2.14.0**, четыре threads.
CPU: size=1, 300000 host / 50000 kernel iterations, пять внутренних повторов
и пять процессов каждой lane. MPS: size=25088, 20000 kernel iterations,
пять повторов и пять процессов. Каждый процесс выполняет полный warmup;
порядок пяти lanes вращается. MPS синхронизируется на границах kernel loop,
поэтому результат отражает throughput подачи kernels вместе с обвязкой,
а не latency отдельного завершённого GPU kernel.

Время — медиана процессных медиан, ns/iteration с общим loop/block harness.
Выигрыш — `1 − median(after / before)` по пяти парным процессам; поэтому он
может немного отличаться от отношения двух медиан в соседних колонках.

| Case | Lane | До, ns | После, ns | Парный выигрыш | Выигравшие пары |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU execution_mode | Native | 69,12 | 57,17 | 17,29% | 5/5 |
| CPU execution_mode | VM | 542,92 | 511,16 | 6,12% | 5/5 |
| CPU public relu | Native | 660,65 | 648,12 | 1,90% | 4/5 |
| CPU public relu | VM | 2306,64 | 2235,91 | 3,31% | 5/5 |
| MPS public relu | Native | 2441,44 | 2412,91 | 1,27% | 4/5 |
| MPS public relu | VM | 4454,73 | 4294,29 | 3,60% | 5/5 |

Чтение default logical mode стало устойчиво дешевле. Выигрыш public relu
меньше: в одной native CPU паре время выросло на 1,82%, в одной MPS паре —
на 1,00%. Это небольшой измеренный эффект, без доказательства отдельного
вклада каждого изменения C++ guard, TaskLocal или dispatch.

Контроли не показывают общего ускорения произвольных calls. Native
`TensorCore.numel` изменился с 49,39 до 50,46 ns (paired +2,86%, 0/5 выигрышей),
`Tensor.numel` — с 74,07 до 74,81 ns (paired +2,02%, 1/5). Эти методы не читают
execution mode и не используют новый guard; причина небольшого роста отдельно
не изолирована. VM public numel: 532,78 → 535,43 ns, paired +0,73%.

Python public relu в той же серии: **407,19 ns CPU** и **2245,18 ns MPS**.
После изменений native всё ещё медленнее него примерно в **1,59× / 1,07×**,
VM — в **5,49× / 1,91×**. Выигрыш этого участка не решает весь разрыв.
CPU specialized Core relu уже занимает 446,54 ns, generic Core — 473,70 ns,
generic Tensor wrapper — 550,12 ns, public — 648,12 ns. Это указывает на
оставшуюся стоимость публичного пути; разности rows не являются независимыми
слагаемыми и не доказывают конкретную долю allocator или method dispatch.

Следующий compiler этап — guarded direct calls и inline коротких forwarding
methods с сохранением method-version guards, overrides и unwind. Для relu
нужно также измерить создание result handle, TensorCore и Sputnik Tensor:
typed metadata ABI само по себе эти allocations не убирает.

Полный MNIST train/eval в этом этапе не замерялся. `numel` действительно есть
в setup и метриках батчей (`examples/mnist/training.s`); прежние отчёты с
формулировкой «только setup» исправлены. FFI series не доказывает ускорение
обучения.

Raw results: [CPU](../build/bench/context-mode/cpu1/results.json),
[MPS](../build/bench/context-mode/mps25088/results.json).
Итоговые hashes, проверка сохранённых inputs, binary/generated C++ snapshots
и test log hashes:
[provenance](../build/bench/context-mode/provenance.json).
