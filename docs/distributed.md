# Distributed, DDP и функциональность torch-ddp

> Целевой проект. Текущая реализация и ограничения: [implementation-status.md](implementation-status.md).

## 1. Что переносим из gem

Проверен исходный архив `torch-ddp 0.3.0`, SHA-256
`888747943493c272fcfc8961b952a2cdfb92cea45da191c8287ba411f78c1347`.
В него входят `lib/torch/distributed.rb`, DDP wrapper, `torchrun`,
`ddp/monkey_patch.rb`, native distributed/CUDA leaves и tests.
Источник: [публикация автора в RubyGems](https://rubygems.org/gems/torch-ddp/versions/0.3.0).

Это ориентир функциональности, а не обещание побитово повторить реализацию:

| Возможность gem | Решение torch |
| --- | --- |
| Process group init/destroy, rank/world size | Group object + явные lifecycle methods |
| Gloo/NCCL и detection | Gloo обязателен на Mac; NCCL — Linux/CUDA этап |
| TCPStore, FileStore, HashStore | Встроены в distributed module; HashStore только intra-process |
| Barrier, all-reduce, broadcast, ReduceOp | Встроенные bang APIs, без global singleton requirement |
| DDP parameter synchronization и gradient hooks | Встроенный DDP с определённым порядком synchronization |
| torchrun, ranks, multi-node fixed world, restarts | Собственный `torch-run`, fresh native worker executables |
| `fork_world` / spawn helper | Spawn-based test/helper API; fork после accelerator init запрещён |
| `map_location`, `weights_only` additions | Ядро checkpoint, независимо от distributed imports |
| Device helpers, scalar item | Нормальный Device и проверяемый Tensor.item с первого релиза |
| CUDA set-device/empty-cache helpers | CUDA namespace в CUDA build; честная unavailable ошибка на Mac |
| Conditional monkey patches | Не нужны: единый API и версия пакета |

В gem `mps → gloo` присутствует как default mapping; native collectives
получают tensors непосредственно. Это не доказательство работоспособности
MPS collectives. В torch backend выбирается по подтверждённой паре
transport/device, а не только по такому mapping.

## 2. Матрица платформ

| Compute / communication | Статус проекта |
| --- | --- |
| macOS CPU + Gloo | Обязателен в `0.1`, два и более processes |
| macOS MPS, один training process | Обязателен в `0.1` |
| macOS MPS Tensor напрямую в Gloo | Не поддерживается в `0.1`, ранняя ошибка |
| MPS compute + явные CPU staging collectives | Отдельный experimental этап после `0.1` |
| Linux CPU + Gloo | Следующий portability target |
| Linux CUDA + NCCL | Следующий accelerator/distributed target |

PyTorch указывает CPU/Gloo и CUDA/NCCL как соответствующие базовые сочетания;
distributed при source build на macOS нужно включать явно.
[PyTorch distributed](https://docs.pytorch.org/docs/2.14/distributed.html)

Несколько processes на одном Apple GPU не считаются способом получить
multi-GPU speedup. `nproc-per-node=auto` на Mac не превращает GPU cores в
отдельные devices. DDP CPU используется для проверки корректности и реальных
CPU jobs; single-process MPS — основной Mac training workflow.

## 3. ProcessGroup и Store

`distributed.open(backend: :gloo, rank:, world_size:, store:, timeout:)`
создаёт group; block form закрывает его через `ensure`. `env://` читает
`RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`, `MASTER_PORT` с validation.
Альтернатива — явный TCPStore/FileStore. Неявный process-global default group
не нужен; DDP/collectives получают конкретный group.

Lifecycle: created → connected → closing → closed либо failed.
После failed/closed новые ops запрещены. `close!` идемпотентен, не выполняет
необъявленный barrier. Ownership отделён от borrowed references DDP/Work;
group нельзя разрушить при незавершённой работе без coordinated abort.

`rank`, `world_size`, `backend`, `closed?`, `failed?` — read API.
`barrier!`, `all_reduce!`, `broadcast!` — effectful API. Pure data-вариант
reduction можно выразить через `tensor.copy` + `all_reduce!`; bare
communication calls, скрывающих group mutation, не добавляем.

Stores имеют `set!`, `add!`, `compare_set!`, `get`, `wait` с timeout.
`get`/`wait` могут блокировать, но не меняют пользовательское stored value.
FileStore использует уникальный job path и явную cleanup policy; старый файл
не переиспользуется случайно. HashStore не выдаётся за межпроцессный rendezvous.

`all_reduce!` по умолчанию SUM; AVG реализуется sum + деление там, где backend
не предоставляет нужной операции. Integer AVG не делает скрытого cast.
MIN/MAX/PRODUCT/bitwise разрешаются согласно dtype/backend capability table.
Broadcast проверяет src в rank domain данной group. Один и тот же порядок
collectives обязателен у всех ranks; локальная mutex этого требования не заменяет.

Начальный API синхронен с точки зрения Sputnik caller, но waits идут через
blocking executor. Позднейший Work API добавит `wait!`, `completed?` и
явное удержание buffers; он не обязателен для `0.1`. Destructors не вызывают
collectives. Ошибка/timeout одного rank завершает всю группу workers;
локальный retry одного коллективного вызова запрещён.

## 4. DDP: управляемая граница backward

`nn.DistributedDataParallel` оборачивает модель с заранее размещёнными
параметрами и конкретной process group. Перед привязкой registry стабилен;
добавлять parameters, менять dtype/device или tied topology после этого нельзя.

Привязка — `ddp.attach!(model, group: ...)`: она явно допускает изменение
исходной модели при initial broadcast. Конструктор wrapper-а сам по себе
не синхронизирует и не перемещает переданную модель. `close!` удаляет native
guard hooks, освобождает association и не закрывает чужую group автоматически.

В первой версии принимаем **явный DDP backward**: `ddp.backward!(loss)`.
Он выполняет local autograd, затем gradient synchronization до return.
Это Sputnik API, соответствующий смыслу DDP; буквальное совпадение с
`loss.backward()` PyTorch не требуется. Wrapper `call!` отмечает iteration и
может синхронизировать buffers; после него пользователь вычисляет обычный loss.

Почему так: per-parameter hook, который сразу делает all-reduce, может
встретить разный порядок/набор gradients на разных ranks. Для первого
релиза выбирается детерминированный reducer после local backward. Он проще
для диагностики, не требует Sputnik callbacks из native workers и не зависит
от приватного c10d Reducer ABI. Перекрытие communication с backward — следующий
оптимизационный этап; отдельный per-parameter blocking all-reduce не является
приемлемой заменой ordered reducer.

Порядок training iteration:

1. При attach ranks сравнивают registry fingerprint: names/order/shapes/dtypes,
   requires-grad, tied topology. Затем rank 0 broadcast parameters и buffers.
2. `ddp.call!` до forward синхронизирует persistent buffers при включённой
   policy. После этого вызывает `model.call!` с локальным input batch.
3. `ddp.backward!` запускает local backward в native DDP scope. Native guards
   на связанных leaf parameters разрешают accumulation только через эту
   границу. Bare `loss.backward!` для attached parameters даёт DdpUsageError,
   а не успешное несинхронизированное обучение.
4. В одинаковом порядке собирается presence mask gradients. Ranks получают
   общий usage plan до data collectives.
5. Dense gradients пакуются в buckets по registry order/dtype/device с
   фиксированным byte budget. SUM all-reduce, деление на world size,
   распределение результатов по gradient buffers.
6. DDP отмечает synchronized gradient generation; связанный optimizer
   разрешает `step!` только после этого и на matching parameter generation.

Scope разрешения backward хранится в native DDP session как защищённый
generation token, видимый native hooks на autograd workers, а не как флаг
TLS только вызывающего OS thread. Одновременный backward одной session
запрещён. Token живёт до завершения engine work; failure переводит session
в failed/dirty до следующей операции. Это не требует вызовов Sputnik из hook.

`find_unused_parameters: false` — default: отсутствие ожидаемого trainable
gradient у любого rank диагностируется согласованно после presence exchange.
При `true` отсутствующий локальный gradient становится нулевым вкладом;
если параметр не использован нигде, `.grad` остаётся `null`. Usage collection
выполняется после backward и не требует анализа Sputnik forward graph.
Sparse gradients в DDP `0.1` отклоняются.

`no_sync!` — block scope для локального накопления. Он устанавливает флаг
DDP session и восстанавливает его в `ensure`. Backward внутри scope остаётся
`ddp.backward!`; после выхода следующий synchronized backward включает весь
накопленный gradient. `step!` при unsynchronized generation запрещён;
`ddp.sync_gradients!` даёт явную финальную синхронизацию без нового forward.
Normalization не применяется повторно к уже synchronized contribution;
это проверяется отдельным тестом нескольких microbatches.

В `0.1` ranks должны иметь одинаковое число steps и одинаковый local batch
size для стандартного mean-loss/world-average semantics. Sampler обеспечивает
drop/pad policy. Uneven-input join, произвольные weighted sample counts,
double backward, reentrant backward и dynamic world resize не обещаются.
Post-backward collective timeout покрывает падение/расхождение ranks.

Broadcast BatchNorm buffers не является SyncBatchNorm: статистика batch
остаётся локальной. Eval/inference выполняется на `ddp.module` после
объявленного завершения distributed training, без ненужных collectives.

## 5. Launcher

`torch-run` входит в ту же поставку. Запускает заранее собранный native
worker executable с аргументами; не компилирует проект независимо на каждом
rank и не сериализует Sputnik closures. Spawn-helper для tests принимает путь
к worker/entrypoint и primitive configuration.

Обязательные возможности `0.1`: standalone single-node, fixed multi-node
world, nproc-per-node, node-rank, rendezvous endpoint/job ID, rank environment,
signal forwarding, bounded restart всего local worker set, exit-code
propagation, per-rank logs и завершение оставшихся workers при failure.

На каждом узле запускается launcher с одинаковым job identity и совместимым
worker build fingerprint. Статический rendezvous не называется полноценным
TorchElastic. `max-restarts` работает при прежнем world size; training resume
требует checkpoint, сам restart прогресс не восстанавливает.

Standalone coordinator удерживает выделенный port до передачи Store;
find-free-port с закрытием socket до запуска имеет race и не годится как
единственный release-путь. `RANK` глобален, `LOCAL_RANK` — только внутри
узла; MPS device index не выводится из rank.

Launcher не использует `fork` уже инициализированного LibTorch/MPS process.
Для stop: soft signal, grace timeout, затем hard termination; resources
и child processes не остаются висеть. Отмена Sputnik task не подменяет
управление независимыми OS processes.

## 6. Checkpoint при distributed training

Default save выполняется rank 0 после согласованной step boundary. Остальные
ranks получают result/status публикации и только затем продолжают. На
ошибке записи надо распространить failure, а не оставить остальных в barrier.

Каждый rank сохраняет собственные RNG/sampler states как именованную часть
checkpoint либо отдельный shard. Model/optimizer data при обычном DDP
реплицированы, поэтому достаточно одного canonical экземпляра. Источник
BatchNorm buffers для snapshot определяется rank-0 policy.

Resume с тем же world size восстанавливает per-rank state. При ином world
size weights можно использовать, но exact data/RNG resume отклоняется без
явной repartition policy. Device remapping использует core checkpoint API,
а не distributed-only overload.

## 7. Приёмка и следующий этап

Обязательные tests: два CPU ranks на Mac, сумма/broadcast, параметры после
каждого step совпадают, gradient/weights сопоставимы с одним процессом на
объединённом batch, unused parameter scenarios, accumulation, wrong collective
order timeout, rank failure, restart/resume, teardown без hang.

MPS staging DDP, если понадобится, оформляется отдельной явной policy:
MPS gradient ready → CPU copy → Gloo → MPS copy → optimizer dependency.
Нужны измерения transfer overhead, сохранность dtype и контроль streams;
включать такой путь под именем обычного MPS-native DDP нельзя.
