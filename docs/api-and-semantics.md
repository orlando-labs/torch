# API и семантика

> Целевой проект. Текущая реализация и ограничения: [implementation-status.md](implementation-status.md).

Все имена ниже — проектируемая поверхность. Это спецификация, не реализация.
Главный источник требования к мутациям — запрос на Sputnik-контракт;
[локальный RFC](../../sputnik-lang/RFC-collection-mutation-conventions-2026-06-13.md)
используется как объяснение правила, с учётом его статуса Draft.

## 1. Mutation contract

Метод без `!` не меняет наблюдаемое состояние receiver-а. Метод с `!` может
его менять. Присваивание, включая индексное, — отдельная явная форма записи.
Суффикс не означает «бросает исключение» или «опасный вариант».

Для ML наблюдаемое состояние включает не только tensor data, но и shape,
strides, device, dtype, requires-grad, gradients, доступность autograd graph
для повторного backward, module buffers/mode, optimizer slots, RNG и DDP
iteration state. Internal allocation/cache bookkeeping не является мутацией
пользовательского значения, но изменение результатов последующих вызовов
через training statistics или RNG является.

Мутации аргументов также не прячутся в обычной функции. Например,
`distributed.all_reduce!(tensor, group: ...)` явно меняет tensor и group
communication state. Эффекты каждого API перечисляют все mutation targets.

Объектный граф может разделяться: `loss.backward!` меняет gradient buffers
связанных leaf tensors; `optimizer.step!` — связанные параметры;
`view.fill!` — общий storage. Это документированные domains мутации.
Bang не означает исключительное владение памятью.

| PyTorch/Ruby операция | Torch | Изменение |
| --- | --- | --- |
| `add`, `add_` | `add`, `add!` | Новый результат / storage receiver-а |
| `copy_`, `zero_`, `fill_` | `copy_from!`, `zero!`, `fill!` | Storage receiver-а |
| `requires_grad_` | `requires_grad!` | Autograd metadata receiver-а |
| `backward` | `backward!` | Связанные leaf gradients и graph lifetime |
| `zero_grad` | `zero_grad!` | Gradients receiver-а/его параметров |
| `optimizer.step` | `optimizer.step!` | Parameters и optimizer slots |
| `scheduler.step` | `scheduler.step!` | Scheduler и связанные learning rates |
| `Module.train/eval` | `train!`, `eval!` | Mode всего module tree |
| `Tensor.to` | `to` | Derived tensor; receiver неизменен |
| `Module.to` | `to` / `to!` | Независимая model copy / перенос существующей модели |
| `load_state_dict` | `load_state_dict!` | Зарегистрированные parameters/buffers |
| `manual_seed`, generator sampling | `rng.seed!`, `rng.randn!` и аналоги | RNG state |
| `register_parameter/buffer/module` | `register_parameter!`, `register_buffer!`, `register_module!` | Registry receiver-а |
| `all_reduce`, `broadcast` | `all_reduce!`, `broadcast!` | Tensor аргумент и protocol state |

Обычные in-place mutators возвращают `self` даже при отсутствии изменений.
Операции, смысл которых — вычислить результат с побочным изменением, имеют
явный иной return contract: `call!` возвращает output, `load_state_dict!`
— LoadReport, sampling — новый tensor, `backward!` — loss receiver,
`step!` — optimizer, `barrier!` — group. `null` не служит признаком
«мутации не произошло».

Нет bare `step`, `backward`, `train`, `eval`, изменяющего `to`, `inplace: true`
на обычной функции или `out:` на обычной tensor operation. В `0.1` out API
вообще не публичный; будущий вариант должен ставить destination receiver-ом
или иметь отдельное bang-имя и перечислять destinations.

## 2. Tensor и значения конфигурации

`Tensor` оборачивает dense strided ATen tensor. `Parameter` — Tensor с
идентичностью в model registry, не другой формат storage. `Device`, `DType`,
`Shape`, `Slice` — обычные проверяемые значения Sputnik.

| Свойство/метод | Контракт |
| --- | --- |
| `shape` | Immutable Tuple[Int], scalar имеет `()` |
| `rank`, `numel`, `strides`, `storage_offset` | Метаданные без materialization данных |
| `dtype`, `device` | DType / Device, не строка с дополнительными методами |
| `contiguous?`, `requires_grad?`, `leaf?`, `view?` | Bool predicates |
| `grad` | Tensor или `null`; live alias gradient buffer, не snapshot |
| `item` | Sputnik scalar при `numel == 1`, независимо от rank; иначе ShapeError |
| `to_a` | Копирование на CPU и вложенные Sputnik arrays; потенциальная синхронизация |
| `inspect` | Краткие metadata; данные только через явно ограниченный `preview` |
| `copy` | Независимый storage, differentiable copy при включённом autograd |
| `snapshot` | Detached независимая копия, read-only средствами пакета |
| `detach` | Новый tensor wrapper без gradient edge, storage общий |

`snapshot` не обещает автоматической shareability foreign handle между
Sputnik tasks: это отдельный runtime contract. Native Core никогда не
экспортируется публично.

`Device` имеет `type`, optional `index`, `to_s`, equality/hash. `cpu`, `mps`
и `mps:0` нормализуются; иной MPS index отклоняется. Разбор `cuda:3` может
быть успешным на Mac: наличие такого device string нужно для checkpoint
metadata. Попытка вычисления/переноса проверяет реальную доступность.

DType явно задаёт размер и численную семантику: float32, float64, float16,
bfloat16, int64, int32, int16, int8, uint8, bool. Базовый float для ML —
float32; integer literals infer int64; Bool infer bool; empty input требует
dtype или использует явно документированный factory default. Ragged и
смешанные нечисловые inputs — ошибка. Mixed numeric inference и promotion
следуют зафиксированным ATen правилам, с тестами scalar/Tensor комбинаций.

Sputnik scalar Int и tensor int64 имеют разные контракты: integer tensor
arithmetic использует ATen fixed-width семантику и не получает автоматически
checked-overflow Sputnik. Граница конверсии scalar → dtype проверяет диапазон;
shape/stride/size arithmetic всегда проверяется на overflow. Это сознательная
уступка эффективности LibTorch, явно отражённая в API, а не обещание полного
числового профиля Sputnik внутри kernels.

Float64 на MPS не преобразуется молча в float32. Поддержка dtype для хранения
не означает поддержку всех ops/gradients. Device + dtype + operation +
forward/backward задают элемент capability matrix.

## 3. Создание, операции, индексация

Factories принимают один shape Tuple/List и keywords `dtype:`, `device:`,
`requires_grad:`. Основные имена: `tensor`, `zeros`, `ones`, `full`, `empty`,
`arange`, `linspace`, `eye`, и семейство `*_like`. `empty` явно означает
неинициализированные данные; используется только там, где сразу следует запись.
`tensor(SputnikArray)` всегда копирует. Zero-copy из временного Sputnik Bytes нет.

Реализованный `from_bytes(bytes, offset: 0, shape: null, device: "cpu")`
копирует payload в принадлежащий Tensor uint8 storage. `shape`, если задана,
должна точно покрывать все bytes после offset. Cast/normalization выполняются
обычными Tensor operations. `index_select(dim, indices)` возвращает новую
выборку, поддерживает повторные indices и autograd; `argmax(dim:, keepdim:)`
возвращает int64 indices, выбирая первую позицию при равных максимумах.

Основной математический словарь сохраняется: arithmetic, `matmul`, `sum`,
`mean`, `min`, `max`, `softmax`, `log_softmax`, `reshape`, `transpose`,
`permute`, `flatten`, `squeeze`, `unsqueeze`, `cat`, `stack`, `split`,
`chunk`, `gather`, `where`, activations. Для осей выбран единый keyword
`dim:` и `keepdim:`; NumPy aliases `axis`/`keepdims` не добавляются.

`+`, `-`, `*`, `/`, `**` возвращают tensors и не изменяют операнды.
Матричное произведение пишется `matmul`; новый оператор для него не нужен.
Rank-zero reductions остаются Tensor, чтобы не обрывать autograd.
`max(dim:)` возвращает именованный result с `values` и `indices`, а не
массив с неочевидными позициями; у result есть deconstruction protocol.

`==` возвращает один Bool: одинаковые dtype/shape и точное поэлементное
равенство после явного требования одинакового device; при несовпадении
device результат false. NaN не равен NaN. На GPU проверка синхронизирует.
Identity сравнивается штатным identity API, approximation — `allclose?`.
Поэлементные сравнения — `eq`, `ne`, `lt`, `le`, `gt`, `ge`, возвращающие
bool Tensor. Mutable Tensor не должен служить value-key; публичные registries
используют stable parameter IDs и строки, а не Tensor keys.

Sputnik считает обычный объект truthy. Поэтому `if tensor` не получает
магического PyTorch bool-conversion и не означает `any`/`all`. Для условий
используются `any?`, `all?`, `allclose?` либо scalar `item`; первые два —
Bool reductions с синхронизацией. Tensor-valued reductions называются
`any` и `all`. Typed/lint диагностика tensor в условии желательна, но не
выдаётся за существующий runtime механизм.

Индексирование принимает Int, Sputnik Range, Slice, bool/int tensor и отдельные
значения `all`, `newaxis`, `ellipsis`. Нормализация диапазонов соответствует
Sputnik, включая endpoint inclusion, negative index и шаг. Если ATen не
представляет отрицательный шаг как view, создаётся результат через явный
index operation; это документируется как copy. Full integer indexing
возвращает rank-zero Tensor; извлечение скаляра — `item`.

`x[index] = value` и `x.set!(index, value)` одинаково явно пишут в base.
Index assignment не реализуется через создание temporary advanced-index
result: используется native index-put. Duplicate indices при scatter/add
могут быть backend-недетерминированными; deterministic mode должен это
диагностировать, а не обещать произвольный порядок.

## 4. Views, aliasing и frozen receivers

По умолчанию сохраняется модель ATen: basic slices, transpose/permute и
совместимые view operations разделяют storage. `reshape` может копировать;
`contiguous` и `to` при отсутствии преобразования могут вернуть alias.
Имена без `!` гарантируют отсутствие изменения при вызове, **не** уникальный
storage результата. `copy` всегда отделяет storage.

`detach` также alias. Безопасное независимое состояние получается через
`snapshot` или `detach.copy`. Autograd version counters и проверки in-place
остаются активными; `.data` escape hatch не экспортируется.

Для каждого wrapper отслеживается read-only/frozen status. Все native write
пути, включая assignment, optimizer, state loading и indirect destinations,
проверяют разрешение на запись. Freeze одного wrapper не делает snapshot
общей памяти и не замораживает произвольные сторонние aliases. Чтобы получить
стабильное read-only значение, используется `snapshot`, создающий свой storage.
Такой контракт проверяется с общим механизмом frozen receivers Sputnik.

Read-only свойство snapshot распространяется на все возвращённые из него
aliases, включая `detach`, slicing и no-op `to`. Иначе через такой alias
можно было бы изменить snapshot. `copy` создаёт новый writable storage.
У frozen trainable leaf нельзя менять и `.grad`: backward/optimizer
проверяют связанные mutation targets, а не только объект, на котором вызван
метод. Native guard этого пути входит в обязательный integration gate.

Expanded/overlapping views не получают запись, если она неоднозначна;
adapter отклоняет её до запуска. `as_strided`, raw storage access, `resize!`
и произвольный pointer import исключены из `0.1`. Mutable methods сначала
валидируют shape/dtype/device, затем выполняют одну native операцию; rollback
после GPU/kernel failure не обещается.

## 5. Autograd

`requires_grad!` допустим только для floating/поддерживаемых complex tensors
и в рамках ATen leaf rules. `backward!(gradient: null, retain_graph: false,
create_graph: false)` на scalar использует единичный gradient. Для non-scalar
нужен shape-compatible gradient. По умолчанию `.grad` накапливается.

`zero_grad!` освобождает ссылку на gradient buffer;
`zero_grad_buffers!` обнуляет существующий buffer без замены storage.
Optimizer различает отсутствующий gradient
и нулевой gradient. `retain_grad!` для non-leaf — явная смена поведения.

Functional `autograd.grad(outputs, inputs, ...)` возвращает gradients, не
записывает `.grad` и **сохраняет** исходный graph для повторного использования.
В нём нет `retain_graph: false`: освобождающий graph вариант называется
`autograd.grad!`. Его bang оправдан изменением дальнейшей пригодности graph,
даже если `.grad` ни у кого не изменился. Стоимость сохранения graph
документируется; обычный training использует `backward!`.

In-place on leaf/view, изменение saved tensor между forward/backward,
повторный backward после освобождения graph — типизированные ошибки,
сохраняющие смысл ATen checks. Никакой автоматической подмены на detach.
`create_graph` поддерживается в обычном autograd для проверенных ops;
в DDP `0.1` higher-order gradients отклоняются.

`no_grad` и `inference_mode` — scoped blocks поверх task-local context из
[архитектуры](architecture.md). Они не меняют `model.training?`. Evaluation
mode и отключение gradient recording — независимые решения. Inference
tensors следуют ограничениям LibTorch; их нельзя незаметно превратить в
обычные trainable tensors.

Sputnik custom backward callbacks и arbitrary gradient hooks не входят в
`0.1`: движок может вызвать их на чужом native thread. Native DDP hooks
работают без перехода обратно в Sputnik.

## 6. Module и композиция моделей

`nn.Module` — Sputnik-класс с ordered registry дочерних modules, parameters и
buffers. Сборка tree выполняется через `register_*!`; краткий DSL можно
реализовать macro expansion в эти методы. Базовый API не зависит от macros,
перехвата каждого ivar assignment или позднего reopening классов.

Имена registry не содержат `.`; state keys используют dot path. Повторное
имя отклоняется, replacement — отдельный явный mutator. Object graph может
иметь tied parameters и shared modules, но не циклы. Traversal детерминирован;
`parameters` deduplicate по стабильной Parameter identity, state dict
сохраняет все именованные пути и alias metadata.

`Sequential`, `ModuleList`, `ModuleDict`, `ParameterList` дают именованную
композицию и обычные Sputnik iteration/block protocols. `forward` реализуется
обычным Sputnik-кодом, комбинирующим differentiable ops. C++ Module subclass
для каждой модели не нужен. Shape contracts можно проверять при входе и
через patterns; compile-time dependent tensor types не обещаются.

| Вызов | Семантика |
| --- | --- |
| `model.call(x)` | Не меняет model/input state; может строить новый autograd graph |
| `model.call!(x)` | Может обновлять model buffers/RNG/DDP state; возвращает output |
| `forward` | Пользовательский hook без мутации receiver-а |
| `forward!` | Пользовательский hook с явной мутацией; вызывается из `call!` |
| `train!`, `eval!` | Рекурсивная смена mode; default нового Module — training |
| `state_dict` | Независимый snapshot параметров и persistent buffers |
| `parameters`, `named_parameters`, `buffers` | Новый контейнер live references, без изменения модели |

`call!` может вызвать обычный `forward`, если тот не мутирует. Обратный
переход запрещён. Stateful training BatchNorm и stochastic training Dropout
требуют `call!`. В eval mode их `call` не меняет состояние. Sequential
проверяет effect requirement всех children **до** первого child call.
Для пользовательского кода контракт задаётся hook-ом и effects; библиотека
не обещает распознать произвольные скрытые мутации динамического user code.

BatchNorm running mean/variance и batch counter — buffers; `call!` меняет их
по проверенной ATen семантике. Dropout расходует generator модели.
Forward не прячет transfer входов на другой device: несовпадение — ошибка,
перенос выполняется явно в data pipeline.

`model.to(device:, dtype:)` создаёт независимую model copy с новыми Parameter
IDs, сохранёнными mode/config/tie topology и без optimizer/DDP associations;
исполняемые методы класса сохраняются. Не копирует live autograd graph и
`.grad`. `to!` переносит текущие registries до привязки optimizer/DDP.
В `0.1` перенос с активной такой привязкой отклоняется: нельзя молча оставить
optimizer на старых TensorImpl. Пользователь создаёт новую training session.
Несerializable external resources в custom Module делают copy unsupported
до предоставления явного copy hook.

## 7. RNG и инициализация

Общего неявно изменяемого random generator в публичном API нет. Stateful
`Generator(seed:, device:)` имеет `rand!`, `randn!`, `randint!`, `randperm!`,
`seed!`, `load_state!`; `state` возвращает независимые bytes/config.
Функциональные factories `rand`/`randn` требуют `seed:` и создают локальный
generator, поэтому повторный вызов с тем же seed не меняет внешнее состояние.

Random-initialized layers принимают `seed:` или immutable initializer
specification с seed. У composed model один root seed порождает seeds по
стабильным именам children; не по порядку случайного hash traversal.
Продвигающий общий generator builder обозначается `build!`.

MPS поддержка explicit native generators проверяется отдельно для каждой
операции. Если selected upstream op использует только global generator,
он не получает видимость task-local RNG. Для `0.1` гарантированный путь —
явный CPU generator и перенос random tensor/mask на MPS; Dropout выражается
через differentiable tensor ops. Native MPS RNG — только после отдельного
теста state isolation. Такой CPU→MPS путь обозначается в diagnostics и
benchmark, не маскируется как MPS-native sampling.

Checkpoint сохраняет states и алгоритм generators. Одинаковый seed не
обещает bitwise equality CPU/MPS или разных upstream versions.

## 8. Optimizers и scheduler

`SGD` (momentum, dampening, nesterov), `Adam`, `AdamW` входят в `0.1`.
Groups заданы Sputnik структурами с явными именами параметров и options.
Параметр не может находиться в двух groups; tied weights обновляются один раз.
State именуется через stable parameter paths/alias topology, не адреса C++.

Первые реализации алгоритмов — Sputnik orchestration поверх ATen ops под
no-grad execution context. Это даёт контролируемые bang-операции и переносимые
state dict. Adam/AdamW используют generic ATen tensor-list primitives и
packed moments при одинаковых dtype/device, независимых storage, наличии
всех gradients и одинаковых счётчиках шагов. Формула и выбор updates остаются
в Sputnik. Missing gradients, mixed dtype и независимые counters поддерживаются
общим путём; shared storage сохраняет последовательный порядок updates.
Параметры и их TensorImpl не заменяются, views и autograd version checks
сохраняются. Не требуется отдельная копия C++ optimizer classes.

`step!` валидирует groups, gradients и association generation до записи.
Нет gradient — update пропускается; при AdamW weight decay для отсутствующего
gradient следует зафиксированной reference-семантике. Closure-based LBFGS
в `0.1` отсутствует. Failed partial step помечает training session dirty;
повторный вызов не выдаётся за безопасный retry.

`clip_grad_norm!` меняет gradients, возвращает прежнюю total norm как Tensor;
`clip_grad_value!` возвращает группу параметров. Scheduler — отдельный объект
с `step!` и независимым state. `StepLR`, `CosineAnnealingLR` и plateau
scheduler покрываются fixtures с восстановлением exact step count.

`torch.backends.execution_batch` исполняет Sputnik-блок на одном blocking
FFI worker в native backend. Вложенные tensor вызовы используют этот worker;
VM исполняет тот же callback через свой ABI. Результат блока, task-local
контекст и исключения сохраняются. Scope не меняет grad/inference/trace mode
и не устанавливает общий LibTorch TLS guard: mode передаётся каждому kernel.
MNIST использует scope для train/eval эпохи, optimizer — для шага.

## 9. Data API

`Dataset` предоставляет `count`/index access; `IterableDataset` — `each`.
`DataLoader` — configuration: batch size, sampler, collate callable,
drop-last, `num_workers: 0`, `prefetch_factor: 2`. `each` создаёт отдельное
состояние итерации и использует Sputnik block conventions. При положительном
`num_workers` index access и collate исполняются в Sputnik `task.spawn` на
нативных тредах, выдача сохраняет sampler order. Одновременно отправлено
не больше `num_workers * prefetch_factor` ещё не выданных batches.

Dataset transformations используют обычные `map`/lazy chains. Tensor batch
создаётся на границе collate: нет FFI вызова на каждый scalar в каждой эпохе.
Nested tuple/map batches сохраняют структуру; conversion policy явная.
`DistributedSampler.indices` функционален: seed + explicit `set_epoch!` +
rank полностью задают порядок. Padding/drop-last объявляются явно.

No-GIL Sputnik позволяет передавать dataset и batches, включая mutable Tensor,
по ссылке без процессов и pickle. Рабочие очереди ссылок защищены Mutex,
каналы несут сигналы готовности. После публикации batch worker его больше
не меняет; shared dataset/collate требуют безопасного параллельного доступа.
Для IterableDataset вызывающая task один раз читает stream и собирает
samples, workers исполняют collate. Исключения сохраняют тип/сообщение,
выход из итерации отменяет и join-ит workers. [Подробный контракт](data-loader.md).
CUDA pin-memory flag на Mac не вводится без подтверждённой пользы/поддержки.
Dataset decoding форматов изображений/аудио относится к будущим пакетам.

## 10. Ошибки

Публичная база `torch.Error < NativeError`; категории: `ShapeError`,
`DTypeError`, `DeviceError`, `DeviceUnavailableError`, `UnsupportedOperation`,
`AutogradError`, `MutationRequiredError`, `ReadOnlyTensorError`,
`CheckpointError`, `StateDictError`, `DistributedError`, `DistributedTimeout`,
`BuildCompatibilityError`. Lifetime/cancellation интегрируются с одноимёнными
runtime errors Sputnik. Errors содержат op, shape/dtype/device и relevant key;
не печатают весь tensor или checkpoint payload.

`UnsupportedCheckpointFeature < CheckpointError` и
`DdpUsageError < DistributedError` используются для ограничений codec и
неверного порядка DDP lifecycle соответственно.
