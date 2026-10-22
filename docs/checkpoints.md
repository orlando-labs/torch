# Checkpoint и state dict

> Целевой проект. Текущая реализация и ограничения: [implementation-status.md](implementation-status.md).

Serialization — часть ядра torch. Она не зависит от инициализации
distributed и работает в процессе, где CUDA/Gloo вообще отсутствуют.

## 1. Три разные операции

1. `load(path, map_location: ..., weights_only: true)` читает данные в новые
   objects/tensors, не меняя модель.
2. `model.load_state_dict!(state, strict: true, ...)` записывает weights и
   buffers в уже созданную модель.
3. `checkpoint.restore!(session, path, ...)` согласованно восстанавливает
   модель, optimizer, scheduler, RNG и data progress.

Перенос при чтении не изменяет device существующей модели. Загрузка CPU
state dict в MPS model выполняет проверенный copy в уже размещённые buffers;
`model.to!` — отдельная операция.

Основные входы: `load(path, map_location: "cpu", weights_only: true)`,
`save(value, path, format: ...)`, `model.validate_state_dict(state, ...)`,
`model.load_state_dict!(state, ...)`, `checkpoint.save(session, path)` и
`checkpoint.restore!(session, path, ...)`. `save` имеет filesystem effect,
но не меняет переданное значение/session; расширение `format` можно выводить
из поддержанного file suffix, а не из произвольного object type.

## 2. Форматы первой поставки

| Формат | Чтение | Запись | Ограничения |
| --- | --- | --- | --- |
| `.safetensors` | Да | Да | Dense contiguous logical tensors; нет произвольных objects/optimizer tree |
| Python `torch.save` ZIP checkpoint (`.pt`, `.pth`) | Да, объявленный subset | Да, tensor/primitive tree | Ограниченный codec; не любой Python pickle |
| Sputnik training checkpoint (`.sputnikkpt`) | Да | Да | Versioned manifest + safetensors shards, ниже |
| C++ InputArchive/OutputArchive | Позднее, отдельный codec | Позднее | Не считать тем же форматом, что Python state dict |
| Serialized whole Python/Ruby/Sputnik model | Нет | Нет | Архитектуру создаёт Sputnik-код |
| TorchScript/export program | Отдельный будущий inference API | Нет в `0.1` | Не перегружает `load` |
| Legacy non-ZIP pickle, arbitrary custom classes, sparse/quantized storages | Ошибка с точным format/type | Нет | Конверсия внешним инструментом, не скрытый fallback |

Extension файла только подсказка. Decoder определяет magic/container и
проверяет schema. Нельзя после ошибки ограниченного reader автоматически
переходить на более разрешающий unpickler.

Safetensors выбран как простой interoperable путь обмена весами, но он не
сохраняет произвольный graph shared storages/views. При экспорте logical
tensors материализуются; tied aliases записываются в manifest собственного
checkpoint. [Спецификация safetensors](https://github.com/safetensors/safetensors)

## 3. `map_location`

Default — `"cpu"`. Это намеренное отличие от Python, который по умолчанию
восстанавливает исходные devices. В PyTorch remapping применяется к storages
с исходными location tags; перенос уже загруженных tensors не эквивалентен
этому поведению. [Контракт torch.load](https://docs.pytorch.org/docs/2.14/generated/torch.load.html)

| Значение | Правило |
| --- | --- |
| Device или device string | Все storages направляются на это устройство |
| Map исходный tag → Device/string | Exact mapping; отсутствующий tag сохраняет исходное размещение |
| Block/callable | Один вызов на storage; получает immutable StorageInfo, возвращает Device/string или `null` |
| `:original` | Сохранить исходные tags; недоступный device — ошибка |

`null` callback-а означает original device; это не default CPU. Сам
`map_location: null` не принимается: для original есть явное `:original`.
Строковые tags нормализуются как Device. Callable — Sputnik-ориентированное
отличие: получает metadata (`source_device`, byte size, storage ID,
tensor names и dtype descriptors), а не mutable Python Storage, и не может
вернуть произвольный Tensor. Так сохраняются aliases и прозрачная политика
placement. Ключи Map — полные device tags, не wildcard-шаблоны.

Обязательные сценарии:

| Checkpoint | Mapping | Ожидаемый результат |
| --- | --- | --- |
| CUDA weights | Не указан | CPU, без CUDA initialization |
| CUDA weights | `"mps"` | CPU decode → MPS transfer, без CUDA allocation |
| Несколько CUDA tags | Map каждого tag на CPU/MPS | Placement каждого storage по исходному tag |
| CUDA weights | Пустая Map | Попытка original placement и понятная ошибка на Mac |
| CPU float64 | `"mps"` | DTypeError до transfer, без скрытого cast |
| Shared storage/views | Любой resolver | Resolver один раз; sharing сохраняется в пределах поддержанного codec |

Чтение выполняется в пять стадий:

1. Inspect container и ограниченно декодировать metadata в приватный inert
   manifest: storage IDs, original tags, dtype, offsets, shapes, strides,
   object tree. Не создавать tensors на original devices.
2. Проверить limits, размеры, aliases, разрешённые типы и object references.
3. На Sputnik task вызвать resolver по storage metadata в детерминированном
   порядке. Snapshot mapping не меняется во время load. Resolver exceptions
   завершают load без изменения модели.
4. Прочитать bytes на CPU в owned native memory. Создать целевые storages
   по уже проверенному плану, затем восстановить views с корректными offsets.
5. Построить Sputnik result tree, сохранив repeated references. В случае ошибки
   освободить все временные native resources.

Storage resolver не вызывается из worker thread decoder-а: inspect/read
могут быть blocking native calls, callbacks — отдельная Sputnik стадия.
Нельзя сначала переместить весь checkpoint на один device, а затем потерять
исходные tags при применении Map. Недопустимо повторно читать файл без
ограничений после ошибки device-specific load.

## 4. `weights_only` и Python interoperability

`weights_only: true` — единственный поддержанный режим `0.1`, также default.
`false` даёт `UnsupportedCheckpointFeature` до decoding объектов. Имя
означает tensor/primitive data tree, а не только голый dict весов: optimizer
metadata тоже допускается в разрешённой schema.

Разрешены tensors, bool/int/float/str/null, lists/tuples и mappings с
primitive keys; state dict normalizes names to Str. Python OrderedDict,
parameter/tensor rebuild и state dict metadata реализуются специальными
data constructors ограниченного reader, а не импортом Python modules.
Первый release должен читать fixtures Python state dict и wrapped
`{"model": ..., "optimizer": ..., "epoch": ...}` с dense tensors.

Основная реализация `.pt` reader — ограниченный pickle opcode interpreter
над ZIP record reader с allowlist tensor-rebuild forms и memo handling.
Protocol 2–5 поддерживаются только в явно протестированных формах, которые
выдаёт выбранная версия `torch.save`; неподдержанный opcode/global отклоняется
с offsets/type. Любое обращение к произвольным globals, reducers или imports
запрещено. Перед включением codec нужны отдельный review и fuzz/corrupt fixtures.

ATen/C++ archive utilities можно использовать для контейнера/storage I/O.
`torch::pickle_load` не объявляется автоматически реализацией Python
`weights_only=True`; проверка результата после unpickle этого контракта
не обеспечивает. Исходный `torch-rb 0.25.0/ext/torch/torch.cpp` использует
pickle helpers, а `torch-ddp 0.3.0/lib/torch/ddp/monkey_patch.rb` добавляет
в том числе post-load validation. Этот путь не копируется как safety boundary.
Источники: [torch-rb 0.25.0](https://rubygems.org/gems/torch-rb/versions/0.25.0),
[torch-ddp 0.3.0](https://rubygems.org/gems/torch-ddp/versions/0.3.0).

Ограничения configurable, с конечными defaults: file bytes, expanded bytes,
record count, tensor count/rank/numel, container depth, string length, memo
entries. Shape×dtype и offset/stride calculations проверяются на overflow
и границы storage; ZIP duplicate records/path traversal отклоняются.
Mapped device не может обойти dtype/size validation.

Учитывается byte-order metadata архива; поддерживаемый byte swapping
выполняется до tensor materialization, неизвестный byte order отвергается.
Absence старого metadata record обрабатывается только явно описанной
version policy, а не предположением, что любой producer был little-endian.

Безопасный codec не является защитой от всех возможных bugs native kernels
или исчерпания ресурсов; точное обещание — отсутствие исполнения pickle
program/произвольных object constructors и enforced resource limits.

## 5. StateDict

`StateDict` — ordered mapping имён в detached Tensor snapshots плюс schema
metadata: module versions, parameter/buffer kind, aliases и origin devices.
`model.state_dict` по умолчанию копирует данные. Сохранённое в переменную
состояние не меняется после следующего optimizer step.

Повторяющиеся имена tied parameter сохраняются; один memoized detached clone
на общий storage удерживает topology, включая поддержанные views.
Для диагностики доступен явный `state_dict_view` с live references. Он не
считается snapshot и не передаётся asynchronous writer-у без materialization.

Snapshot снимается при отсутствии конкурирующего training step и после
необходимой device synchronization. Иначе даже набор отдельных правильных
tensor copies мог бы принадлежать разным steps. Session checkpoint API
обеспечивает эту границу; bare `state_dict` требует quiescent model.

`load_state_dict!` default policy:

- `strict: true`: точное множество keys; missing/unexpected — ошибка.
- `strict: false`: допускаются missing/unexpected keys, перечисленные в
  LoadReport; shape/type errors не замалчиваются.
- Shape должна совпадать. Dtype по умолчанию совпадает; разрешённый cast
  задаётся явной policy, никогда не происходит из-за MPS автоматически.
- Destination device задаётся моделью; source tensor может быть на другом
  поддержанном устройстве.
- Используется копирование в существующие parameters/buffers под no-grad.
  Parameter identity и optimizer references сохраняются. `assign: true`
  с заменой objects отсутствует в `0.1`.
- Tied destinations с несколькими source keys проверяются на согласованность
  значений до записи; конфликт — ошибка, не «последний ключ победил».
- Nonpersistent buffers не загружаются; module metadata проходит version
  migration только явно зарегистрированными Sputnik transformations.

Return — `LoadReport`: loaded/missing/unexpected keys, conversions,
alias decisions и warnings. `validate_state_dict` возвращает plan/report
без mutation. После preflight stage allocations/transfers выполняются до
первой destination write насколько возможно. Ошибки schema/shape не могут
оставить частично изменённую модель.

Аппаратный сбой/OOM во время commit нескольких copies может дать частичную
запись: полной транзакционности in-place loading не обещаем. Session становится
dirty и требует восстановления. Для all-or-nothing на уровне приложения
нужно загрузить в отдельную model/session и заменить ссылку после успеха.

DDP prefix `module.` не удаляется эвристически. `state.strip_prefix("module.")`
возвращает преобразованное состояние и проверяет collisions. Собственные
DDP checkpoints используют имена исходной модели без wrapper prefix.

## 6. Полный training checkpoint

`.sputnikkpt` — каталог generation с versioned JSON manifest и одним или
несколькими safetensors shards. Manifest содержит package/upstream identity,
model config identifier, named parameter groups, optimizer hyperparameters и
slots, scheduler, epoch/global_step, sampler cursor/seed, RNG states,
dtype/device policy, distributed world metadata и hashes shards.

Byte states RNG хранятся как uint8 payload. Optimizer slots привязаны к
каноническим именам и alias groups, а не к индексам случайного traversal.
Для model weights и optimizer state допускаются разные placement policies:
moment slots обычно идут на parameter device, scalar counters — туда,
где требует конкретный optimizer. Строка `map_location: "mps"` не должна
необдуманно переносить служебные CPU counters native optimizer-а.

Сохранение: quiescent snapshot → staging generation рядом с destination →
запись и проверка shards → manifest с checksums → atomic publication
указателя `latest` на готовую generation. Смена manifest/pointer atomic в
пределах filesystem; crash durability требует fsync files и directory.
Старую generation не удаляют до публикации новой. Readers не читают staging.
Однофайловые `.pt`/`.safetensors` публикуются через temp + rename.

`restore!` сначала валидирует все компоненты, затем восстанавливает model,
optimizer groups/slots, scheduler, RNG и sampler progress. Exact resume
обещается только на той же поддержанной platform/version/config и при
контролируемых источниках random/nondeterminism. Перенос CPU↔MPS — functional
resume с численной погрешностью, не bitwise replay.

Для foreign optimizer checkpoints нужны adapters SGD/Adam/AdamW, которые
сопоставляют parameter IDs исходного файла с declared parameter order/names
и проверяют shape/groups. Без достаточной информации загружаются weights,
а optimizer restore явно отклоняется; угадывать mapping запрещено.

## 7. Приёмка

Обязательны: C++ LibTorch→Sputnik и Sputnik→C++ LibTorch dense state dict;
совместимость с зафиксированным corpus Python-format checkpoints, без запуска
Python в тестах пакета; настоящий CUDA-tagged fixture→CPU/MPS на машине без
CUDA; разные tags и shared storages; wrapped
dict/OrderedDict metadata; scalar/empty/noncontiguous tensors; strict и
non-strict reports; tied weights; optimizer reference preservation; safe
rejection custom globals; limits/corrupt ZIP; interrupted write; resume
против uninterrupted training на одинаковом backend.

Настоящий CUDA-tagged fixture создаётся/проверяется на CUDA host либо берётся
из зафиксированного trusted test corpus с documented provenance. Подмена
строки `cpu` в архиве не является единственным доказательством interop.
