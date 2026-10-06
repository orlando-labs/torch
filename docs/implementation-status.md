# Реализация: 4 октября 2026

Это development-срез, не завершённый `0.1` из архитектурного плана.
В проекте нет зависимости от Python ни для сборки/генерации, ни для запуска,
ни для тестов. LibTorch используется через C++20 и C ABI Sputnik. Shell/CMake
строят пакет; launcher и генератор тестовых checkpoint написаны на C++.

## Что реализовано

| Область | Текущий контракт |
| --- | --- |
| Сборка | Homebrew LibTorch 2.14.0_1, explicit prefix, C++20 adapter, rpath, native dependency lock; VM и native executable |
| Backend | CPU; настоящий MPS; capability queries и synchronize |
| Tensor | Factory/like family; owned uint8 from_bytes; DType/Device, operators, reductions/argmax, index_select, softmax, cat/stack/where |
| Shape | reshape/transpose/select/permute/flatten/unsqueeze/squeeze/narrow, split/chunk с сохранением alias и read-only policy |
| Aliasing | reshape/transpose/select, detach; независимый differentiable copy и generic deep_copy hook; read-only detached snapshot с защитой aliases |
| Mutation/autograd | add!/sub!/mul!/fill!/zero!/copy_from!, backward!/grad!; nullary zero_grad!/zero_grad_buffers!; functional grad и higher-order derivative; task-local scopes |
| NN | 96 публичных классов в отдельных файлах: CNN, Transformer, RNN/LSTM/GRU, normalization, activations, losses, containers; nn.init и functional |
| Stateful forward | train!/eval!, buffers, BatchNorm running stats и Dropout с явным RNG; call! с preflight обычного call |
| Optimizer | Sputnik SGD с momentum/dampening/Nesterov/decay; Adam/AdamW; clip gradients; dedup TensorImpl; in-memory slots snapshot/restore |
| Schedulers/RNG | StepLR, CosineAnnealingLR, ReduceLROnPlateau; Generator rand!/randn!/randperm! и state/restore |
| Data | Indexed/streaming datasets и DataLoader; native thread workers с bounded prefetch и передачей ссылок; nested collate; vectorized TensorDataset batches; explicit sampler epoch |
| State dict | Parameter + buffer snapshots, strict key/shape/dtype preflight, no-grad copy в существующие identities |
| Файлы | Flat Str→dense Tensor `.pt` ZIP; restricted pickle protocols 2/4, OrderedDict metadata, shared storages/strides/offsets при чтении |
| ONNX | Python-free ATen trace и standard lowering, opsets 18–20, Program bytes/atomic save, именованные inputs/outputs и dynamic_axes; [контракт и границы](onnx.md) |
| Placement | CPU default; `map_location: "mps"` или Map tag→device; проверка всех destinations до materialization |
| Distributed | CPU Gloo/FileStore, SUM all-reduce, broadcast, barrier, explicit close; launcher environment |
| DDP | Explicit synchronize!, backward! с упорядоченным reduction; проверка parameter schema на ranks, global usage vector, averaging по world size |
| Launcher | `sputniktorch-run --nproc N [--timeout seconds] program`; отдельные процессы/группы, rank environment, timeout/failure cleanup |

## Границы serialization

`load` не вызывает `torch::pickle_load` или Python. GLOBAL допускается только
как descriptor ограниченного списка storages, OrderedDict и tensor rebuild.
REDUCE никогда не вызывает произвольную функцию; BUILD принимает только
ограниченную `_metadata` state dict. Bounds, dtype, names, storage descriptors
и destinations проверяются до создания публичного результата.

Пределы: 16 MiB pickle, 2 GiB суммарного decoded ZIP содержимого, до 100000
records/entries, rank ≤ 64 и ограниченные stack/memo/metadata traversals.
Legacy non-ZIP, protocol 5, sparse/quantized/complex tensors, custom objects,
optimizer trees, произвольные reducers и hooks отклоняются. Это ограниченный
начальный codec, ещё без fuzzing и независимого security review.

`save` материализует CPU contiguous tensors и публикует temporary file через
rename в том же каталоге. Alias topology при записи пока не сохраняется;
fsync/durability и полный training checkpoint не реализованы. Название `.pt`
не означает поддержку любых объектов из `torch.save`.

## Ограничения DDP

Сейчас только contiguous dense floating CPU leaf parameters и равные размеры
local batches. MPS training — один процесс. Нет MPS collectives, TCPStore,
all-gather public API, gradient buckets/overlap, no_sync, optimizer generation
guards, elastic/restart protocol или координированного checkpoint всех ranks.
Порядок и schema parameters должны оставаться стабильными. Все ranks должны
войти в одинаковые calls; аварийные сценарии ограничены group timeout и
внешним watchdog launcher. Cancellation текущего Work ждёт его timeout.

## Что остаётся по исходному проекту

Полноценный frozen/native mutation bridge, operator/schema generator,
расширенный indexing, copy модели с сохранением tied topology, optimizer
parameter groups, safetensors и native training session checkpoint,
расширенный map_location callable, `LoadReport`, оставшиеся семейства NN и
standalone relocatable distribution. Изменение `shape`/`strides`
результата (сейчас Array) не изменяет metadata тензора.

Поэтому P1–P7 нельзя считать закрытыми. [Каталог](module-coverage.md)
разделяет реализованную поверхность и оставшиеся портирования;
[Sputnik style](sputnik-style.md) фиксирует идиомы и наблюдаемую мутацию.

## Изменения в Sputnik

- `library_dirs` и `runtime_library_dirs` в native manifest, package
  serialization/digests и обоих link paths; старые digests не меняются при
  отсутствующих полях.
- Явные реэкспорты imported names в emitter и merged graph linker; alias
  сохраняет identity исходной функции/класса.
- Имена native error classes не получают повторный module prefix при merge;
  это необходимо для типизированного rescue.
- Native executable находит тип foreign handle по полному имени модуля;
  обычный Sputnik-конструктор игнорирует результат `init`, даже если последняя
  операция вернула native handle. Это сохраняет wrapper вместо подмены его
  внутренним C++ handle.
- `destroy!` в VM и native executable использует lifecycle owned/collected handle:
  вызывает правильную сигнатуру destructor с живым ABI context, сохраняет
  tombstone и допускает повторное закрытие. Метод-thunk больше не используется
  для вызова destructor. Bare-вызов использует тот же lifecycle, что и
  явный call-tail; обе формы сохраняют ABI деструктора.
- Graph linker ABI/cache version поднята, чтобы старые graphs не скрывали
  исправления. Добавлены focused regression tests.
- Native backend компилирует primitive conversion properties `.str`, `.int`,
  `.float`, `.bool`, `.symbol`, `.array`, `.tuple`, `.set`, `.map` с учётом
  property call-site flags. Array identity сохраняется; `map` с block не
  подменяется преобразованием. Fixture проверяется отдельно в VM/native.
- Добавлено прямое исполнение `P_CHECK_EQ` для literal `case!`, включая
  failure edges в scalar-flow/liveness analysis. Benchmark keyword checks
  не отвергают одноимённый пользовательский метод, например Tensor.mean.
- Array literals со spread (`MAKE_LIST_SPREAD`) тоже исполняются напрямую:
  порядок Array/Tuple/finite Range сохраняется, раскрытие Set в Array
  отвергается. Sputnik splats не заменены ручной конкатенацией ради компилятора.
- Array/Tuple срезы по Range исполняются нативно, включая отрицательные
  границы и open-ended finish. Fixture совмещает такой срез со spread.
- Private classes в merged graph разрешаются по полному имени модуля, даже
  если не экспортированы или имеют одинаковые leaf names. Это нужно для
  пользовательских Dataset subclasses, а не только публичных Torch classes.
- Lookup объявленного native error имеет приоритет перед ordinary class
  resolution. Исключение сохраняет native identity и попадает в typed rescue
  после добавления полного разрешения private classes.
- `fs.read_bytes` исполняется напрямую через общий файловый backend Sputnik:
  capability `fs.read`, `Str`/`Path`, binary Bytes и ограничение размера.
  Это позволяет разбирать MNIST IDX в Sputnik без VM fallback.

Чужие существовавшие изменения в рабочем дереве Sputnik не откатывались.

## Проверки

Матрица `tools/test --mps` проверена 1 октября на указанном Homebrew baseline:
Tensor/serialization/DDP, numerical optimizer steps и restore, все 96
публичных CPU NN-классов, MPS CNN/Transformer/recurrent, task-local scopes
и higher-order grad. [MNIST showcase](mnist-showcase.md) обучает неглубокую
residual CNN: полный train/test split, DataLoader, AdamW и cosine schedule.
На MPS пять эпох дали test accuracy 98.69%; checkpoint round trip сохранил
метрики. Короткий MNIST smoke входит в MPS suite, когда датасет подготовлен.
Логи: `build/full-suite-mnist.log` и заключительный
`build/full-suite-mnist-resumed.log` с повтором affected CPU/VM checks после
исправления bare native destructor и итоговым `PASS sputniktorch test suite`.
Сборка требует `--require-full-native`: тела функций, entry и runtime path
не допускают bytecode fallback. ABI/lifecycle, emitter и graph linker
regression tests Sputnik также прошли.

Bootstrap разрешает также `2.14.1` с тем же Gloo pin; фактические runtime
проверки пока относятся к `2.14.0_1`. MPS fixes `2.14.1` не требуют изменений
текущего Sputnik API: SVD/svdvals/lstsq ещё не входят в allowlist.
[Версии, результаты и границы проверки](build-report.md).

Результаты полного прогона фиксируются в `build/logs/` через `tools/test`.
MPS job запускается с `--mps` на машине с доступом к Metal, Gloo — с доступом
к loopback sockets. Тесты не подменяют эти backends заглушками.

ONNX проверен 2–3 октября отдельно через C++ ONNX Runtime: MNIST graph и
обученный checkpoint с CPU и MPS, 15 operator/composition graphs в CPU VM/native и MPS
native. Dynamic batch проверен на 1/3; MHA/GRU/LSTM пока фиксируют example
shape. Общий FFI bridge повторно прошёл ordinary tensor/autograd/training
регрессии в CPU VM/native и MPS native. [Результаты](build-report.md)
и [границы трассировки](onnx.md) отделены от полного ONNX покрытия.
