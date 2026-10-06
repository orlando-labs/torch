# План реализации и критерии готовности

Ниже — полная целевая матрица, а не перечень уже реализованных функций.
Начальная реализация и проверенные результаты перечислены в
[implementation-status.md](implementation-status.md). Внешний номер `0.1` назначается
только после полного обязательного цикла; промежуточные P0–P7 — внутренние
milestones, чтобы distributed и checkpoint не исчезли из первой поставки.

## 1. Последовательность

| Этап | Зависит от | Результат | Условие перехода |
| --- | --- | --- | --- |
| P0: LibTorch feasibility | Проект | Один pinned CPU/MPS/Gloo binary, native probes, build report | MPS backward и two-process Gloo реально работают |
| P1: Sputnik integration | P0 | C++20/external artifacts/rpath, handles/errors, contexts, blocking waits | Native executable relocation + lifetime/context tests |
| P2: Tensor core | P1 | Device/DType/Tensor, factories/indexing/ops, mutation/alias manifest | CPU/MPS parity и bang/alias coverage |
| P3: Autograd + Module | P2 | Grad APIs, registry/copy/tied state, call/call!, core layers | MLP/CNN/маленький Transformer forward/backward |
| P4: Serialization | P2; module apply после P3 | PT subset, safetensors, StateDict, map_location, training format | CUDA-tag fixture на Mac, round-trip, predecode validation |
| P5: Training stack | P3/P4 | Optimizers/schedulers/RNG/data, full session resume | MPS training и uninterrupted-vs-resumed comparison |
| P6: Distributed | P1/P3/P5 | Stores/groups/reducer/DDP/launcher, per-rank checkpoints | Two-rank equivalence, failure/timeout/restart tests |
| P7: Release gate | P0–P6 | `0.1`, coverage report, distributable artifacts, user docs | Все обязательные строки матрицы пройдены |

Изолированные work items могут разрабатываться независимо после нужной
границы, но интеграционные gates не пропускаются. Отказ одного backend
не маскируется общей зелёной сборкой CPU.

## 2. P0 — первая реализационная задача

Это следующий шаг после обсуждения проекта. Он намеренно мал по surface API,
но снимает наиболее дорогие архитектурные риски.

1. Зафиксировать upstream SHA/submodules, AppleClang/SDK/CMake и target.
2. Подключить готовый shared LibTorch с CPU, MPS, Gloo без использования Python.
3. Подготовить минимальный C++ consumer для metadata, matmul, backward,
   parameter update и CPU↔MPS transfer.
4. Подготовить two-process Gloo probe; проверить Store transport, exported
   c10d headers/symbols и clean shutdown.
5. Проверить реальные `.pt` fixtures: dense state dict, OrderedDict, CUDA
   location tags, tied storage; зафиксировать forms для restricted decoder.
6. Сделать маленький Sputnik native leaf round-trip Tensor handle → op → result.
   Manual link допустим только как spike, с записью недостающего manifest API.
7. Составить build report и ADR с окончательным selected upstream version.

Stop condition P0: если MPS+Gloo в одной pinned сборке не работают,
сначала исправить recipe/минимальный upstream patch или явно сменить baseline.
Не разворачивать весь NN API поверх непроверенной binary foundation.

## 3. Разбивка work items

| ID | Работа | Acceptance evidence |
| --- | --- | --- |
| A01 | Native external-artifact schema и standard selection | Dependency digests, rebuild на смену flags, link всех build lanes |
| A02 | LibTorch CMake consumer и rpath/resource bundle | Запуск из нового каталога без исходного prefix |
| A03 | TensorCore + typed fault bridge | GC/destroy/views без double-free/use-after-free |
| A04 | ExecutionContext и per-call TLS guards | Две Sputnik tasks с разным grad mode без leakage |
| A05 | Mutation/frozen/copy guards | Bang audit, assignment/view writes и generic copy hooks |
| T01 | Device/DType/factory normalization | Scalars, empty/ragged inputs, checked shape arithmetic |
| T02 | Schema generator + curated ops manifest | Reproducible generation; все write aliases маркированы |
| T03 | Index/view/transfer semantics | Shared-storage, overlap rejection, no silent dtype/device cast |
| G01 | Autograd + grad/no-grad scopes | Finite differences, accumulation, graph reuse/consumption |
| N01 | Module registries, copy, tied parameters | Stable names/order, no duplicate update, alias topology |
| N02 | Core layers и stateful forward contracts | BN/Dropout call error до partial Sequential execution |
| C01 | Safe `.pt` reader/writer subset | C++ LibTorch round-trip, no arbitrary globals, limits/fuzz |
| C02 | Safetensors и placement planner | Correct source-tag mapping, CPU-first CUDA fixture |
| C03 | StateDict validate/apply + LoadReport | Strict mismatch без изменения модели, optimizer identity intact |
| C04 | Checkpoint generation/publish/restore | Interrupted write, checksum failure, exact supported resume |
| O01 | SGD/Adam/AdamW + scheduler | Reference steps, parameter groups, missing gradients |
| R01 | Explicit RNG + initializers | State snapshot/restore, независимость model generators |
| D01 | Dataset/Loader/Sampler | Deterministic epochs/ranks, tail/padding policy, bounded prefetch |
| P01 | Gloo stores/groups/collectives | Two workers, timeout, rank errors и teardown |
| P02 | Ordered post-backward reducer + DDP guards | Single-vs-distributed gradient equivalence |
| P03 | Launcher + restart protocol | Signals, failed rank cleanup, fixed multi-node config validation |
| P04 | Per-rank checkpoint coordination | Нет зависания peers при rank-0 I/O error |
| Q01 | Release matrix и developer docs | Чистый build/install/run в поддержанной среде |

## 4. Функциональная матрица первой версии

Обозначение «обязательно» означает release gate, а не нынешнюю реализацию.
Каждый operator entry разделяет CPU/MPS и forward/backward; необязательные
комбинации отклоняются явно.

| Область | Обязательно для `0.1` | Позднее |
| --- | --- | --- |
| Devices | CPU, один MPS device, capability report | CUDA/NCCL, дополнительные accelerators |
| Dtypes | CPU bool/int8/uint8/int16/int32/int64/float32/float64; MPS bool/int32/int64/float32 для заявленных ops | Широкие fp16/bf16, complex, quantized, float8 |
| Factories | tensor, zeros/ones/full/empty, arange/linspace/eye, like family, seeded random | Расширенные distributions |
| Tensor math | Broadcasting arithmetic, unary math, reductions, matmul/bmm, cat/stack/split, where, gather/scatter по allowlist | Полный ATen surface, advanced linalg/FFT/sparse |
| Shape/index | Basic и нужный ML advanced indexing, reshape/view/transpose/permute/flatten | Unsafe as_strided/raw storage/resize |
| Autograd | backward!, grad/grad!, accumulation, leaf rules, no_grad/inference_mode | Sputnik custom autograd.Function, arbitrary hooks |
| Containers | Module, Sequential, ModuleList/Dict, ParameterList, tied weights | Lazy/meta construction и tracing-driven registry |
| Dense NN | Linear, Embedding dense gradients, ReLU/GELU/SiLU, Dropout, Flatten | Sparse embedding gradients в DDP |
| CNN | Conv1d/2d, max/avg/adaptive pooling, BatchNorm1d/2d | Полный набор 3D/transposed вариантов |
| Transformer primitives | LayerNorm, RMSNorm, softmax/log_softmax, attention с explicit mask/causal policy | Fused SDPA backend-specific paths и большой catalog готовых моделей |
| Loss | MSE/L1, cross entropy/NLL, BCE-with-logits, reduction modes | Все специализированные loss functions |
| Training | SGD/Adam/AdamW, clip grads, StepLR/Cosine/plateau, explicit RNG | LBFGS/closure optimizers, fused optimizers, AMP/GradScaler |
| Data | Dataset/IterableDataset, DataLoader с native tasks и bounded prefetch по ссылке, collate, distributed sampler | Универсальные media codecs, weighted sampler, pin-memory |
| Serialization | Safe dense `.pt` subset, safetensors, state dict, map_location, full native training checkpoint | Arbitrary pickle/whole-model load не планируются; sharded foreign formats отдельно |
| Distributed | Gloo CPU, stores, collectives, DDP, fixed-world launcher/restarts | MPS staging, NCCL, communication overlap, elastic world, FSDP |
| Interop | Sputnik Array/Bytes copy, Python fixture compatibility | Nsputnik zero-copy/DLPack после lifetime/stream contract |

Attention baseline составляется из matmul, scaling, masking и softmax в Sputnik
над native ops. Он обеспечивает понятную проверяемую семантику без требования
fused kernel на каждом MPS build. PyTorch `torch.compile` не получается
автоматически из того, что сам Sputnik компилируется в native code.

## 5. Проверки семантики

### Tensor и mutation

Для каждой публичной операции фиксируются value result, alias result,
mutation targets и exception behavior. Non-bang вызов сравнивает data,
metadata, versions и gradients receiver-а до/после. Bang проверяется на
ordinary/frozen/aliased/noncontiguous/overlapping storage и leaf/non-leaf.

Специальные regression cases: `detach` share, `copy` independent,
`to` no-op alias, differentiable clone, advanced-index write, negative-step
copy, singleton `item`, scalar rank-zero, empty dimensions, huge shape
overflow, missing gradient vs zero gradient. Tensor truthiness соответствует
Sputnik; не тестировать несуществующий Python bool hook.

### Autograd и NN

CPU float64 finite-difference gradcheck для differentiable primitives;
float32 CPU/MPS reference comparisons с per-op tolerances. Проверяется
ошибка изменённого saved tensor, freed graph, retained graph и second-order
на subset ops. Разные task contexts, nesting scopes, exception/cancellation
и blocking executor migration не смешивают grad mode.

BN running statistics и Dropout RNG изменяются только через разрешённые
calls. Eval и no-grad независимы. Copy модели сохраняет values/config/ties,
но меняет parameter identities; `to!` после optimizer attach отвергается.

### Checkpoint

Тесты пакета запускаются без Python. Fixtures создаются C++-генератором
или поставляются как проверенные бинарные данные внешнего происхождения.
Metadata фиксирует generator/source hash, framework version, devices и
artifact hash. Сравниваются logical values, dtype, shape и заявленный alias
contract; не предполагается одинаковый бинарный файл.

Отдельные негативные cases: malicious GLOBAL/REDUCE, unknown rebuild forms,
oversized memo/container, bad storage bounds, ZIP duplicate records, corrupt
shard, unknown schema, prefix collision, tied-weight conflict, partial
mapping с недоступным original device. Проверка safety выполняется до
публичной materialization результатов.

### Distributed

Reference loss/gradient на concatenated batch сопоставляется с DDP на равных
local batches. Проверяются SUM/AVG normalization, unused gradients,
microbatch accumulation, tied parameters, different registration order,
buffer sync до forward, no_sync restoration, stale optimizer generation.

Fault injection: rank выходит до init/во время forward/во время collective,
rank 0 не может записать checkpoint, порт занят, wrong world size,
неверный hostname, SIGTERM, истёкший timeout. Каждый test имеет внешний
watchdog, чтобы ошибка реализации не повесила CI навсегда.

## 6. Численная и производительная приёмка

CPU float32, CPU float64 и MPS float32 — отдельные jobs. Условия tolerance
зависят от операции и magnitudes, фиксируются в test manifest. Допуск не
повышается автоматически после failure. Для одинакового MPS/CPU результата
не обещается bitwise identity.

Минимальные end-to-end workloads:

1. Линейная регрессия/MLP: корректный градиент и снижение deterministic loss.
2. Маленький CNN с BatchNorm/Dropout: stateful train + eval + checkpoint.
3. Маленький causal Transformer: Embedding, normalization, attention mask,
   cross entropy, несколько optimizer steps.
4. Тот же training session с save/resume в середине против непрерывного run.
5. CPU DDP 2 ranks против one-process reference.

Performance report разделяет native binding overhead, bulk op latency,
tensor conversion, data loading и end-to-end training. Измеряет CPU и MPS
после warmup; GPU timing содержит явные sync boundaries. Цель — основное
время больших ops внутри LibTorch, без scalar-by-scalar Sputnik/FFI loops.
Фиксированного процента от Python PyTorch до измерений не обещаем.

Memory tests отслеживают удержанные graphs, tensor/view lifetime, snapshots,
optimizer slots, MPS allocated/reserved memory и многократный train/load
цикл. `empty_cache!` не используется для сокрытия утечек живых tensors.

## 7. CI, artifacts и воспроизводимость

CPU CI выполняется на каждом изменении; настоящий MPS runner необходим для
MPS release gate. Skip на машине без GPU не считается pass MPS coverage.
Gloo loopback job запускает реальные отдельные executables. Дополнительная
чистая macOS среда проверяет binary relocation и runtime resources.

Pinned LibTorch строится/кэшируется по native lock, а не для каждого test.
Изменения operator schemas, compiler/SDK, cxxflags, native sources,
dependency paths/identity или Metal resources инвалидируют нужный cache.
Upstream upgrade — отдельная задача со сравнением coverage/ABI/serialization.

Планируемая структура после начала реализации:

| Каталог | Содержание |
| --- | --- |
| `src/` | Публичные Sputnik modules и private declarations |
| `native/` | C++20 adapter, ABI bridge, version-specific upstream access |
| `codegen/` | Generator и curated schema manifest |
| `deps/` | Recipe/lock metadata; большие build trees вне source package |
| `tests/` | Семантика, native lifecycle, MPS, distributed, fixtures |
| `examples/` | Training/inference/checkpoint/distributed workflows |
| `docs/` | Этот проект, API reference и compatibility reports |

Source tree, CMake/shell build и интеграционные тесты уже созданы. Остальные
пункты таблицы вводятся по мере реализации; см. текущую матрицу статуса.

## 8. Риски и принятые ответы

| Риск | Решение / обязательное доказательство |
| --- | --- |
| Несовпадение C++ ABI / toolchain | Точный native lock, один LibTorch, C ABI с Sputnik |
| MPS source build формально прошёл, kernels отсутствуют | Реальный forward/backward и relocated Metal resource test |
| Gloo отсутствует/не линкуется на Mac | Explicit distributed/libuv build и two-process P0 probe |
| Native TLS расходится с Sputnik tasks | Per-call context snapshot, обязательный runtime integration |
| Silent mutation через view/gradient/module | Schema aliases + hand-reviewed non-tensor state + mutation tests |
| Decoder сложнее ожидаемого | Ограниченный поддержанный subset, ранний P0 fixture audit, отдельный review |
| Checkpoint портит optimizer references | Copy into stable parameters, strict preflight, no implicit assignment |
| DDP виснет на порядке hooks | Ordered post-backward reducer и global usage plan |
| First release разрастается | Curated feature matrix; дальнейший ATen catalog и optimizations deferred |
| Generic Sputnik copy/freeze обходит adapter | Explicit hooks/guards и integration tests до NN release |

Вопросы P0 — технические проверки, а не недостающие требования пользователя:
точный toolchain/OS floor, exported c10d symbols, расположение Metal resources,
usable MPS RNG APIs и точные Python tensor rebuild forms. Для каждого уже
определён gate и допустимый ответ; неизвестное не превращается в скрытое
предположение в публичном API.

## 9. Что считается завершённой `0.1`

Один пакет собирается на Mac, запускается вне build tree, обучает модели на
MPS, сохраняет/восстанавливает session, читает поддержанные Python checkpoints
с CUDA tags без CUDA, и запускает CPU DDP через собственный launcher.
Его non-bang/bang/alias contracts подтверждены tests. Пользователь не
устанавливает отдельный DDP package ради загрузки весов.

Release включает известные ограничения, exact versions, ops capability
matrix и результаты P0–P7. Раздел «позднее» не выдаётся за реализованные
возможности, а experimental MPS distributed не подменяет CPU/Gloo gate.
