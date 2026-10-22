# Typed resource results, 2026-10-09

Этот этап продолжает [оптимизацию вызовов](call-fast-profile.md) и
[typed numel](typed-leaf-abi-profile.md). Публичный `Tensor.relu` теперь вызывает
`TensorCore.relu(execution_mode())`, затем создаёт прежнюю Sputnik-обёртку
`Tensor`. C++ остаётся границей LibTorch; композиция остаётся в Sputnik.

## Результат и ABI

Свежий foreign handle использует одну общую allocation для runtime resource,
shared ownership control block и tagged VM tail box либо native wrapper.
Ресурс `TensorCore` адаптера остаётся отдельным. Копии, экспортированный shared
alias, reimport и общий tombstone сохраняют прежнюю семантику. Перед adoption
host готовит metadata и arena slot с null pointer и borrowed ownership;
отказ оставляет ресурс расширению. Unknown tag теперь возвращает null handle.
[Контракт хранения](../../sputnik-lang/docs/engineering/foreign-result-storage.md).

Общий classifier коротких тел теперь распознаёт bodiless `init`, где
`CLOSE_UPVALUES` предшествует синтетическому `LOADNULL`. Раньше fast path не
активировался. Exact positional auto-assigns на ivars выполняются без callee
register frame после проверки всего mapping. Defaults, captures, keywords,
watched VM state и сложные тела сохраняют binder. `after_init!` выполняется.

Новый optional ABI v2 `HANDLE_INT_HANDLE` передаёт borrowed receiver pointer и
raw `int64_t`, возвращая свежий collected resource pointer того же типа.
Обе реализации host проверяют tag/liveness и reclaim policy, готовят результат
и принимают его после успешного возврата. `SputnikCtx`, opaque argument/result
handles, arena и reader/builder callbacks на этом пути не нужны. Fault callback
только копирует diagnostic; после завершения leaf host сначала освобождает
непринятый output, затем передаёт ошибку Sputnik caller.
[Контракт ABI v2](../../sputnik-lang/docs/engineering/typed-leaf-abi-v2.md).

V1 layout и offsets сохранены. Loader предпочитает v2, затем v1; отсутствующий
sidecar сохраняет ordinary thunk. Старая VM проверена с новым extension header:
она игнорирует v2 export и получает правильный результат через ordinary ABI.
Leaf не допускает callbacks/reentry, suspension, blocking waits и C++ exception
через C boundary. Неподходящий call shape сохраняет ordinary ABI diagnostics.

Для `relu` обе ABI используют один C++ helper с прежним `ExecutionGuard`:
autograd, inference и ONNX tracing сохраняются; результат независим и writable,
в том числе при read-only snapshot на входе. Compiler/runtime не знают Torch
selector или `TensorCore` layout.

## A/B typed relu

Before уже содержит coallocation и исправленный empty constructor fast path.
После него изменены только SDK `sputnik_ext.{h,cpp}`, `sputnik_ext_runtime.h`,
generated host в `tools/sputnik/main.cpp`, Torch core declaration, публичный
wrapper и `native/tensor/tensor.cpp`. Snapshot benchmark inputs исключает
параллельные изменения indexing. Before compiler/SDK, adapter sources,
manifest, full-native executable и build log сохранены отдельно.

Apple M4, arm64, LibTorch/PyTorch **2.14.0**, четыре threads. CPU size=1,
100000 host / 30000 kernel iterations; пять внутренних повторов и пять
processes каждой lane. MPS size=25088, 10000 kernel iterations, те же повторы.
Lanes выполнялись последовательно с rotating order и отдельным прогревом.
Checksums, source/executable hashes и full native coverage проверены; bytecode
fallback отсутствует. Сборки и тесты не выполнялись во время измерений.
Python используется только как optional benchmark reference.

Таблица показывает медианы процессных медиан в ns/iteration с одинаковым
loop/captured-block harness. Эти числа включают kernel, dispatch, wrappers и
создание результата; это не чистое время ABI.

| CPU case | Native до | Native после | VM до | VM после | Python |
| --- | ---: | ---: | ---: | ---: | ---: |
| loop + captured block | 22.37 | 22.52 | 248.85 | 247.40 | 14.35 |
| Core numel | 44.87 | 45.12 | 384.99 | 386.51 | 41.92 |
| public numel | 54.35 | 53.66 | 420.31 | 419.42 | 41.89 |
| execution mode | 57.99 | 61.54 | 521.42 | 525.12 | 46.95 |
| specialized ordinary Core relu | 446.19 | 440.86 | 842.86 | 834.57 | 481.88 |
| generic Core relu | 467.12 | 460.95 | 938.20 | 950.99 | 481.46 |
| generic relu + Tensor | 527.80 | 529.52 | 1487.62 | 1496.84 | 483.36 |
| public relu | 618.64 | 563.62 | 2182.90 | 1862.91 | 412.96 |

Парный выигрыш public relu (`1 - median(after/before)`): native **8.52%**,
VM **14.59%**, обе lanes выиграли 5/5 пар. Metadata/ordinary ABI controls не
получили сопоставимого ускорения. Native execution-mode control замедлился на
6.06% в пяти парах, хотя его алгоритм здесь не менялся; это отдельный результат,
который нельзя скрывать или объяснять стоимостью нового leaf без ablation.

| MPS public relu | Native до | Native после | VM до | VM после | Python |
| --- | ---: | ---: | ---: | ---: | ---: |
| ns/iteration | 2393.83 | 2337.78 | 4110.48 | 3808.81 | 2269.85 |

На MPS парный выигрыш native **2.34%** (4/5 пар), VM **7.00%** (5/5).
Синхронизация стоит на границах цикла: измеряется подача kernels и завершение
всего цикла, а не latency одного завершённого kernel.

[CPU raw results](../build/bench/typed-results/stage1-torch/build/bench/cpu-typed-v2/results.json),
[MPS raw results](../build/bench/typed-results/stage1-torch/build/bench/mps-typed-v2/results.json).

## Оставшиеся расходы

Публичный CPU relu пока медленнее Python примерно в 1.36 раза у native и
4.51 раза у VM. На MPS native близок к Python, но этот прогон не доказывает
превосходство. MNIST после этого изменения ещё не измерялся; microbenchmark
не доказывает ускорения всей модели.

Уточнение после проверки кода 2026-10-10: metadata `type_hook_ids` из
`mode as Int` исключает plain method entry, но сама по себе **не включает**
`needs_param_shaping`. Exact positional вызов создаёт `NativeClosure` и
проходит через общий shaped-method/code-id dispatcher, без вызова
`native_shape_method_args`. При этом сам binder не исполняет эти type hooks,
а native `TYPECHECK` сейчас
понижается в no-op; фактический Int guard этого Torch binding находится в ABI.
VM probe с обычной `identity(value as Int)` также принимает Bool. Поэтому это
не измеренная стоимость runtime type assertion, а дополнительная invocation
и dispatch из-за metadata. Убирать их надо с ясным контрактом typed guards, сохраняя
defaults, keyword shaping, source diagnostics и будущую семантику type checks.
Проверка этого пути: [новый этап MNIST](mnist-runtime-stage4.md).

VM foreign SEND ещё использует общий dispatch; quick cached SEND исключает
foreign handle. Loop/frame и execution-mode controls показывают, что замена
одного ABI не устраняет большую часть интерпретаторных расходов. Native result
по-прежнему содержит shared lifetime metadata и копию tag; coallocation не
отменяет их стоимость. Эти пути требуют отдельного profile/ablation.

## Проверки

Пройдены extension и полный VM suites, dynamic VM/static native resource leaf
fixtures с полным native coverage. Проверены full-width Int, final release,
export/reimport, explicit destroy, tombstone, output-on-error cleanup, missing
output, wrong types и старый host fallback. Extension suite проходит и с
tagged, и с variant Value representation.

В рабочем дереве Torch пройдены CPU, task-local autograd idioms и ONNX operators
в VM/native, плюс настоящий MPS в обеих lanes с `PYTORCH_ENABLE_MPS_FALLBACK=0`.
Каждая native сборка требует полного покрытия. Optional runner tests проходят.

`blocking` не нужен обычным коротким CPU kernels: перенос добавляет очереди и
переходы между потоками, не сокращая вычисление. В resumable VM он может
освободить scheduler worker ради других tasks. В baseline этого замера native
`future.get()` оставлял вызывающий worker ожидающим. Последующее исправление
[blocking](../../sputnik-lang/docs/engineering/native-blocking.md) паркует native
async strand, сохраняя C++ стек на отдельном host thread; это пока имеет цену
host thread на активную async-задачу. Torch compute/metadata вызываются inline;
ожидания устройства и I/O имеют отдельные bindings, а длинная цепочка может
явно использовать `execution_batch` для VM fairness.
