# Direct method entry and trivial bodies, 2026-10-08

Продолжение [execution mode](execution-mode-profile.md): compiler/runtime
оптимизируют сам вызов и короткие тела без знания Torch selectors или layout.

## Native

После обычного разрешения метода и class/call-site guard immutable method
record содержит pointer на сгенерированный entry. Для exact positional calls
без defaults, clauses, auto-assigns, type hooks и специальных параметров entry
обходит shaped-method dispatcher и общий switch по code id. Source trace и
bridge request scope сохраняются; обычные тела получают прежнюю invocation
с owning self/block. Keywords и сложные сигнатуры сохраняют binder.

Общий bytecode summary распознаёт возвращение аргумента, primitive constant,
self, ivar и nullary send на ivar. Primitive returns не создают native register
file/frame, включая вход через обычный code dispatcher. Недостающие аргументы
у raw closure body сохраняют прежний путь. Forwarding entry сохраняет owning
self и receiver на время вложенного user send и заново использует его class
guard; callbacks могут удалить ссылки caller. Builtin/special receivers и
selectors с особым lowering сохраняют исходное тело.

## VM

Live method resolution, class/world version guards и точная сигнатура
предшествуют выполнению summary. Primitive method returns и capture-free
function closures с `CodeKind::Method` обходят frame pool и очистку registers.
Loop blocks с `CodeKind::Block` остаются на прежнем пути. Простые ivar readers
используют путь только для готового layout без watched object state;
import/shape transitions,
watched fields, constructors, properties, named callables и compile-time
instruction budget сохраняют полный invocation.

Nullary ivar forwarding обходит callee frame только для foreign handle с
зарегистрированным typed leaf descriptor. Descriptor проверяется при каждой
доставке, поэтому замена или удаление leaf возвращает обычный путь. Этот
контракт запрещает callbacks, suspension и удержание borrowed resource.
Лёгкий source frame без registers/captures сохраняет pc вложенного send для
fault/rescue. Watch, неподходящий target, invalid/tombstoned handle, обычный
или blocking thunk сохраняют полное исполнение wrapper. ABI не меняется.

Для тел длиннее пяти instructions VM отказывается от summary до проверки
дополнительных signature restrictions. Native C++ и FFI executable hash от
этого последнего изменения не меняются.

## Проверки и измерения

Перед изменениями сохранены compiler, full-native FFI executable, generated
C++ и build log в `build/bench/call-fast`. Torch adapter и modules в этом этапе
не менялись; before/after VM могут использовать один manifest.

Новый language fixture проверяет polymorphism/mixins, primitive/ivar/self
returns, properties, builtin conversion fallback и удаление ссылки caller
во вложенном callback. Отдельные failing programs проверяют property error
и trace forwarding send. `native_call_buffers` дополнен typed forwarding с
fault/rescue/ensure и восстановлением последующих вызовов.

Emitter regression проверяет summary на настоящем emitted module до и после
serialization/deserialization, bare/explicit call flags и отказ при реальном
block register. Это существенно: optional register sentinel бывает `-1`
до serialization и `UINT32_MAX` после неё. Первый CPU запуск выявил ошибку
этой проверки; в нём forwarding не активировался. Он сохранён как
`build/bench/call-fast/cpu1` и не является итоговым измерением.

Пройдены emitter/VM/extension suites, native trivial methods, call buffers,
dispatch cache и language idioms. CPU, idioms, MPS и training/data MPS
проверены в VM и native. После активации forwarding повторены CPU/MPS/training
в обеих lanes; после последнего VM-only изменения повторены language VM/ABI,
call buffers и все четыре Torch VM suites. Source diagnostics для failing
property/forwarding programs совпадают в VM/native и проверяются `cmp`.
Настоящие MPS проверки используют `PYTORCH_ENABLE_MPS_FALLBACK=0`.

Timing lanes выполняются последовательно с warmup, вращением порядка,
проверкой checksums, full-native coverage и hashes; builds/tests/samples
не пересекаются с timing. Python остаётся optional benchmark/reference.

Apple M4, arm64, Apple clang 21, LibTorch/PyTorch **2.14.0**, четыре threads.
CPU: size=1, 300000 host / 50000 kernel iterations; пять внутренних повторов
и пять отдельных процессов каждой lane. MPS relu: size=25088, 20000 kernel
iterations; MPS numel: size=25088, 1000000 host iterations; те же повторы.
MPS kernels синхронизируются на границах loop, поэтому измеряется throughput
подачи kernels вместе с обвязкой, а не latency одного завершённого kernel.

MNIST: SmallResNet 19706 параметров, 2048/512 examples, batch=32, три эпохи,
seed=7, четыре threads, run scope и packed Python optimizer. Один полный
process warmup, затем пять процессов каждой lane с rotating order. Checksums,
epoch losses/accuracy и checkpoint должны совпасть. Время train+evaluation
не включает imports, data preparation, startup и checkpoint.

Before MNIST native собран сохранённым compiler с сохранёнными runtime/
bytecode sources. Остальные SDK sources не менялись и отдельно проверены по
hashes. Torch modules/adapter одинаковы для старого и нового compiler. Это
baseline данного этапа, а не старый бинарник из более раннего исследования.

## Итоговые вызовы

Время — медиана процессных медиан, ns/iteration с общим loop/block harness.
Выигрыш — `1 − median(after / before)` по пяти парным процессам. Пары
вращают порядок lanes; paired percentage не обязана совпадать с отношением
двух соседних медиан. Это сравнение всего нового compiler/runtime, а не
изолированная оценка каждого pointer, summary или frame.

| Case | Lane | До, ns | После, ns | Парный выигрыш | Выигравшие пары |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU free constant | Native | 40,28 | 35,63 | 11,42% | 5/5 |
| CPU free constant | VM | 379,48 | 320,67 | 15,50% | 5/5 |
| CPU identity, 4 arguments | Native | 55,00 | 40,07 | 27,15% | 5/5 |
| CPU identity, 4 arguments | VM | 442,98 | 380,58 | 14,11% | 5/5 |
| CPU method constant | Native | 45,30 | 35,48 | 21,66% | 5/5 |
| CPU method constant | VM | 384,89 | 322,86 | 16,24% | 5/5 |
| CPU method identity | Native | 49,28 | 37,58 | 23,80% | 5/5 |
| CPU method identity | VM | 381,67 | 327,10 | 14,23% | 5/5 |
| CPU TensorCore.numel | Native | 50,52 | 46,56 | 8,54% | 5/5 |
| CPU TensorCore.numel | VM | 383,79 | 385,04 | −0,94% | 2/5 |
| CPU Tensor.numel | Native | 74,94 | 53,48 | 28,95% | 5/5 |
| CPU Tensor.numel | VM | 536,29 | 413,02 | 23,24% | 5/5 |
| MPS Tensor.numel | Native | 75,07 | 53,36 | 28,23% | 5/5 |
| MPS Tensor.numel | VM | 534,00 | 413,62 | 22,54% | 5/5 |
| CPU public relu | Native | 645,83 | 644,43 | 0,41% | 3/5 |
| CPU public relu | VM | 2252,42 | 2262,96 | −0,29% | 1/5 |
| MPS public relu | Native | 2402,36 | 2408,55 | 0,25% | 3/5 |
| MPS public relu | VM | 4293,83 | 4292,15 | 0,39% | 3/5 |

У коротких language calls и public numel есть устойчивый локальный выигрыш.
Прямой Core numel в VM не выигрывает: его wrapper frame уже отсутствовал.
Public numel после изменений: Python **41,72 ns** CPU size=1 и **45,28 ns**
MPS size=25088. Native остаётся примерно **1,28× / 1,18×** медленнее Python;
VM — **9,90× / 9,13×** с этим harness. MPS metadata row не запускает GPU
kernel внутри loop; больший размер также убирает Python small-int cache
для возвращаемого numel.

Простой loop/block control: native 24,30 → 24,00 ns; VM 246,62 → 246,05 ns;
Python 15,43 ns. Loop blocks не оптимизировались. Разности rows полезны как
controlled contrasts, но не являются независимыми слагаемыми чистого call.

Public relu не получил подтверждённого ускорения. Python в финальной серии:
**404,16 ns CPU**, **2224,38 ns MPS**. Native всё ещё медленнее примерно в
**1,59× / 1,08×**, VM — **5,60× / 1,93×**. Ранний отказ VM уменьшил наблюдавшуюся
CPU regression: предыдущая серия без него давала paired +1,90%, финальная
даёт +0,29%. Это разные rotating series, не отдельное доказательство точной
стоимости одной проверки.

Есть сохранённая регрессия отдельного native `execution_mode`: **56,45 →
60,72 ns**, paired **+8,27%**, проигрыш во всех 5 парах. В предыдущей серии
с тем же финальным native binary наблюдалось +6,54%. VM mode: 517,32 →
520,31 ns, paired +0,14%. Причина native роста отдельно не изолирована;
изменение кодогенерации нельзя объявить ускорением всех calls. Logical
autograd state продолжает читаться и сохраняет task isolation.

Raw results: [CPU](../build/bench/call-fast/cpu1-decline/results.json),
[MPS relu](../build/bench/call-fast/mps25088-relu-decline/results.json),
[MPS numel](../build/bench/call-fast/mps25088-numel-decline/results.json).
Предыдущий исправленный forwarding, до раннего VM отказа: `cpu1-final`,
`mps25088-relu`, `mps25088-numel`. Их native executable совпадает с финальным.

## MNIST

Все epoch loss/accuracy и checkpoint checks прошли во всех lanes, включая
сохранённые native/VM. Ниже median train+evaluation, seconds; тот же workload
и adapter до/после, без профилирования.

| Device | Native до | Native после | VM до | VM после | Python |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU | 2,902775 | 2,912225 | 2,982254 | 2,977361 | 2,942890 |
| MPS | 0,779025 | 0,782103 | 0,894304 | 0,883106 | 0,807961 |

Paired after/before native: CPU **1,00326**, MPS **1,00866** (обе lanes 1/5
выигрышей). VM: CPU **1,00173** (2/5), MPS **0,99237** (4/5). На CPU и в native
не получено ускорение тренировки; небольшой native рост сохраняется в
данных. MPS VM paired выигрыш всего 0,76%, CV самой lane 1,32%, и одна пара
проигрывает 2,60% — убедительного большого эффекта на обучение нет.

Native/Python paired median: CPU **0,99422**, MPS **0,97584**, победы **5/5**
в обоих workloads. Запас на CPU маленький; это преимущество уже существовало
в baseline и не доказывает ускорение MNIST новым этапом. VM/Python: CPU
**1,01689** (2/5 побед), MPS **1,10005** (0/5). VM близка к Python на этом CPU
workload; на MPS сохраняется около 10% отставания. Это ограниченный split с
batch=32, а не полный MNIST или универсальное свойство языка.

Generated MNIST C++ вырос с 14,558 до 14,835 MB, executable — с 7,245 до
7,642 MB, около 5,5%. Direct method entries добавляют function bodies и cold
trace/unwind paths. Стоимость увеличенного code size/placement не изолирована;
рост нельзя игнорировать при дальнейшей работе над native.

Raw results: [CPU MNIST](../build/bench/call-fast/mnist-cpu32/results.json),
[MPS MNIST](../build/bench/call-fast/mnist-mps32/results.json).

## Что осталось

Этот этап удалил конкретные лишние dispatch/frame операции у коротких
вызовов. Public relu всё ещё создаёт result resource/foreign handle и Sputnik
Tensor, вызывает обычный free-function/ABI путь и читает logical mode.
Разница CPU C++ kernel control → специализированный Core → generic Core →
Tensor/public показывает оставшуюся обвязку, но не разделяет её allocator,
marshaller и dispatch части. Следующая native работа должна отдельно
проверять direct guarded free-function calls, storage/lifetime result handle
и Tensor allocation, а также footprint дублируемых cold trace entries.
Для VM остаются loop/block frames и обычные effectful/suspendable wrappers;
их нельзя объединять по контракту typed leaf metadata без сохранения
resumption, callbacks, source traces и unwind.

Compiler/native/generated C++ snapshots, source/build/test hashes и
проверка SDK inputs:
[provenance](../build/bench/call-fast/provenance.json).
