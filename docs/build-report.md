# Проверенные сборки: 1–4 октября 2026

Проверены расширенный module suite и MNIST showcase после Sputnik style pass.
Результаты CPU/VM, MPS и двухпроцессного Gloo приведены ниже. Homebrew PyTorch
содержит работоспособные MPS и Gloo в одной установке. Собственная сборка
PyTorch для этого baseline не понадобилась.

## Среда и зависимости

| Компонент | Проверенное значение |
| --- | --- |
| Host | macOS 26.6.2 (25G83), arm64 |
| SDK | Command Line Tools, macOS SDK 26.5 |
| Compiler | AppleClang 21.0.0, clang-2100.1.1.101 |
| LibTorch | Homebrew `pytorch 2.14.0_1`, upstream version `2.14.0` |
| Prefix | `/opt/homebrew/Cellar/pytorch/2.14.0_1` |
| Build | CMake 4.4.3, Ninja 1.13.2; adapter C++20 |
| Gloo headers | `74cc005ae13f69c11d8a41e50b42025b6730e796` |
| Gloo archive SHA256 | `1d7260c64008e5ad430a01f63dfd94a3e04ea663f2694a3834f3877b4da66833` |
| TorchConfig SHA256 | `142f1960602a4831c63871d7d734da6b9d288b9b2c76eecfda7b6c5de396eb0a` |

Homebrew не поставляет Gloo headers, необходимые для c10d consumer. Bootstrap
загружает зафиксированный архив headers и проверяет checksum. Используется
Gloo из `libtorch_cpu`; отдельная библиотека Gloo не строится.

Homebrew хранит C++ libraries внутри своего `site-packages/torch/lib`.
Это layout установки: sputniktorch использует только headers и `torch`,
`torch_cpu`, `c10`. В package source, bootstrap, launcher и tests нет
вызовов Python, pip или torchgen. Formula Homebrew имеет собственные
зависимости, включая Python, которые не требуются для запуска sputniktorch.

## Результаты

| Проверка | Результат |
| --- | --- |
| C++ CPU и MPS | Forward `14`, gradient `[2,4,6]`, SGD trained weight `3` |
| Sputnik native CPU/MPS | Actual device, autograd, views/copy/snapshot, typed faults, dtype checks, seeded RNG и Linear training |
| Sputnik idioms, VM/native CPU | Conversion properties, literal case!, splats/slices; task-local grad scopes с exception restore; functional и higher-order grad; независимый grad! snapshot |
| NN catalog, VM/native CPU | Совместно tests создают все 96 публичных NN-классов; forward/backward, numerical references, initialization, containers и registry invariants |
| NN catalog, native MPS | 2D CNN/BatchNorm с explicit mutation, embedding, LayerNorm/RMSNorm, attention/Transformer, RNN/LSTM/GRU cells и multi-layer sequences, padding/layout |
| Training/data, VM/native CPU и native MPS | Numerical SGD momentum, Adam и AdamW decay; clipping, Adam snapshot/resume с сохранением parameter references; три schedulers; indexed/streaming DataLoader и deterministic samplers |
| Threaded DataLoader, 4 октября, VM/full-native CPU | Native tasks, передача тех же samples/Tensor storage по ссылке, bounded prefetch, sampler order и explicit epochs, indexed/streaming tails, null custom batches, inherited grad mode, propagation ошибок и cancellation/join при раннем return; 32 рабочих очереди без блокировки свободных scheduler threads |
| Threaded TensorDataset, 4 октября, full-native MPS | Векторные batches из двух Tensor сохраняют порядок и partial tail при `num_workers: 2, prefetch_factor: 1`; training/data suite проходит на настоящем MPS с `PYTORCH_ENABLE_MPS_FALLBACK=0` |
| MNIST showcase, MPS | SmallResNet, 19 706 параметров, полные 60 000 train / 10 000 test, пять эпох: test accuracy 98.69%, loss 0.044511; checkpoint сохраняет метрики |
| Sputnik VM CPU | Tensor/autograd baseline, modular imports и все расширенные CPU module suites |
| Модульные imports | Прямой import Tensor, native errors и nested Sequential model |
| Native serialization | Round-trip, default CPU для CUDA tags, Map `map_location`, shared strided storage views |
| Native MPS checkpoint | CUDA-tagged fixture загружен на MPS; shared-storage alias сохранён |
| Restricted parser | OrderedDict `_metadata`, protocol 4 FRAME; bad bounds, forbidden GLOBAL, truncation, unsupported protocol и duplicate keys отвергнуты |
| Совместимость writer | C++ LibTorch `torch::pickle_load` прочитал собственный sputniktorch output |
| Native Gloo/DDP, 2 ranks | SUM, broadcast, barrier; gradients совпали с concatenated-batch reference; SGD training |
| Native и VM DDP errors | Schema mismatch обнаружен на обоих ranks; повторный close безопасен; доступ после close даёт LifetimeError |
| Launcher | Worker failure передан вызывающему процессу; внешний timeout завершает workers |
| Python runtime | CPU native executable и MPS showcase успешно запущены с `PATH=/nonexistent`; `DYLD_PRINT_LIBRARIES` не обнаружил libpython/Python.framework/torch_python |
| Sputnik ABI regression | `sputnik_ext_tests: ok`; bare constructor/lifecycle fixture вернул `24` в VM и native; collected TensorCore teardown проверен в обоих backends |
| MNIST optimization, 4 октября | Generic foreach/packed AdamW, bulk zero_grad, metadata и execution scopes проверены в VM/full-native на CPU/MPS и ONNX; borrowed string lifetime и literal defaults — в regression tests Sputnik. [Замеры, провалы и точные границы преимущества](mnist-benchmark-results.md) |

Все native builds требуют `--require-full-native`. Проверены
`native_bytecode_fallback: false`, `native_graph_fallback_code_count: 0`,
`native_graph_full_coverage: true`, `native_graph_runtime_bridge: false` и
`native_graph_vm_independent: true`. Нулевой счётчик VM code bodies сам по
себе не считается доказательством отсутствия whole-program fallback.

| Native program | Нативных тел / всего |
| --- | --- |
| CPU/MPS tensor baseline | 534 / 534 |
| Sputnik idioms | 543 / 543 |
| NN catalog CPU / MPS | 1134 / 1134; 1137 / 1137 |
| NN families CPU | 1165 / 1165 |
| Training/data CPU / MPS | 670 / 670; 673 / 673 |
| Serialization | 535 / 535 |
| DDP и DDP errors | 536 / 536 |
| MPS MNIST showcase | 1282 / 1282 |

Tests MPS не допускают автоматический CPU fallback; seeded random отдельно
генерируется на CPU и переносится на заданный device по объявленному
контракту. Проверка MPS не означает поддержку всех 3D ATen operators на GPU.

Двухпроцессные tests и Metal jobs запускались вне sandbox с доступом к
loopback sockets и GPU. Недоступность Metal внутри sandbox не считалась pass.
Полный Xcode для consumption готового LibTorch не потребовался.

## Воспроизведение и логи

```sh
./tools/build tests/cpu.s --fetch-deps --probes
./tools/fetch-mnist
./tools/test --mps
./tools/build examples/showcase.s
PYTORCH_ENABLE_MPS_FALLBACK=0 PATH=/nonexistent ./build/sputnik/sputniktorch.showcase
```

`build/native-lock.json` содержит dependency identity текущей сборки.
`build/logs/` содержит build/VM logs, `cpu-dylibs.log` и
`showcase-dylibs.log`. CPU/module/serialization/native DDP проверки сохранены
в `build/full-suite-mnist.log`; после исправления VM bare destructor повторены
CPU и VM DDP, выполнены MPS suites и MNIST smoke, с итоговым
`PASS sputniktorch test suite` в `build/full-suite-mnist-resumed.log`.
Полное обучение сохранено в `build/logs/mnist-full-train.log`, audit библиотек —
в `build/logs/mnist-full-dylibs.log`.
Сообщения launcher о failure и timeout в конце успешного прогона относятся
к обязательным негативным tests.

В соседнем Sputnik добавлены fixtures conversion properties/splats/slices,
private class identity, typed NativeError rescue, bare native teardown и
binary `fs.read_bytes` с capability и size limit. Они проверяются в VM и
с обязательной нативной компиляцией через `make test-native-language-idioms`.
Set spread в Array проверяется отдельно как ожидаемый TypeError.
Существующие scalar, sequence и Str/Bytes native fixtures также собраны
с `--require-full-native` и вернули ожидаемые `12`, `26` и `23`.

Style pass разделил многострочные методы, применил keyword shorthand,
postfix guards и bare nullary calls, включая `each`, `zero_grad!`, `step!`
и `eval!`; `times`/`Array.of` используются в recurrent sequences и batching.
Аудит 566 определений не обнаружил пропущенных
разделителей, inline prefix guards или повторов одноимённых kwargs.
Правила сохранены в `AGENTS.md` и [sputnik-style.md](sputnik-style.md).

Bootstrap принимает также LibTorch `2.14.1`, чьи MPS fixes относятся к
SVD/svdvals/lstsq. Gloo submodule pin двух тегов совпадает. Этот прогон
выполнен на Homebrew `2.14.0_1`; runtime `2.14.1` пока не проверен.
[Разбор patch-релиза](macos-libtorch.md#patch-релиз-2141).

## ONNX: 2–3 октября

Проверен [ONNX exporter](onnx.md) на том же Homebrew `2.14.0_1` baseline.
Пакет использует C++ tracer/serializer LibTorch и собственный ATen→standard
ONNX lowering. Python, отдельный protobuf/ONNX codec и ONNX Runtime
в зависимости пакета не добавлены. Optional validator использует официальный
C++ ONNX Runtime 1.30.0 arm64 с проверенным SHA256 архива.

| Проверка | Результат |
| --- | --- |
| MNIST SmallResNet, VM/native | Экспортирован inference-граф, 37 узлов, 85 146 bytes; dynamic batch 1/3 совпадает с ONNX Runtime, max error 1.19209e-7 |
| Обученный checkpoint | Native example загрузил `build/mnist-resnet.pt` на CPU и экспортировал ONNX; batch 1/3 совпадает, max error 7.62939e-6 |
| Обученный checkpoint, MPS | 3 октября checkpoint загружен с map_location MPS, SmallResNet выполнен и экспортирован на MPS без CPU fallback; CPU ONNX Runtime подтвердил batch 1/3, max error 5.72205e-6 |
| CPU operators, VM/native | 15 графов: tensor math/types/layout, task scope, masked/causal attention, normalization, activations, padding, pooling, ConvTranspose, MHA, GRU и LSTM; max error 1.19209e-6 |
| MPS operators, native | Те же 15 графов вычислены на MPS без automatic CPU fallback, экспортированы и проверены в CPU ONNX Runtime; max error 3.8147e-6 |
| Формы | Dynamic batch 1/3 для tensor/CNN/primitive attention; fixed batch 2 для MHA/GRU/LSTM |
| Отказы | Typed ONNXError для input mutation, state loading, item boundary, backward/grad, unsupported operator, nested trace, invalid names/axes/opset; scope восстановлен, вход и существующий destination сохранены |
| State binding | Tied parameters получают один initializer; input identity collisions отклонены |
| Общий FFI bridge | Обычные tensor/autograd/training paths прошли в CPU VM/native и MPS native, включая higher-order grad и state-dict restore |
| Python runtime | Executables и C++ validator запущены с PATH=/nonexistent; DYLD audit не обнаружил libpython, Python.framework или torch_python |

Численный validator проверяет shape/dtype, rtol=1e-4/atol=1e-5 для floating
outputs и точное совпадение integer/bool. Проверка не означает полного
покрытия ONNX или произвольных динамических Sputnik branches/shapes.

Логи CPU полного ONNX прогона: `build/logs/onnx-suite.log` с итоговым
`PASS ONNX export + independent C++ runtime validation`. MPS export:
`build/logs/onnx-mps-native.log`, `onnx-mps-dylibs.log` и
`onnx-mps-*-runtime-dylibs.log`; numerical results получены отдельными
запусками того же validator. Воспроизведение: `./tools/fetch-onnxruntime`,
затем `./tools/test-onnx --mps` с доступом к Metal.

Нативное покрытие ONNX MNIST test — 1192/1192 тел, operator test —
1214/1214 CPU и 1216/1216 MPS. Объединённая регрессия общего моста:
693/693 CPU и 695/695 MPS. Во всех этих build reports отсутствуют
bytecode fallback и runtime bridge, подтверждены full coverage и VM
independence. Логи регрессий: `vm-onnx-regression.log`,
`onnx-regression-native.log`, `onnx-regression-mps-native.log` в `build/logs/`.
Аудит 603 Sputnik-определений проверил разделители multiline methods,
keyword shorthand и отсутствие inline prefix guards.

CLI-пример с `--checkpoint/--device/--output/--references` собран с
1187/1187 native bodies и прошёл CPU/MPS export. Логи полного обученного
MPS графа: `onnx-trained-mps.log`, `onnx-trained-mps-runtime.log`,
`onnx-trained-mps-dylibs.log` в `build/logs/`. Повторное CPU сравнение CLI
примера: `onnx-trained-cpu-runtime.log`. Python dylib audit чистый.

## Границы результата

Это проверка расширенной development-реализации. Полная матрица `0.1` не
закрыта: frozen mutation bridge, safetensors, file-level training resume и
остальные release gates перечислены в
[implementation-status.md](implementation-status.md).

Executable использует установленный Homebrew prefix. Relocation полного
LibTorch/Metal bundle, чистая независимая машина и старые версии macOS ещё
не проверены. CMake warning об отсутствующем optional kineto и warning SDK
о deprecated spawn API не помешали компиляции и выполнению.
