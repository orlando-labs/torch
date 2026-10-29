# torch

PyTorch-подобный пакет для Sputnik поверх C++ LibTorch: Tensor, autograd,
CNN/Transformer/RNN, оптимизаторы, загрузка данных, state dict, ONNX export и CPU/Gloo DDP.
Композиция моделей и алгоритмы обучения реализованы в Sputnik.

**Статус: development, 4 октября 2026.** Это работающая начальная
реализация, а не завершённая матрица будущего `0.1`.
[Точный состав и ограничения](docs/implementation-status.md).
[Каталог модулей и границы совместимости](docs/module-coverage.md).
[Обновление пакета в ноутбуках без ручной правки зависимостей](docs/notebook-packages.md).
[Автоматическая регистрация полей-модулей после `init`](docs/automatic-module-registration.md).
[Стиль Sputnik и контракт мутаций](docs/sputnik-style.md).
[DataLoader: нативные треды, ссылки и bounded prefetch](docs/data-loader.md).
[Отчёт о проверенной сборке CPU/MPS/Gloo](docs/build-report.md).
[Сборка source/bytecode/mixed пакетов с native source или prebuilt](docs/package-building.md).
[ONNX API, поддержанные операции и ограничения трассировки](docs/onnx.md).

Пакет, его сборка, launcher и тесты **не используют Python**. Требуются
Sputnik, C++20, CMake, Ninja и готовый LibTorch. Homebrew-формула `pytorch`
сама устанавливает дополнительные компоненты, включая Python; torch
не вызывает их и не линкуется с `torch_python`/`libpython`.

## Сборка на Mac

Проверяемый baseline — macOS arm64, Homebrew `pytorch 2.14.0_1`, CPU + MPS
и CPU/Gloo. Нужен соседний `sputnik-lang` с изменениями native manifest,
реэкспортов, handle lifecycle и native backend из этой задачи. Сборка требует
полного native coverage и не разрешает bytecode fallback.

```sh
brew install pytorch cmake ninja
cd ../sputnik-lang
make build/sputnik -j3
cd ../torch
./tools/build tests/cpu.s --fetch-deps --probes
./build/sputnik/torch.cpu_test
./tools/build examples/train.s
./build/sputnik/torch.train_example
./tools/build examples/showcase.s
./tools/fetch-mnist
./build/sputnik/torch.showcase
```

`--fetch-deps` загружает только Gloo headers из зафиксированного submodule
commit и проверяет SHA256 архива: Homebrew не поставляет их вместе с c10d
headers. Собственная копия Gloo не компилируется; используется библиотека
из того же LibTorch. Повторным сборкам сеть не нужна.

Bootstrap также принимает LibTorch `2.14.1`: его Gloo pin совпадает с
`2.14.0`. Запуск на `2.14.1` пока не проверен; MPS fixes и условия обновления
описаны в [Mac/LibTorch](docs/macos-libtorch.md).

Другой prefix задаётся `--torch-prefix /path/to/libtorch` или
`LIBTORCH_PREFIX`; корень Sputnik — `--sputnik-root` или `SPUTNIK_ROOT`.
Поддерживаемая версия проверяется до компиляции. Manifest и lock создаются
в `build/`. Пока executable зависит от исходного установленного prefix;
автономный relocatable bundle ещё не реализован.

Корневого `sputnik.build.yaml` здесь нет: `tools/build` генерирует manifest
выбранного entry, например `build/torch.cpu_test.build.json` для
`tests/cpu.s`. Native объявления указывают `blocking` прямо после `from "..."`;
отдельного списка `blocking_symbols` в генераторе больше нет. Вычислительные
kernels вызываются inline; ожидания и I/O отмечены отдельно.

Для отдельного прогона без обновления зарегистрированных notebooks используйте
`tools/build tests/losses.s --no-notebook` (также работает с `--run-vm`).

```sh
./tools/test         # CPU, VM/native, serialization, Gloo/DDP, launcher failures
./tools/test --mps   # дополнительно реальное обучение на MPS
```

MPS и loopback sockets должны быть доступны процессу. В sandbox Codex
Metal может оказаться недоступен: MPS проверяется запуском вне sandbox,
а не заменой GPU на CPU. Логи сборок находятся в `build/logs/`.

## API

```sputnik
from torch import tensor, no_grad
from torch.nn import Linear, MSELoss
from torch.optim import AdamW

x = tensor([[1.0], [2.0], [3.0]], device: "mps")
model = Linear(1, 1, seed: 7, device: "mps")
optimizer = AdamW(model.parameters, lr: 0.05, weight_decay: 0.0)
loss = MSELoss()
target = x * 2 + 1

200.times:
  optimizer.zero_grad!
  loss(model(x), target).backward!()
  optimizer.step!

model.eval!
no_grad(): print model(x).to_a
```

`copy` выделяет независимый storage, `detach` сохраняет alias, `snapshot`
делает независимую read-only копию. Изменения данных/gradients/параметров
явно помечены `!`. `load` по умолчанию размещает storages на CPU; исходный
CUDA device не создаётся. Поддерживаются destination device и Map
исходных tags → destinations. `Linear.weight` имеет форму `[out, in]`.

Обучающий вызов BatchNorm/Dropout или содержащей их модели — `model.call!(x)`.
Обычный `model(x)` требует `eval!` либо модели без изменяемого forward-state.
`no_grad():` и `inference_mode():` являются task-local scopes, независимыми от
train/eval. RNG имеет явный объект `Generator`; sampler меняет эпоху только
через `set_epoch!`. [MNIST showcase](examples/showcase.s) обучает неглубокую
residual CNN (19 706 параметров), печатает test accuracy и сохраняет checkpoint
с повторной загрузкой. [Запуск и архитектура](docs/mnist-showcase.md).
[MNIST benchmark: sputnik-vm, sputnik-built и Python](bench/mnist/README.md)
с повторными замерами, проверкой метрик и профилем стадий batch.

`DataLoader(dataset, num_workers: 4, prefetch_factor: 2)` использует нативные
треды Sputnik и передаёт dataset/batches по ссылке: bounded prefetch без
процессов и pickle. Порядок sampler сохраняется; выход из `each` отменяет
и join-ит workers. `num_workers: 0` оставляет синхронную загрузку.

`from torch.onnx import export_model` экспортирует eval-модель без Python.
Результат `Program` содержит ONNX bytes и `save(path)`. Пример с обученным
checkpoint: [export_mnist.s](examples/export_mnist.s). Независимый C++
ONNX Runtime нужен только для optional `tools/test-onnx`, не для пакета.

## Структура

```text
src/torch.s                  # только публичные реэкспорты
src/device.s, dtype.s, backends.s
src/autograd/{context,functional}.s
src/errors.s                     # иерархия native faults
src/tensor/{core,tensor,factories}.s
src/nn/*.s                       # по одному NN-классу в файле; init/functional
src/optim/{optimizer,sgd,adam,adamw,grad}.s
src/optim/lr_scheduler/*.s
src/random/{core,generator}.s
src/data/*.s                     # datasets, samplers, collate, DataLoader
src/serialization/state_dict.s
src/onnx/{exporter,program,trace,core}.s
src/distributed/{core,process_group,ddp}.s
native/{tensor,autograd,nn,optim,random,serialization,distributed,onnx}/
native/bridge.h                   # общий C ABI / exception boundary
cmake/BuildSputnik.cmake            # dependency lock и build manifest
tools/build, tools/test           # shell, без интерпретаторов скриптовых языков
tools/launcher.cpp                # локальный launcher двух и более процессов
tests/                           # Sputnik integration и C++ probes/fixtures
```

Реализации классов находятся в своих файлах; корневой модуль не содержит
их копий. Архитектурный проект сохранён как целевой контракт:
[архитектура](docs/architecture.md), [API](docs/api-and-semantics.md),
[checkpoints](docs/checkpoints.md), [distributed](docs/distributed.md),
[Mac/LibTorch](docs/macos-libtorch.md), [этапы](docs/implementation-plan.md).
