# MNIST showcase

Модули `examples/mnist/*.s` объявляют имена `sputniktorch.mnist.*` через
`package`. Bootstrap читает эти декларации и регистрирует пары name/path
в build manifest; каталог `examples` не добавляется в import search paths.
Поэтому пример использует абсолютные module imports, хотя файлы находятся
в examples. Это пространство имён примера, не реэкспорт публичного пакета.
Текущий bootstrap включает эти модули во все manifests; отдельный manifest
примеров ещё не выделен. Относительные imports с ведущей точкой Sputnik
пока не поддерживает.

Пример обучает `SmallResNet` на MNIST средствами Sputnik. По умолчанию используются
все 60 000 train и 10 000 test изображений, MPS, пять эпох, batch 128 и seed 7.
Test split участвует только в оценке; gradients и BatchNorm statistics на нём
не обновляются. Train и test сохраняются на CPU, на MPS передаётся текущий batch.

```sh
./tools/fetch-mnist
./tools/build examples/showcase.s
PYTORCH_ENABLE_MPS_FALLBACK=0 ./build/sputnik/sputniktorch.showcase
```

Датасет загружается отдельно от сборки и исполнения, с [зеркала MNIST,
используемого torchvision](https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py).
Shell script проверяет четыре канонические MD5 суммы перед распаковкой gzip.
Он не запускает Python. Повторное исполнение программы не обращается к сети.

Архитектура: stem `Conv2d(1, 16, 3)` → BatchNorm → ReLU, residual-блок
16 → 16, residual-блок 16 → 32 со stride 2 и projection shortcut,
global average pooling и Linear(32, 10). Всего шесть convolutions и
19 706 обучаемых параметров. Каждый residual-блок имеет две convolutions 3×3.
AdamW использует lr 0.003 и weight decay 0.0001; cosine schedule уменьшает
lr до 0.0003. Loss — cross entropy; accuracy вычисляется через `argmax`.

Классы находятся в отдельных файлах:

- [ResidualBlock](../examples/mnist/residual_block.s) — residual и projection;
- [SmallResNet](../examples/mnist/network.s) — композиция сети;
- [MNIST](../examples/mnist/dataset.s) — нормализация и TensorDataset;
- [IDX reader](../examples/mnist/idx.s) — uint8 IDX header и dimensions;
- [Options](../examples/mnist/options.s) — стандартный Sputnik ArgParser;
- [showcase](../examples/showcase.s) — train/eval и checkpoint round trip.
- [Training](../examples/mnist/training.s) — общий train/eval-цикл примера и бенчмарка.

Python-эквивалент и воспроизводимое сравнение VM/native/Python находятся
в [bench/mnist](../bench/mnist/README.md). Python нужен только этому
опциональному сравнению; обычная сборка и suite остаются без Python.

`forward!` явно разрешает обновление BatchNorm buffers, `forward` применяется
после `eval!`. `zero_grad!`, `step!`, `each` и `times` используют bare-вызовы.
`TensorDataset.batch` выбирает изображения через `index_select`, без
поштучного преобразования пикселей или изображений в Sputnik arrays.
Custom collation продолжает получать отдельные samples.

Каждая эпоха печатает sample-weighted train/test loss, test accuracy и время.
После обучения программа сохраняет state dict с параметрами и BatchNorm
buffers, создаёт новый model, загружает веса через `map_location` и повторяет
оценку. По умолчанию checkpoint — `build/mnist-resnet.pt`.

Параметры запуска: `--data`, `--checkpoint`, `--device cpu|mps`, `--epochs`,
`--batch-size`, `--seed`, `--train-limit`, `--test-limit`. Limits задаются
явно и предназначены для коротких проверок, например:

```sh
./build/sputnik/sputniktorch.showcase --device cpu --epochs 1 \
  --train-limit 128 --test-limit 64 --batch-size 32 \
  --checkpoint build/mnist-smoke.pt
```

`tools/test --mps` собирает showcase и исполняет короткий MNIST smoke,
если датасет уже подготовлен. Suite не скачивает его автоматически.
Smoke не служит измерением качества обучения на полном MNIST.

## Проверенный результат

1 октября 2026, Homebrew LibTorch 2.14.0_1, macOS arm64, MPS без CPU fallback:

| Эпоха | Train loss | Test loss | Test accuracy |
| --- | --- | --- | --- |
| 1 | 0.535504 | 0.149105 | 96.38% |
| 2 | 0.101828 | 0.167604 | 94.52% |
| 3 | 0.069920 | 0.072129 | 97.83% |
| 4 | 0.052130 | 0.070501 | 97.94% |
| 5 | 0.042346 | 0.044511 | 98.69% |

Использованы default параметры и полные splits. Checkpoint после загрузки
вернул те же test loss и accuracy. Epoch loop занял суммарно около 24 секунд;
сюда не входят сборка, загрузка данных и заключительная проверка checkpoint.
Лог: `build/logs/mnist-full-train.log`. Процесс запущен с `PATH=/nonexistent`,
а dylib trace не содержит Python runtime libraries. Native build покрывает
все тела функций без VM fallback.
