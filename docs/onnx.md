# ONNX export

`torch.onnx` экспортирует inference-граф Sputnik `Module` в стандартный
ONNX. Модель исполняется в Sputnik; C++ tracer LibTorch записывает ATen-операции,
адаптер переводит их в ONNX, а serializer и structural checker LibTorch
создают и проверяют protobuf. Python symbolic registry не используется.
Обычный PyTorch ATen→ONNX pass обращается к Python symbolic functions,
поэтому этот pass здесь заменён собственным набором правил.
[Исходный pass PyTorch](https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/passes/onnx.cpp).

Для сборки и экспорта достаточно существующего LibTorch. Отдельные
`onnx`, protobuf, ONNX Runtime и Python в зависимости пакета не добавлены.

## API

```sputnik
from torch.onnx import export_model

model.eval!
program = export_model(model, images,
  input_names: ["images"], output_names: ["logits"],
  dynamic_axes: {"images": {0: "batch"}, "logits": {0: "batch"}})
program.save("model.onnx")
print "#{program.node_count} nodes, #{program.byte_count} bytes"
```

Полная сигнатура:

```sputnik
export_model(model, inputs, path: null, input_names: null, output_names: null,
  dynamic_axes: {}, kwargs: {}, opset_version: 18)
```

- `model` — `Module` после явного `eval!`. Экспорт не меняет train/eval mode.
- `inputs` — Tensor либо Array/Tuple позиционных Tensor inputs.
- `kwargs` передаются в Sputnik forward. Tensor values становятся graph inputs
  после позиционных inputs в порядке Map; остальные values фиксируются
  примером исполнения. Tensor inputs должны иметь разные identities и
  не совпадать с model parameters/buffers.
- Выход — Tensor либо вложенные Array/Tuple из Tensor. Структура рекурсивно
  разворачивается в список graph outputs; имена задаются в этом порядке.
- Имена по умолчанию — `input_0`, `output_0` и далее. Имена должны быть
  уникальными, непустыми и не начинаться с цифры.
- `dynamic_axes` — Map имени input/output в Map неотрицательного axis→label.
- Поддержаны opsets 18–20, по умолчанию 18.
- `path` сохраняет файл после завершения экспорта и проверки графа.

Имя `export_model` выбрано потому, что `export` — ключевое слово Sputnik.
`Program` содержит `bytes`, `byte_count`, `node_count`, `initializer_count`,
`opset_version`, `input_names`, `output_names`. `initializer_count` считает
связанные state tensors, включая неиспользованные buffers. Tied parameters
с одной TensorImpl получают один initializer. Значения parameters/buffers
копируются на CPU; в ONNX не сохраняются MPS device tags.

`Program.save(path)` возвращает self и не меняет объект. Он пишет временный
файл в том же каталоге, выполняет fsync и публикует через rename. Ошибка
трассировки, lowering или записи сохраняет существующий destination;
временный файл удаляется. Directory fsync и external-data sidecars пока
не реализованы: модель сохраняется одним файлом с inline tensor data.
[ONNX IR и initializer tensors](https://onnx.ai/onnx/repo-docs/IR.html).

## Трассировка и семантика Sputnik

Экспорт выполняет forward в task-local `no_grad` scope. Trace token тоже
принадлежит логической Sputnik-задаче, наследуется дочерней задачей и
передаётся каждым tensor FFI call. C++ guard присоединяет tracer к текущему
worker только на время kernel call и восстанавливает предыдущий TLS state.
Граф защищён mutex; перенос blocking FFI между workers не теряет trace.
Контекст и owned trace handle освобождаются через `ensure`, включая ошибки.

Forward должен быть чистым: без изменения Sputnik fields, RNG, внешнего I/O
или запуска незавершённых фоновых задач. Tensor mutation, state-dict copy,
training BatchNorm, backward/functional grad и материализация `item`/`to_a`
во время trace отвергаются. Это защита tensor boundary, а не транзакционный
откат произвольного Sputnik-кода. Вложенные экспорты не поддерживаются.

Трассировка записывает **исполненный путь**. Sputnik branches, loops, metadata
из `shape` и static kwargs не превращаются в symbolic control flow.
`dynamic_axes` задаёт symbolic размеры входов/выходов, но не переписывает
константы, полученные из example shape. Динамический batch SmallResNet,
tensor math и primitive attention проверен на batch 1 и 3. MHA использует
Sputnik shape для reshape, а recurrent layers разворачивают sequence loop:
для них пока проверяется фиксированная форма. Новые формы нужно проверять
отдельно; объявление axis само по себе не доказывает их поддержку.

## Набор правил lowering

| Область | Реализованные операции |
| --- | --- |
| Tensor math | add/sub/mul/div/pow, matmul/mm/bmm/addmm; comparisons и boolean logic; square, abs/neg, exp/log, sqrt/rsqrt/reciprocal, sin/cos, floor/ceil/round/erf/isnan |
| Типы и reductions | cast/to, sum/mean/amax/amin/any/all, argmax, softmax/log_softmax, clamp |
| Layout | reshape/view, flatten по произвольному диапазону, transpose/permute, squeeze/unsqueeze, narrow/slice, select/index_select/embedding, cat/stack, repeat/expand, where/masked_fill |
| Constants | zero/one/full factories с фиксированной формой, arange; native zero/one/full-like через Shape→ConstantOfShape |
| CNN | Conv/ConvTranspose 1D–3D, linear, eval BatchNorm, GroupNorm, LayerNorm, global adaptive pooling, обычный average/max pooling |
| Padding | constant, reflect и replicate |
| Activations | ReLU, sigmoid/tanh, exact/tanh GELU, SiLU/Mish, Softplus, LogSigmoid, LeakyReLU/ELU/SELU, hard activations/shrink, PReLU, Threshold |
| Composition | Sequential/residual models, RMSNorm, masked/causal attention, MHA и recurrent composition при фиксированной форме, eval Dropout |

Набор правил не равен полной поверхности PyTorch. Неизвестная операция
даёт `ONNXError` с именем ATen operator и opset; ATen/custom fallback в
готовом графе не допускается. Дополнительные outputs вроде pooling indices
поддерживаются только когда не используются. Non-global adaptive pooling
требует static spatial dimensions, равномерно делимых на target sizes.
AveragePool divisor override и training Dropout не поддержаны. Bilinear,
PixelShuffle/Unshuffle, Unfold/Fold и специализированные loss kernels пока
не имеют отдельных правил; Sputnik-composed MSE/L1 работают через tensor math.

Structural checker LibTorch проверяет сериализованную модель; это не
исчерпывающая проверка каждого dtype/shape/operator сочетания. Независимые
численные проверки исполняют графы в C++ ONNX Runtime.

## Примеры и проверки

[export_mnist.s](../examples/export_mnist.s) загружает обученный
`build/mnist-resnet.pt` с `map_location` выбранного device, экспортирует
`build/mnist-trained.onnx` и сохраняет inputs/outputs для независимого сравнения.
По умолчанию device — CPU; checkpoint/output/references задаются CLI options.

```sh
./tools/build examples/export_mnist.s
PATH=/nonexistent ./build/sputnik/torch.export_mnist_example
PYTORCH_ENABLE_MPS_FALLBACK=0 PATH=/nonexistent ./build/sputnik/torch.export_mnist_example \
  --device mps --output build/mnist-trained-mps.onnx --references build/mnist-trained-mps-reference.pt

# Optional test dependency, не зависимость сборки или runtime пакета:
./tools/fetch-onnxruntime
./tools/test-onnx
./tools/test-onnx --mps
```

Test runner использует официальный C++ ONNX Runtime 1.30.0 macOS arm64,
архив проверяется по SHA256. Runtime связан только с отдельным validator
`torch_onnx_runtime`. Он сравнивает shape/dtype, floating values с
rtol=1e-4/atol=1e-5 и integer/bool outputs точно. Native programs запускаются
с `PATH=/nonexistent`, а dylib audit отвергает `libpython`, Python.framework
и `torch_python`. MPS job отключает automatic CPU fallback.
[Официальный C++ runtime release](https://github.com/microsoft/onnxruntime/releases/tag/v1.30.0).
