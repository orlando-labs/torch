# Каталог Sputnik modules

Это реализованный allowlist, а не заявление о полном покрытии всех Python
namespaces PyTorch. Исходная функциональная матрица находится в
[implementation-plan.md](implementation-plan.md); release `0.1` ещё не закрыт.

## NN: 96 публичных классов

| Семейство | Sputnik-классы |
| --- | --- |
| Registry/containers | Module, Sequential, ModuleList, ModuleDict, ParameterList, ParameterDict |
| Dense | Linear, Bilinear, Embedding, Identity, Flatten |
| Activation | ReLU, ReLU6, Sigmoid, Tanh, GELU, SiLU, Mish, LeakyReLU, ELU, SELU, Softplus, Softsign, Tanhshrink, Hardtanh, Hardsigmoid, Hardswish, LogSigmoid, Softshrink, Hardshrink, Threshold, PReLU, GLU, Softmax, LogSoftmax, Softmin, Softmax2d |
| Convolution | Conv1d/2d/3d, ConvTranspose1d/2d/3d |
| Pooling | MaxPool1d/2d/3d, AvgPool1d/2d/3d, AdaptiveAvgPool1d/2d/3d, AdaptiveMaxPool1d/2d/3d |
| Normalization | BatchNorm1d/2d/3d, LayerNorm, RMSNorm, GroupNorm |
| Padding/layout | ConstantPad1d/2d/3d, ReflectionPad1d/2d/3d, ReplicationPad1d/2d/3d, PixelShuffle, PixelUnshuffle, Unfold, Fold |
| Stateful regularization | Dropout |
| Attention/Transformer | MultiheadAttention, TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder, Transformer |
| Recurrent | RNNCell, LSTMCell, GRUCell, RNN, LSTM, GRU |
| Loss | MSELoss, L1Loss, SmoothL1Loss, HuberLoss, CrossEntropyLoss, NLLLoss, BCELoss, BCEWithLogitsLoss, KLDivLoss |

`nn.functional` содержит соответствующие primitives и Sputnik composition.
`nn.init`: zeros!/ones!/constant!, uniform!/normal!, Xavier и Kaiming uniform/
normal. Инициализация сохраняет leaf identity и работает в no-grad scope.

CPU tests `nn_catalog.s` и `nn_families.s` вместе создают каждый публичный
NN-класс непосредственно либо как child. Проверяются forward/backward
семейства и отдельные численные эталоны. Это не исчерпывающая проверка каждой
комбинации kwargs/dtype/device. MPS job проверяет 2D CNN, BN, embedding,
normalization, attention/Transformer, recurrent gates/sequences и layout;
3D operators не заявляются универсально поддержанными на MPS.

## Остальные namespaces

| Namespace | Реализовано |
| --- | --- |
| tensor | Operators/unary math/comparisons, reductions/argmax, index_select, owned uint8 from_bytes, softmax, shape/views, factory family, explicit mutation |
| autograd | backward!/grad!, functional grad, higher-order differentiation, task-local grad/inference scopes |
| optim | SGD с momentum/dampening/Nesterov/L2 decay; Adam/AdamW, clipping, in-memory state_dict/restore |
| optim.lr_scheduler | StepLR, CosineAnnealingLR, ReduceLROnPlateau, in-memory state/restore |
| random | Explicit Generator, rand!/randn!/randperm!, seed/state/restore |
| data | Dataset/IterableDataset, TensorDataset с vectorized batch, Subset, ConcatDataset; Sequential/Random/DistributedSampler; nested collate; indexed/streaming DataLoader с native tasks и bounded prefetch по ссылке |
| device/dtype/backends | Device/DType values, CPU/MPS capabilities, synchronize |
| serialization | Restricted dense `.pt`, CPU/MPS/Map map_location, strict Module state apply |
| onnx | Inference trace, standard ONNX opsets 18–20, Program/bytes/save, named inputs/outputs и dynamic_axes; [правила и ограничения](onnx.md) |
| distributed | CPU Gloo ProcessGroup, DDP, launcher; включено в основной пакет |

## Отличия и оставшиеся портирования

- Attention/Transformer — batch-first `[batch, time, embedding]`, Bool mask:
  `true` разрешает связь. Fully masked rows дают zero output/gradient. MHA
  возвращает Tensor, не tuple attention weights; нет fused SDPA, additive
  float mask, key/value dimensions, cached decoding или attention dropout.
- RNN/LSTM/GRU — dense batch-first, unidirectional stack, explicit initial/
  final state. Packed sequences, bidirectional mode и inter-layer dropout
  отсутствуют. State-dict names отражают Sputnik projection submodules,
  не обещают совместимости с Python NN module keys.
- BatchNorm использует fixed momentum; cumulative `momentum=null`, sync BN,
  InstanceNorm и специальные normalization variants ещё не перенесены.
- Нет lazy/meta layers, EmbeddingBag, channel/alpha Dropout, interpolation/
  Upsample, unpooling, fractional/LP pooling, специализированных losses и
  quantized/sparse modules. Нет torch.linalg/fft/special/AMP/FSDP/RPC и
  Python graph compiler stack.
- Optimizers имеют один набор гиперпараметров, без parameter groups и
  closure/fused algorithms. Slots связываются с parameters по стабильному
  порядку. File-level training session resume ещё не реализован: вложенный
  optimizer state нельзя передать существующему flat `.pt` writer.
- DataLoader поддерживает sync и native thread workers с bounded prefetch
  (`num_workers`, `prefetch_factor`), без serialization/IPC. Stream читается
  в вызывающей task, collate — в workers; length неизвестна. Weighted sampler
  и pin-memory пока отсутствуют. [Контракт](data-loader.md).
- Tensor advanced indexing/gather/scatter, generic frozen bridge, model copy
  с сохранением ties, safetensors, LoadReport и coordinated DDP checkpoint
  остаются отдельными задачами исходного release plan.

Расширение каталога должно добавлять реальную Sputnik API и backend tests.
Список имён без исполнения либо делегирование Python не считается портом.
