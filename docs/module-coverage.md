# Каталог Sputnik modules

Это реализованный allowlist, а не заявление о полном покрытии всех Python
namespaces PyTorch. Исходная функциональная матрица находится в
[implementation-plan.md](implementation-plan.md); release `0.1` ещё не закрыт.

## NN: 129 публичных классов

| Семейство | Sputnik-классы |
| --- | --- |
| Registry/containers | Module, Sequential, ModuleList, ModuleDict, ParameterList, ParameterDict |
| Dense | Linear, Bilinear, Embedding, EmbeddingBag, Identity, Flatten |
| Activation | ReLU, ReLU6, Sigmoid, Tanh, GELU, SiLU, Mish, LeakyReLU, ELU, SELU, Softplus, Softsign, Tanhshrink, Hardtanh, Hardsigmoid, Hardswish, LogSigmoid, Softshrink, Hardshrink, Threshold, PReLU, GLU, Softmax, LogSoftmax, Softmin, Softmax2d |
| Convolution | Conv1d/2d/3d, ConvTranspose1d/2d/3d |
| Pooling | MaxPool1d/2d/3d (включая indices), AvgPool1d/2d/3d, AdaptiveAvgPool1d/2d/3d, AdaptiveMaxPool1d/2d/3d (включая indices), MaxUnpool1d/2d/3d, LPPool1d/2d/3d |
| Normalization | BatchNorm1d/2d/3d, InstanceNorm1d/2d/3d, LayerNorm, RMSNorm, GroupNorm, LocalResponseNorm |
| Padding/layout | ConstantPad1d/2d/3d, ReflectionPad1d/2d/3d, ReplicationPad1d/2d/3d, PixelShuffle, PixelUnshuffle, Unfold, Fold, Upsample, UpsamplingNearest2d, UpsamplingBilinear2d |
| Stateful regularization | Dropout, Dropout1d/2d/3d, AlphaDropout, FeatureAlphaDropout |
| Attention/Transformer | MultiheadAttention, TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder, Transformer |
| Recurrent | RNNCell, LSTMCell, GRUCell, RNN, LSTM, GRU |
| Loss | MSELoss, L1Loss, SmoothL1Loss, HuberLoss, CrossEntropyLoss, NLLLoss, BCELoss, BCEWithLogitsLoss, KLDivLoss, PoissonNLLLoss, GaussianNLLLoss, SoftMarginLoss, MultiLabelSoftMarginLoss, HingeEmbeddingLoss, MarginRankingLoss, CosineEmbeddingLoss, MultiMarginLoss, MultiLabelMarginLoss, TripletMarginLoss, TripletMarginWithDistanceLoss, CTCLoss |
| Distance | PairwiseDistance, CosineSimilarity |

`nn.functional` содержит соответствующие primitives и Sputnik composition.
`nn.init`: zeros!/ones!/constant!, uniform!/normal!, Xavier и Kaiming uniform/
normal. Инициализация сохраняет leaf identity и работает в no-grad scope.
Value-класс `PackedSequence` перечислен отдельно в `nn.utils.rnn` и не входит
в число 129 layer/container-классов фасада `nn`.

CPU tests `nn_catalog.s`, `nn_families.s`, `normalization.s`, `spatial.s`, `losses.s`, `embeddings.s`, `recurrent.s`, `packed.s` и `attention.s`
вместе создают каждый публичный NN-класс непосредственно либо как child. Проверяются forward/backward
семейства и отдельные численные эталоны. Это не исчерпывающая проверка каждой
комбинации kwargs/dtype/device. MPS job проверяет 2D CNN, BN, embedding,
normalization, attention/Transformer, recurrent gates/sequences и layout;
3D operators не заявляются универсально поддержанными на MPS.

## Остальные namespaces

| Namespace | Реализовано |
| --- | --- |
| tensor | Operators/unary math/comparisons, reductions/argmax, advanced indexing и indexed writes, gather/scatter/reduce, masked operations, nonzero/take, owned uint8 from_bytes, softmax, shape/views, factory family, explicit mutation |
| fft | 22 функции: complex/real/Hermitian FFT и inverse, 1D/2D/ND, normalization, padding/truncation, frequency grids и shifts |
| linalg | 42 функции: decompositions, solve/inverse, least squares, norms, rank/condition, matrix functions и products |
| special | 56 функций: error/gamma/Bessel/polynomial functions, probability transforms, stable logarithms и normalized exponentials |
| nn.utils.rnn | PackedSequence; pack/pad/unpad/unpack utilities, sorting/inverse permutations, CPU metadata и device transfers; 6 non-mutating grammatical synonyms |
| autograd | backward!/grad!, functional grad, higher-order differentiation, task-local grad/inference scopes |
| optim | SGD с momentum/dampening/Nesterov/L2 decay; Adam/AdamW, clipping, in-memory state_dict/restore |
| optim.lr_scheduler | StepLR, CosineAnnealingLR, ReduceLROnPlateau, in-memory state/restore |
| random | Explicit Generator, rand!/randn!/randperm!, seed/state/restore |
| data | Dataset/IterableDataset, TensorDataset с vectorized batch, Subset, ConcatDataset; Sequential/Random/DistributedSampler; nested collate; indexed/streaming DataLoader с native tasks и bounded prefetch по ссылке |
| device/dtype/backends | Device/DType values, complex64/complex128, CPU/MPS capabilities, synchronize |
| serialization | Restricted dense `.pt`, CPU/MPS/Map map_location, strict Module state apply |
| onnx | Inference trace, standard ONNX opsets 18–20, Program/bytes/save, named inputs/outputs и dynamic_axes; [правила и ограничения](onnx.md) |
| distributed | CPU Gloo ProcessGroup, DDP, launcher; включено в основной пакет |

Численные эталоны и autograd для новых namespaces проверяются в
`tests/numerical.s`; `tests/numerical_mps.s` проверяет float32/complex64,
FFT, нормы и special на реальном MPS с отключённым CPU fallback. Поддержка
конкретных dtype/device определяется LibTorch; MPS не поддерживает
float64/complex128. Complex `item`/`to_a` возвращают пары `[real, imag]`;
создание complex Tensor доступно через `complex(real, imag)`, `polar` и
`view_as_complex`. Вызовы поддерживают module aliases, например
`import torch.fft as fft`.

Tensor indexing проверяется общим тестом `tests/indexing.s` на CPU и через
`tests/indexing_mps.s` на MPS: значения, дубликаты, gradients, basic-view/
advanced-copy contract, frozen и noncontiguous writes. Torch non-bang names
сохранены; дополнительно есть Sputnik-style `gathered`, `scattered`,
`index_added`, `masked_filled` и другие грамматические синонимы.
[API и alias contract](tensor-indexing.md).

`tests/normalization.s` и `tests/normalization_mps.s` проверяют cumulative
BatchNorm (`momentum: null`), InstanceNorm с batched/unbatched input,
running stats, affine/bias/dtype, reset и state round trip; LocalResponseNorm;
channel/alpha Dropout, RNG replay, чистые формы с явной mask и bang-формы
с Generator. Проверяются численные результаты, gradients и настоящий MPS
без автоматического CPU fallback. RNG остаётся существующим явным CPU
Generator с переносом маски на устройство входа, не скрытым MPS fallback.
[Контракт нормализации и Dropout](normalization-regularization.md).

`nn.functional.interpolate` поддерживает nearest/nearest-exact,
linear/bilinear/bicubic/trilinear, area и CPU Lanczos, size/scale_factor,
align_corners, recompute_scale_factor и antialias. `tests/spatial.s` и
`tests/spatial_mps.s` проверяют координаты, фильтры, gradients, pool indices,
unpool round trip и LP pooling. На MPS проверяется ограничение ATen для
non-divisible area/adaptive pooling и отсутствие adaptive MaxPool3d;
CPU fallback не включается. [Spatial API и ограничения](spatial-operators.md).

`tests/losses.s` и `tests/losses_mps.s` проверяют Poisson/Gaussian NLL,
margin/ranking/triplet losses, distances, KL `batchmean` и CTC. Есть численные
эталоны, gradients (включая второй порядок Gaussian NLL), custom Sputnik/Module
distance, weight buffers/state restore, CTC padded/concatenated targets,
разные lengths и `zero_infinity`. CTC работает на настоящем MPS; MultiMargin
и MultiLabelMargin в проверенной версии MPS backend отсутствуют и явно
отклоняются без CPU fallback. [Loss API и ограничения](loss-functions.md).

`Embedding` и `EmbeddingBag` поддерживают pretrained/freeze, отрицательный
padding_idx, frequency scaling, CPU sparse gradients, sum/mean/max bags,
CSR offsets и per-sample weights. Max-norm требует явной bang-формы;
`embedding_renormed` сохраняет источник. `tests/embeddings.s` проверяет CPU
и MPS через общий full-native executable, `tests/embeddings_mps.s` служит
MPS VM entry. Sparse MPS явно отклоняется; автоматический CPU fallback
отключён. [Embedding API и ограничения](embedding-operators.md).

`RNN`/`LSTM`/`GRU` поддерживают bidirectional stacks, batch/time-first и
unbatched input, inter-layer dropout, bias/dtype и LSTM projections.
`tests/recurrent.s` сравнивает forward и все gradients с независимым ATen
CPU-эталоном; `tests/recurrent_mps.s` выполняет тот же Sputnik-код на MPS.
RNG изменяется только через bang-вызовы; single-layer/eval не расходуют его.
[Recurrent API и ограничения](recurrent-networks.md).

`nn.utils.rnn` добавляет packed/padded conversions и list-of-sequences utilities,
left/right padding, original batch-order restoration и защищённые metadata.
RNN/GRU/LSTM считают только активные строки packed batch, включая bidirectional
stacks и projected LSTM; initial/final states остаются в исходном batch order.
`tests/packed.s` и `tests/packed_mps.s` сравнивают значения и все gradients
с ATen CPU-эталонами, проверяют padding zero gradients, permutation/device
invariants, RNG и реальный MPS без fallback. Torch default names сохранены;
есть `packed_sequence`, `padded_sequence`, `unpadded_sequence` и другие формы.
[Packed sequence API и ограничения](packed-sequences.md).

Attention поддерживает additive/padding/per-head masks, SDPA kernels,
GQA, scale, attention dropout, time-first/unbatched input, kdim/vdim,
bias/zero slots и optional weights. Transformer прокидывает layout,
padding masks, bias/dtype/activation и attention dropout bang-вызовы.
`tests/attention.s` сравнивает outputs и gradients с ATen/LibTorch CPU-эталоном;
`tests/attention_mps.s` проверяет настоящий MPS без CPU fallback.
[Attention API и совместимость defaults](attention-transformers.md).

## Отличия и оставшиеся портирования

- Attention/Transformer — batch-first и Tensor-output defaults сохранены;
  MHA `need_weights: true` возвращает pair. Старое Bool `mask:` разрешает
  связь при `true`, Torch-style MHA `attn_mask:`/`key_padding_mask:` запрещают.
  Cached decoding, NestedTensor MHA fastpath и custom Transformer activation
  callable отсутствуют. Выбор math/fused SDPA kernel принадлежит ATen backend.
- RNN/LSTM/GRU — dense/packed sequences, explicit initial/final state; batch-first
  default модулей сохранён; новые pack/pad utilities по умолчанию time-first.
  Fused recurrent execution и packed ONNX export отсутствуют.
  State-dict names отражают Sputnik projection submodules,
  не обещают совместимости с Python NN module keys.
- SyncBatchNorm и lazy normalization variants ещё не перенесены.
- Нет lazy/meta layers, fractional pooling, AdaptiveLogSoftmaxWithLoss,
  LinearCrossEntropyLoss, quantized modules и общего sparse Tensor API. Нет AMP/FSDP/RPC и
  Python graph compiler stack.
- Optimizers имеют один набор гиперпараметров, без parameter groups и
  closure/fused algorithms. Slots связываются с parameters по стабильному
  порядку. File-level training session resume ещё не реализован: вложенный
  optimizer state нельзя передать существующему flat `.pt` writer.
- DataLoader поддерживает sync и native thread workers с bounded prefetch
  (`num_workers`, `prefetch_factor`), без serialization/IPC. Stream читается
  в вызывающей task, collate — в workers; length неизвестна. Weighted sampler
  и pin-memory пока отсутствуют. [Контракт](data-loader.md).
- Generic frozen bridge, model copy
  с сохранением ties, safetensors, LoadReport и coordinated DDP checkpoint
  остаются отдельными задачами исходного release plan.

Расширение каталога должно добавлять реальную Sputnik API и backend tests.
Список имён без исполнения либо делегирование Python не считается портом.
