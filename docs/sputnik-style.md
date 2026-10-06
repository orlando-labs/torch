# Sputnik как основной язык пакета

Каждый публичный NN-класс имеет отдельный `.s` файл. `sputniktorch.s`,
`nn.s`, `optim.s`, `data.s`, `autograd.s` и scheduler facade только
импортируют и реэкспортируют имена. Native handles и C ABI — детали границы.

Sputnik отвечает за регистрацию параметров/buffers, Transformer projections
и residuals, recurrent gates и проход по времени, SGD/Adam/AdamW, scheduler,
sampler, collation и batching. C++ отвечает за ATen kernels, native lifetime,
checks на этой границе и форматы LibTorch. Отдельного C++ алгоритма SGD нет.

## Идиомы

- Многострочные функции/методы отделяются пустой строкой; подряд идут только
  однострочные определения. Короткие wrappers и metadata getters записываются
  в одну строку, длинные сигнатуры и argument lists переносятся.
- `layer.call!(x, mask:, causal:)` передаёт одноимённые locals без повторения.
  `mask: @mask` и `seed: seed + index` сохраняют явное выражение. Короткие
  guards и conditional assignments используют postfix `if`/`unless`:
  `x = @norms[0](x) unless @norm_first`.
- `def init(@field, @option: default)` сохраняет конфигурацию, `attr` задаёт
  read-only accessors. Общие helpers инициализации имеют явный `!`.
- Tensor реализует `+`, `-`, `*`, `/`, `**` и `[]`. `shape`, `dtype`,
  `device`, `square`, `item`, `leaf?` читаются как nullary properties.
  Bare-форма применяется также к `optimizer.zero_grad!`, `optimizer.step!`,
  `model.eval!`, `batches.each |batch|:` и `epochs.times |epoch|:`.
  Методы с optional arguments (`sum`, `mean`, `backward!`) также допускают
  bare-форму: defaults применяются как при пустом explicit call. Скобки нужны
  для вызова lexical function value, включая импортированные функции.
  Конструктор `ReLU()` создаёт объект, `ReLU` обозначает класс.
- Array/Map blocks, `$it`, `transform_values`, `reduce`, `Array.of`, Range,
  splats и kwargs составляют layers и данные. State LSTM и batch
  распаковываются Tuple patterns. Policy выбирается через `case!`.
  Recurrent sequences используют `times`/`Array.of`, indexed batching —
  `times` и Range slice без ручного loop counter.
- `no_grad():`, `with_grad():`, `inference_mode():` возвращают результат и
  восстанавливают режим при исключении. Binding принадлежит Sputnik task;
  перед blocking FFI передаётся snapshot режима.
- Состояние обозначается в имени: `backward!`, `grad!`, `step!`, `seed!`,
  `rand!`, `load_state_dict!`, `set_epoch!`, `train!`, `call!`.

## Мутация

Обычный Tensor operator создаёт значение; view может разделять storage.
`copy` независим и differentiable, `detach` сохраняет alias, `snapshot`
создаёт независимые read-only данные. Мутатор проверяет read-only flag и
in-place правила autograd. Generic frozen/native mutation bridge остаётся
release gate; snapshot не выдаётся за полную реализацию Sputnik `freeze`.

`Module.call` заранее проверяет stateful descendants: Sequential не начинает
менять running statistics до обнаружения неправильного вызова. BatchNorm
обновляет buffers только через `call!`/`forward!`; Dropout расходует свой
Generator только через `call!`/`forward!`. Eval не расходует RNG. Train/eval
распространяется также на вновь зарегистрированный child.

`zero_grad!` имеет настоящую nullary-сигнатуру и освобождает gradient buffer;
`zero_grad_buffers!` обнуляет существующие buffers, сохраняя их storage.
Обе операции вызываются без пустых скобок.

`DataLoader.each` не меняет sampler epoch. Random/DistributedSampler строят
порядок из seed и epoch; повторное чтение возвращает тот же порядок.
Recurrent modules возвращают hidden state вместо сохранения его в receiver.
Optimizer snapshots не разделяют writable storage со slots; restore сохраняет
parameter references. `grad` не накапливает `.grad`; `grad!` копирует входные
значения, чтобы не создавать writable alias к чужому snapshot.

[showcase.s](../examples/showcase.s) обучает небольшую residual CNN на MNIST:
распаковка `(features, labels)`, callable layers/loss, explicit epoch, AdamW,
cosine schedule, scoped inference, test metrics и checkpoint round trip.
Dataset, IDX-reader, residual-блок и сеть находятся в отдельных файлах
[examples/mnist](../examples/mnist). Данные читаются как `Bytes`, заголовок
IDX разбирается в Sputnik, batch выбирается одной tensor operation. Загрузка
датасета — отдельный shell script с проверкой канонических контрольных сумм.
