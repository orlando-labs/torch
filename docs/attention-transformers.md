# Attention и Transformer

Проекции, layout, masks, stacks и управление dropout написаны на Sputnik.
C++ boundary вызывает ATen primitives и SDPA. Python не нужен для сборки,
исполнения или тестов. Контракт сопоставлен с официальными
[MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
и [SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention).

## Functional SDPA

```sputnik
scaled_dot_product_attention(query, key, value, mask: null, attn_mask: null,
  causal: false, is_causal: null, scale: null, enable_gqa: false,
  dropout_p: 0.0, dropout_mask: null)
scaled_dot_product_attention!(query, key, value, generator, mask: null,
  attn_mask: null, causal: false, is_causal: null, scale: null,
  enable_gqa: false, dropout_p: 0.0)
```

`mask:` и `attn_mask:` здесь синонимы: Bool `true` разрешает связь, floating
mask прибавляется к logits. Одновременная передача двух имён отклоняется.
Mask должна broadcast к scores, находиться на том же device и иметь Bool
либо dtype входа. Поддержаны rank >= 2, broadcast ведущих осей, разные
query/key lengths и отдельная последняя размерность value.

`causal:`/`is_causal:` задают верхне-левую triangular mask, в том числе для
прямоугольных sequences; её можно сочетать с явной mask. Противоречащие
значения aliases отклоняются. `scale:` заменяет стандартное `1/sqrt(E)`.
GQA требует равных key/value heads, делящих query heads; повторение heads
выполняется Sputnik `index_select`, сохраняя gradients исходных tensors.

Без dropout functional вызывает ATen SDPA. Выбор конкретного math/fused
kernel зависит от backend, dtype и формы; наличие Flash kernel не обещается.
При dropout или ONNX trace используется явная композиция matmul/softmax.
Fully masked rows дают zero values и zero query/key/value gradients в обоих
путях. Безопасный softmax не создаёт NaN intermediate для строк из `-inf`.

Functional dropout действует независимо от train/eval: `dropout_p` задаёт
вызывающий код. При `0 < p < 1` чистой форме нужна Bool `dropout_mask:` точной
формы probabilities. Bang-форма использует переданный Generator. `p: 0/1`
не расходует RNG. Generator остаётся явным CPU RNG с переносом маски на
device входа; это не автоматический CPU fallback attention kernels.

## MultiheadAttention

```sputnik
MultiheadAttention(embed_dim, num_heads, dropout: 0.0, bias: true,
  add_bias_kv: false, add_zero_attn: false, kdim: null, vdim: null,
  batch_first: true, seed: 0, device: "cpu", dtype: "float32")
model(query, key: null, value: null, mask: null, attn_mask: null,
  key_padding_mask: null, causal: false, is_causal: null,
  need_weights: false, average_attn_weights: true)
```

Sputnik defaults сохранены: `batch_first: true` и `need_weights: false`,
возвращающий **Tensor**, а не `(Tensor, null)`. При `need_weights: true`
возвращается `(output, weights)`. Это явное отличие от upstream default.
Входы: `[N,L,E]` либо `[L,N,E]` при `batch_first: false`; unbatched `[L,E]`
игнорирует layout flag. Key/value могут иметь свои feature widths `kdim/vdim`.

Маски имеют два явно разделённых контракта:

- Старое `mask:` сохраняет SDPA semantics: Bool `true` разрешает связь;
  допускается broadcast к `[N,heads,L,S]`.
- Torch-style `attn_mask:`: Bool `true` запрещает связь, floating values
  прибавляются к logits. Форма `[L,S]` либо `[N*heads,L,S]`; для unbatched
  input вторая форма `[heads,L,S]`.
- `key_padding_mask:`: Bool `true` скрывает key, floating values добавляют
  bias. Форма `[N,S]` либо `[S]` для unbatched input. Её можно сочетать с
  любой одной attention mask, включая смешанные Bool/floating masks.

Weights имеют форму `[N,L,S]` при усреднении heads, иначе `[N,heads,L,S]`.
Unbatched input убирает только batch axis. Возвращаются probabilities после
attention dropout. `add_bias_kv` добавляет обучаемые leaf parameters
`bias_k/bias_v`, `add_zero_attn` — zero slot. Дополнительные slots открыты в
masks и causal mode. Выход полностью скрытой строки после output projection
может содержать `output_projection.bias`, хотя attention value равен нулю.

При training и `0 < dropout < 1` `call!`/`forward!` расходуют RNG; обычный
вызов отклоняется до исполнения. `rng_state`/`load_rng_state!` дают replay.
Eval и endpoints dropout не меняют RNG. Проверка input/mask и проекции
проходят до sampling; ошибка их валидации не расходует RNG.

State-dict keys остаются Sputnik projection submodules (`query`, `key`,
`value`, `output`), не upstream packed `in_proj_weight`. Gaussian initialization
линейных weights и zero biases сохранены; одинаковый seed не обещает
upstream initialization parity.

## Transformer

Encoder/decoder layers, stacks и `Transformer` прокидывают `batch_first`,
`bias`, `dtype`, `layer_norm_eps` и `activation: :gelu/:relu`.
Старые defaults `batch_first: true`, GELU и decoder causal mode сохранены.
Dropout теперь применяется также к attention probabilities, а не только
residual/feedforward branches. Bang вызовы рекурсивно доходят до MHA RNG.

Layers/stacks принимают `attn_mask`, `key_padding_mask`; decoder дополнительно
`memory_attn_mask`, `memory_key_padding_mask` и `memory_causal`.
Старые `mask`/`memory_mask` сохраняют allowed semantics. На `Transformer`
эти kwargs называются `source_attn_mask`, `target_attn_mask`,
`memory_attn_mask`, `source_key_padding_mask`, `target_key_padding_mask`,
`memory_key_padding_mask`, `source_causal`, `target_causal`, `memory_causal`.
Unbatched input и read-only/noncontiguous layout также поддержаны.

## Проверки и оставшиеся ограничения

`tests/attention.s` и `tests/attention_mps.s` проверяют outputs, weights,
input/parameter/additive-mask gradients, разные layouts/dimensions, masks,
GQA, extra slots, dropout replay и Transformer composition.
`tests/attention_reference.cpp` использует независимые CPU ATen SDPA и
LibTorch C++ functional MHA с теми же весами. Он линкуется только в attention
tests; production adapter его не содержит. Сравнение MPS с CPU-эталоном
делает явные copies исключительно для численного QA.

VM и full-native tests пройдены на CPU (float32/float64) и реальном MPS
(float32), с `PYTORCH_ENABLE_MPS_FALLBACK=0`. Native build содержит 2208/2208
native function bodies, full body coverage, zero VM/bytecode fallback и
VM-independent executable. Запуск с `PATH=/nonexistent` и dylib audit
подтверждают отсутствие Python runtime и `torch_python`. Проверенный
LibTorch baseline: 2.14.0; поддержка bootstrap для 2.14.1 не считается
результатом проверки этой версии.

Регрессии `nn_catalog`, `nn_families` и ONNX
operators также пройдены; ONNX structural export проверен на CPU/MPS.
Независимый ONNX Runtime validator в этой проверке не запускался: optional
C++ runtime в текущем workspace не установлен.

Cached decoding/static projected key-value API, NestedTensor fastpath,
custom activation callable и upstream constructor cloning contracts для
Transformer stacks, а также отдельные final-norm options пока не реализованы.
ONNX export использует composition, не fused SDPA; MHA reshapes по-прежнему
требуют проверенной фиксированной формы, как описано в [onnx.md](onnx.md).
