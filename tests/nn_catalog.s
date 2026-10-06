package sputniktorch.nn_catalog_test

import sputniktorch
from sputniktorch.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d, Flatten, Linear, Sequential
from sputniktorch.nn import Embedding, LayerNorm, RMSNorm, GELU, CrossEntropyLoss, MultiheadAttention, Dropout
from sputniktorch.nn import Transformer, RNNCell, LSTMCell, GRUCell, ConstantPad2d, PixelShuffle, PixelUnshuffle, Unfold, Fold
from sputniktorch.nn import RNN, LSTM, GRU
from sputniktorch.autograd import no_grad
from sputniktorch.nn.functional import scaled_dot_product_attention
from sputniktorch.checks import check

export main, run

def run(device):
  model = Sequential([
    Conv2d(1, 3, 2, seed: 7, device:),
    BatchNorm2d(3, device:), ReLU(), MaxPool2d(2),
    Flatten(), Linear(12, 2, seed: 8, device:)
  ])
  x = sputniktorch.randn([2, 1, 6, 6], seed: 9, device:)
  refused = false
  try:
    model(x)
  rescue ValueError |error|:
    refused = true
  check(refused, "stateful Sequential preflight")
  refused = false
  try:
    model[1].forward(model[0](x))
  rescue ValueError |error|:
    refused = true
  check(refused, "direct non-mutating BatchNorm forward is guarded")
  check(model[1].num_batches_tracked.item == 0, "no partial running stats update")
  output = model.call!(x)
  check(output.shape == [2, 2], "CNN shape")
  output.square.mean().backward!()
  check(model[0].weight.grad != null, "convolution gradient")
  check(model[1].num_batches_tracked.item == 1, "BatchNorm training state")
  saved = model.state_dict
  model.eval!
  model(x)
  check(model[1].num_batches_tracked.item == 1, "eval is observational")
  model.load_state_dict!(saved)

  embedding = Embedding(8, 4, padding_idx: 0, device:)
  indices = sputniktorch.tensor([[0, 1], [2, 3]], device:)
  encoded = embedding(indices)
  check(encoded.shape == [2, 2, 4], "embedding")
  check(encoded[0][0].sum().item == 0, "padding embedding")
  normalized = LayerNorm(4, device:)(encoded)
  check(normalized.shape == encoded.shape, "LayerNorm")
  check(RMSNorm(4, device:)(encoded).shape == encoded.shape, "RMSNorm")
  check(GELU()(normalized).shape == encoded.shape, "GELU")
  attention = MultiheadAttention(4, 2, device:)
  attention(encoded, causal: true).square.sum().backward!()
  check(attention.query_projection.weight.grad != null, "attention gradients")

  q = sputniktorch.ones([1, 1, 2, 2], device:, requires_grad: true)
  values = sputniktorch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], device:)
  causal = scaled_dot_product_attention(q, q, values, causal: true)
  check(causal.to_a == [[[[1.0, 2.0], [2.0, 3.0]]]], "causal attention")
  mask = sputniktorch.zeros([2, 2], dtype: "bool", device:)
  blocked = scaled_dot_product_attention(q, q, values, mask:)
  check(blocked.sum().item == 0, "fully masked attention rows")
  blocked.sum().backward!()
  check(q.grad.abs.sum().item == 0, "fully masked gradients")

  logits = sputniktorch.zeros([2, 3], device:, requires_grad: true)
  target = sputniktorch.tensor([0, 2], device:)
  loss = CrossEntropyLoss()(logits, target)
  check((loss.item - 1.0986123).abs < 0.00001, "cross entropy reference")
  loss.backward!()

  dropout = Dropout(p: 0.5, seed: 12)
  rng_state = dropout.rng_state
  first = dropout.call!(sputniktorch.ones([16], device:))
  dropout.load_rng_state!(rng_state)
  check(first.to_a == dropout.call!(sputniktorch.ones([16], device:)).to_a, "explicit RNG replay")

  transformer = Transformer(4, 2, num_encoder_layers: 1, num_decoder_layers: 1,
    dim_feedforward: 8, dropout: 0.2, device:)
  encoded = encoded.detach
  transformed = transformer.call!(encoded, encoded)
  check(transformed.shape == [2, 2, 4], "Transformer encoder/decoder layout")
  transformed.square.mean().backward!()
  complete_backward = transformer.parameters.all?: $it.grad != null
  check(complete_backward, "Transformer backward across every parameter")
  transformer.eval!
  check(transformer(encoded, encoded).shape == transformed.shape, "Transformer eval")

  cell_input = sputniktorch.ones([2, 3], device:)
  rnn = RNNCell(3, 2, device:)
  lstm = LSTMCell(3, 2, device:)
  gru = GRUCell(3, 2, device:)
  # Zero projections give a closed-form reference for all three gate systems.
  no_grad():
    [rnn, lstm, gru].each |cell|:
      cell.parameters.each |p|: p.zero!
  previous = sputniktorch.ones([2, 2], device:, requires_grad: true)
  check(rnn(cell_input, hidden: previous).sum().item == 0, "RNN cell reference")
  (hidden, memory) = lstm(cell_input, state: (previous, previous))
  check(memory.to_a == [[0.5, 0.5], [0.5, 0.5]], "LSTM forget/input gates")
  check((hidden[0][0].item - 0.2310586).abs < 1e-5, "LSTM output gate")
  next_hidden = gru(cell_input, hidden: previous)
  check(next_hidden.to_a == [[0.5, 0.5], [0.5, 0.5]], "GRU update gate")
  (next_hidden.sum() + hidden.sum()).backward!()
  check(previous.grad != null, "recurrent state gradients")
  sequence = sputniktorch.randn([2, 3, 3], seed: 11, device:)
  [RNN(3, 2, num_layers: 2, device:), LSTM(3, 2, num_layers: 2, device:),
    GRU(3, 2, num_layers: 2, device:)].each |model|:
    (output, state) = model(sequence)
    check(output.shape == [2, 3, 2], "stacked recurrent layout")
    output.square.mean().backward!()
    check(model.cells[0].input_projection.weight.grad != null, "backpropagation through time")
    (continued, final_state) = model(sequence, state:)
    check(continued.shape == output.shape, "explicit recurrent state")

  image = sputniktorch.arange(16, dtype: "float32", device:).reshape([1, 1, 4, 4])
  patches = Unfold(2)(image)
  reconstructed = Fold([4, 4], 2)(patches)
  check(reconstructed.shape == image.shape, "fold/unfold")
  pixels = sputniktorch.arange(16, dtype: "float32", device:).reshape([1, 4, 2, 2])
  check(PixelUnshuffle(2)(PixelShuffle(2)(pixels)).to_a == pixels.to_a, "pixel shuffle inverse")
  check(ConstantPad2d(1)(image).shape == [1, 1, 6, 6], "spatial padding")
  print "PASS NN catalog on #{device}"
  0

def main(): run("cpu")
