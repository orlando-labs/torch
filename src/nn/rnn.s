package torch.nn.rnn

from torch.nn.recurrent import Recurrent

export RNN

class RNN < Recurrent:
  def init(input_size, hidden_size, num_layers: 1, nonlinearity: :tanh, bias: true,
      batch_first: true, dropout: 0.0, bidirectional: false, seed: 0, device: "cpu", dtype: "float32"):
    _initialize_recurrent!("rnn", input_size, hidden_size, num_layers, nonlinearity,
      bias, batch_first, dropout, bidirectional, 0, seed, device, dtype)
