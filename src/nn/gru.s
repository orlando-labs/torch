package torch.nn.gru

from torch.nn.recurrent import Recurrent

export GRU

class GRU < Recurrent:
  def init(input_size, hidden_size, num_layers: 1, bias: true, batch_first: true,
      dropout: 0.0, bidirectional: false, seed: 0, device: "cpu", dtype: "float32"):
    _initialize_recurrent!("gru", input_size, hidden_size, num_layers, :tanh,
      bias, batch_first, dropout, bidirectional, 0, seed, device, dtype)
