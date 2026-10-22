package torch.nn.lstm

from torch.nn.recurrent import Recurrent

export LSTM

class LSTM < Recurrent:
  def init(input_size, hidden_size, num_layers: 1, bias: true, batch_first: true,
      dropout: 0.0, bidirectional: false, proj_size: 0, seed: 0, device: "cpu", dtype: "float32"):
    _initialize_recurrent!("lstm", input_size, hidden_size, num_layers, :tanh,
      bias, batch_first, dropout, bidirectional, proj_size, seed, device, dtype)
