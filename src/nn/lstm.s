package sputniktorch.nn.lstm

from sputniktorch.nn.recurrent import Recurrent

export LSTM

class LSTM < Recurrent:
  def init(input_size, hidden_size, num_layers: 1, seed: 0, device: "cpu"):
    _initialize_recurrent!("lstm", input_size, hidden_size, num_layers, :tanh, seed, device)
