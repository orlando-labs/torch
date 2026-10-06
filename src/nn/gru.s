package sputniktorch.nn.gru

from sputniktorch.nn.recurrent import Recurrent

export GRU

class GRU < Recurrent:
  def init(input_size, hidden_size, num_layers: 1, seed: 0, device: "cpu"):
    _initialize_recurrent!("gru", input_size, hidden_size, num_layers, :tanh, seed, device)
