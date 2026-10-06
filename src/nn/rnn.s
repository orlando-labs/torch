package sputniktorch.nn.rnn

from sputniktorch.nn.recurrent import Recurrent

export RNN

class RNN < Recurrent:
  def init(input_size, hidden_size, num_layers: 1, nonlinearity: :tanh, seed: 0, device: "cpu"):
    _initialize_recurrent!("rnn", input_size, hidden_size, num_layers, nonlinearity, seed, device)
