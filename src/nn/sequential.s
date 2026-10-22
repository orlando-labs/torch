package torch.nn.sequential

from torch.nn.module import Module

export Sequential

class Sequential < Module:
  def init(layers):
    _initialize_registry!
    @layers = layers.copy
    @layers.each_with_index |layer, index|:
      register_module!(index.str, layer)

  def forward(x): @layers.reduce(x) |value, layer|: layer(value)
  def forward!(x): @layers.reduce(x) |value, layer|: layer.call!(value)
  def [](index): @layers[index]
  def count(): @layers.count
