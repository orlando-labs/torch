package torch.nn.module_list

from torch.nn.module import Module

export ModuleList

class ModuleList < Module:
  def init(layers = []):
    _initialize_registry!
    @layers = []
    layers.each |layer|: append!(layer)

  def append!(layer):
    register_module!(@layers.count.str, layer)
    @layers.push!(layer)
    self

  def [](index): @layers[index]
  def count(): @layers.count

  def each(&block):
    @layers.each(&block)
    self
