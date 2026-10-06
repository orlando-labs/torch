package sputniktorch.nn.module_dict

from sputniktorch.nn.module import Module

export ModuleDict

class ModuleDict < Module:
  def init(layers = {}):
    _initialize_registry!
    layers.each |name, layer|: register_module!(name.str, layer)

  def [](name): @modules[name.str]
  def count(): @modules.count

  def each(&block):
    @modules.each(&block)
    self
