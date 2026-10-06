package sputniktorch.nn.parameter_dict

from sputniktorch.nn.module import Module

export ParameterDict

class ParameterDict < Module:
  def init(parameters = {}):
    _initialize_registry!
    parameters.each |name, parameter|: register_parameter!(name.str, parameter)

  def [](name): @parameters[name.str]
  def count(): @parameters.count

  def each(&block):
    @parameters.each(&block)
    self
