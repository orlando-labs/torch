package sputniktorch.nn.parameter_list

from sputniktorch.nn.module import Module

export ParameterList

class ParameterList < Module:
  def init(parameters = []):
    _initialize_registry!
    @items = []
    parameters.each |parameter|: append!(parameter)

  def append!(parameter):
    register_parameter!(@items.count.str, parameter)
    @items.push!(parameter)
    self

  def [](index): @items[index]
  def count(): @items.count

  def each(&block):
    @items.each(&block)
    self
