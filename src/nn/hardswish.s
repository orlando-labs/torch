package sputniktorch.nn.hardswish

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import hardswish

export Hardswish

class Hardswish < Module:
  def init(): _initialize_registry!
  def forward(x): hardswish(x)

