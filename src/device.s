package sputniktorch.device

export Device, device_text

class Device:
  def init(name):
    @name = name.to_str
    if @name == "mps:0":
      @name = "mps"
    if @name == "cpu:0":
      @name = "cpu"

  def to_str(): @name
  def type(): @name.split(":")[0]

def device_text(value): value.to_str
