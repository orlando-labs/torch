package torch.onnx.program

from torch.onnx.core import save_program

export Program

class Program:
  attr bytes
  attr node_count
  attr initializer_count
  attr opset_version
  attr input_names
  attr output_names

  def init(@bytes, @node_count, @initializer_count, @opset_version, @input_names, @output_names)
  def byte_count(): @bytes.count

  def save(path):
    save_program(@bytes, path)
    self
