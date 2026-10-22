package torch.dtype

export DType

class DType:
  attr name
  attr byte_size

  def init(name):
    @name = name.str
    sizes = {bool: 1, uint8: 1, int8: 1, int16: 2, int32: 4, int64: 8,
      float16: 2, bfloat16: 2, float32: 4, float64: 8, complex64: 8, complex128: 16}
    raise ValueError("unknown dtype: #{@name}") unless sizes.has_key?(@name)
    @byte_size = sizes[@name]

  def floating?(): @name in ["float16", "bfloat16", "float32", "float64"]
  def complex?(): @name in ["complex64", "complex128"]
  def integral?(): @name in ["uint8", "int8", "int16", "int32", "int64"]
  def bool?(): @name == "bool"
  def to_str(): @name
  def ==(other): @name == other.to_str
  def !=(other): not (self == other)
