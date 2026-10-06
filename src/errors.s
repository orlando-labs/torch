package sputniktorch.errors


export Error, ShapeError, DTypeError, DeviceError, AutogradError, ReadOnlyTensorError, DistributedError, CheckpointError
export ONNXError

class Error < NativeError
class ShapeError < Error
class DTypeError < Error
class DeviceError < Error
class AutogradError < Error
class ReadOnlyTensorError < Error
class DistributedError < Error
class CheckpointError < Error
class ONNXError < Error
