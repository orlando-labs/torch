package torch.fft

from torch.autograd.context import execution_mode
from torch.tensor.value import Tensor
from torch.tensor.core import factory
from torch.device import device_text

export fft, ifft, fft2, ifft2, fftn, ifftn, rfft, irfft, rfft2, irfft2, rfftn, irfftn
export hfft, ihfft, hfft2, ihfft2, hfftn, ihfftn, fftfreq, rfftfreq, fftshift, ifftshift

native def _fft(op as Str, inputs, options, mode as Int) from "amt_fft"

def _dims(dim): if dim == null then null else (if Int === dim then [dim] else dim.array)

def _transform(op, x, size, dim, norm):
  Tensor(_fft(op, [x._core], [size, dim, norm.str], execution_mode()))

def fft(x, n: null, dim: -1, norm: :backward): _transform("fft", x, n, dim, norm)
def ifft(x, n: null, dim: -1, norm: :backward): _transform("ifft", x, n, dim, norm)
def rfft(x, n: null, dim: -1, norm: :backward): _transform("rfft", x, n, dim, norm)
def irfft(x, n: null, dim: -1, norm: :backward): _transform("irfft", x, n, dim, norm)
def hfft(x, n: null, dim: -1, norm: :backward): _transform("hfft", x, n, dim, norm)
def ihfft(x, n: null, dim: -1, norm: :backward): _transform("ihfft", x, n, dim, norm)

def fftn(x, s: null, dim: null, norm: :backward): _transform("fftn", x, _dims(s), _dims(dim), norm)
def ifftn(x, s: null, dim: null, norm: :backward): _transform("ifftn", x, _dims(s), _dims(dim), norm)
def rfftn(x, s: null, dim: null, norm: :backward): _transform("rfftn", x, _dims(s), _dims(dim), norm)
def irfftn(x, s: null, dim: null, norm: :backward): _transform("irfftn", x, _dims(s), _dims(dim), norm)
def hfftn(x, s: null, dim: null, norm: :backward): _transform("hfftn", x, _dims(s), _dims(dim), norm)
def ihfftn(x, s: null, dim: null, norm: :backward): _transform("ihfftn", x, _dims(s), _dims(dim), norm)

def fft2(x, s: null, dim: [-2, -1], norm: :backward): fftn(x, s:, dim:, norm:)
def ifft2(x, s: null, dim: [-2, -1], norm: :backward): ifftn(x, s:, dim:, norm:)
def rfft2(x, s: null, dim: [-2, -1], norm: :backward): rfftn(x, s:, dim:, norm:)
def irfft2(x, s: null, dim: [-2, -1], norm: :backward): irfftn(x, s:, dim:, norm:)
def hfft2(x, s: null, dim: [-2, -1], norm: :backward): hfftn(x, s:, dim:, norm:)
def ihfft2(x, s: null, dim: [-2, -1], norm: :backward): ihfftn(x, s:, dim:, norm:)

def fftfreq(n, d: 1.0, dtype: "float32", device: "cpu"):
  Tensor(factory("fftfreq", [n], dtype.str, device_text(device), false, d))

def rfftfreq(n, d: 1.0, dtype: "float32", device: "cpu"):
  Tensor(factory("rfftfreq", [n], dtype.str, device_text(device), false, d))

def fftshift(x, dim: null): Tensor(_fft("fftshift", [x._core], [_dims(dim)], execution_mode()))
def ifftshift(x, dim: null): Tensor(_fft("ifftshift", [x._core], [_dims(dim)], execution_mode()))
