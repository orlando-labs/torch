package torch.tensor.complex

from torch.tensor.core import binary
from torch.tensor.value import Tensor

export complex, polar, real, imag, conj, angle, view_as_real, view_as_complex

def complex(real, imag): Tensor(binary(real._core, imag._core, "complex"))
def polar(abs, angle): Tensor(binary(abs._core, angle._core, "polar"))
def real(x): x.real
def imag(x): x.imag
def conj(x): x.conj
def angle(x): x.angle
def view_as_real(x): x.view_as_real
def view_as_complex(x): x.view_as_complex
