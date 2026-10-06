package sputniktorch.nn_families_test

import sputniktorch
from sputniktorch.nn import ReLU, Sigmoid, Tanh, GELU, SiLU, Mish, LeakyReLU
from sputniktorch.nn import ELU, SELU, Softplus, Softsign, Tanhshrink, Hardtanh, Hardsigmoid
from sputniktorch.nn import Hardswish, LogSigmoid, Softshrink, Hardshrink, Softmax, LogSoftmax, Softmin
from sputniktorch.nn import ReLU6, Threshold, PReLU, Identity, GLU, Linear, Bilinear
from sputniktorch.nn import Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d, MaxPool1d
from sputniktorch.nn import AvgPool1d, AdaptiveAvgPool1d, AdaptiveMaxPool1d, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, AdaptiveMaxPool2d
from sputniktorch.nn import MaxPool3d, AvgPool3d, AdaptiveAvgPool3d, AdaptiveMaxPool3d, BatchNorm1d, BatchNorm2d, BatchNorm3d
from sputniktorch.nn import ConstantPad1d, ReflectionPad1d, ReplicationPad1d, ConstantPad2d, ReflectionPad2d, ReplicationPad2d, ConstantPad3d
from sputniktorch.nn import ReflectionPad3d, ReplicationPad3d, GroupNorm, Softmax2d, MSELoss, L1Loss, SmoothL1Loss
from sputniktorch.nn import HuberLoss, BCEWithLogitsLoss, BCELoss, NLLLoss, KLDivLoss, ModuleDict, ParameterList
from sputniktorch.nn import ParameterDict
from sputniktorch.nn.init import xavier_uniform!, xavier_normal!, kaiming_uniform!, kaiming_normal!, zeros!, ones!
from sputniktorch.checks import check

export main

def main():
  x = sputniktorch.linspace(-2.0, 2.0, steps: 12).reshape([3, 4]).requires_grad!(true)
  activations = [ReLU(), Sigmoid(), Tanh(), GELU(), SiLU(), Mish(),
    LeakyReLU(), ELU(), SELU(), Softplus(), Softsign(), Tanhshrink(),
    Hardtanh(), Hardsigmoid(), Hardswish(), LogSigmoid(), Softshrink(),
    Hardshrink(), Softmax(), LogSoftmax(), Softmin(), ReLU6(),
    Threshold(0.0, 0.0), PReLU(), Identity()]
  activations.each |layer|:
    output = layer(x)
    check(output.shape == x.shape, "activation layout")
    output.sum().backward!()
    check(x.grad != null, "activation backward")
    x.zero_grad!
  check(GLU()(x).shape == [3, 2], "GLU gate split")
  check(Linear(4, 2, bias: false)(x).shape == [3, 2], "bias-free dense layer")
  Bilinear(4, 4, 2)(x, x).sum().backward!()

  images = [sputniktorch.randn([2, 2, 8], seed: 0),
    sputniktorch.randn([2, 2, 8, 8], seed: 1), sputniktorch.randn([2, 2, 8, 8, 8], seed: 2)]
  convolutions = [Conv1d(2, 4, 2, groups: 2), Conv2d(2, 4, 2, groups: 2), Conv3d(2, 4, 2, groups: 2)]
  transposed = [ConvTranspose1d(2, 4, 2), ConvTranspose2d(2, 4, 2), ConvTranspose3d(2, 4, 2)]
  pools = [[MaxPool1d(2), AvgPool1d(2), AdaptiveAvgPool1d(2), AdaptiveMaxPool1d(2)],
    [MaxPool2d(2), AvgPool2d(2), AdaptiveAvgPool2d(2), AdaptiveMaxPool2d(2)],
    [MaxPool3d(2), AvgPool3d(2), AdaptiveAvgPool3d(2), AdaptiveMaxPool3d(2)]]
  normalizers = [BatchNorm1d(4), BatchNorm2d(4), BatchNorm3d(4)]
  padding = [[ConstantPad1d(1), ReflectionPad1d(1), ReplicationPad1d(1)],
    [ConstantPad2d(1), ReflectionPad2d(1), ReplicationPad2d(1)],
    [ConstantPad3d(1), ReflectionPad3d(1), ReplicationPad3d(1)]]
  3.times |index|:
    output = convolutions[index](images[index])
    normalizers[index].call!(output).square.mean().backward!()
    check(convolutions[index].weight.grad != null, "grouped convolution and normalization backward")
    expanded = transposed[index](images[index])
    expanded.sum().backward!()
    check(transposed[index].weight.grad != null, "transposed convolution backward")
    pools[index].each |pool|:
      input = images[index].copy.requires_grad!(true)
      pool(input).sum().backward!()
      check(input.grad != null, "pooling backward")
    padding[index].each |pad|:
      input = images[index].copy.requires_grad!(true)
      pad(input).sum().backward!()
      check(input.grad != null, "padding backward")
  check(GroupNorm(2, 2)(images[1]).shape == images[1].shape, "GroupNorm")
  check(Softmax2d()(images[1]).shape == images[1].shape, "spatial softmax")

  target = sputniktorch.zeros_like(x)
  [MSELoss(), L1Loss(), SmoothL1Loss(), HuberLoss(), BCEWithLogitsLoss()].each |loss|:
    loss(x, target).backward!()
    check(x.grad != null, "regression loss backward")
    x.zero_grad!
  BCELoss()(x.sigmoid, target).backward!()
  NLLLoss()(x.log_softmax(dim: 1), sputniktorch.tensor([0, 1, 2])).backward!()
  KLDivLoss()(x.log_softmax(dim: 1), sputniktorch.full(x.shape, 0.25)).backward!()

  weights = sputniktorch.zeros([4, 4], requires_grad: true)
  [ &xavier_uniform!, &xavier_normal!, &kaiming_uniform!, &kaiming_normal! ].each |initialize|:
    initialize(weights, seed: 9)
    first = weights.to_a
    initialize(weights, seed: 9)
    check(weights.to_a == first and weights.leaf?, "deterministic initialization of leaf parameters")
  zeros!(weights)
  check(weights.sum().item == 0, "zero initialization")
  ones!(weights)
  check(weights.sum().item == 16, "one initialization")

  modules = ModuleDict({dense: Linear(4, 2), activation: ReLU()})
  parameters = ParameterList(modules.parameters)
  named = ParameterDict({weight: parameters[0]})
  check(named["weight"]._identity == parameters[0]._identity, "parameter container identity")
  duplicate_refused = false
  try:
    modules.register_buffer!("dense", weights)
  rescue ValueError |error|:
    duplicate_refused = true
  check(duplicate_refused, "parameter/module/buffer namespace is shared")
  cycle_refused = false
  try:
    modules.register_module!("self", modules)
  rescue ValueError |error|:
    cycle_refused = true
  check(cycle_refused, "recursive module registration is rejected")
  print "PASS NN families + initialization + containers on CPU"
  0
