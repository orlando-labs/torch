package torch.onnx_regression_test

from torch.checks import run as check_tensors
from torch.idioms_test import main as check_idioms
from torch.training_test import run as check_training

export run, main

# Tracing adds execution-mode arguments to the shared tensor/autograd bridge.
# Exercise the ordinary paths, including state loading and higher-order grad.
def run(device):
  check_tensors(device)
  check_idioms()
  check_training(device)
  print "PASS ordinary tensor/autograd/training after ONNX bridge on #{device}"
  0

def main(): run("cpu")
