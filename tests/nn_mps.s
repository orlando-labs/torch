package torch.nn_mps_test

from torch.nn_catalog_test import run
from torch.auto_registration_test import run as run_auto_registration

export main

def main():
  run_auto_registration("mps")
  run("mps")
