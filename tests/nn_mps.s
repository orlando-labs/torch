package sputniktorch.nn_mps_test

from sputniktorch.nn_catalog_test import run
from sputniktorch.auto_registration_test import run as run_auto_registration

export main

def main():
  run_auto_registration("mps")
  run("mps")
