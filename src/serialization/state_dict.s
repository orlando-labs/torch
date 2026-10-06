package sputniktorch.serialization.state_dict

from sputniktorch.errors import Error
from sputniktorch.tensor.value import Tensor
from sputniktorch.device import device_text

export load, save

native def _load(path, map_location) from "amt_load"
native def _save(entries, path) from "amt_save"

def load(path, map_location: "cpu"):
  location = map_location
  if Map === location:
    location = location.map |source, target|: [source, device_text(target)]
  else:
    location = device_text(location)
  result = {}
  _load(path, location).each |pair|:
    result[pair[0]] = Tensor(pair[1])
  result

def save(state, path):
  entries = state.map |name, value|: [name, value._core]
  _save(entries, path)
