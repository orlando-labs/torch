package torch.serialization.state_dict

from torch.errors import Error
from torch.tensor.value import Tensor
from torch.device import device_text

export load, save

native def _load(path, map_location) from "amt_load" blocking
native def _save(entries, path) from "amt_save" blocking

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
