#include "runtime/sputnik_ext.h"

extern "C" uint32_t sputnik_ext_abi_version(void) {
  return SPUTNIK_EXT_ABI_VERSION;
}
