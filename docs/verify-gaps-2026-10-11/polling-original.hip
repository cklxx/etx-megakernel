#define ETX_IMPORT_LDS 21120
#define ETX_IMPORT_SLOTS 8
#include "etx_lowering.h"
#include "etx/abi.h"
#include "etx/primitives.h"
#include "etx/import.h"
extern "C" __global__ void k_polling_wave0(etx_params p, const unsigned char* a) {
  etx_imp_touch(a, 160);
  etx_wait_DEVICE(p.events, p.ctrl_abort);
}
