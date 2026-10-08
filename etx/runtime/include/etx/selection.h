// Host side of the variant selection (docs/PLAN-generic-decoder-2026-10-08.md section 6.3).
// Include plan_data.h before this file. plan_data.h publishes the planner's selection: etx_sel_phase,
// etx_sel_export and etx_sel_kind (etx/codegen/manifest.py selection_block). The host reads the selection here
// only. It does not derive it again from the environment, from a compile define or from the model shapes.
// Plain C++: no HIP. tests/test_selection_manifest.py compiles and runs it on the CPU.
#pragma once
#include <string.h>

#ifndef ETX_SEL_N_PHASES
#error "plan_data.h publishes no variant selection: build the plan with the planner (etx/codegen/manifest.py)"
#endif
#if ETX_SEL_THREADS != ETX_THREADS
#error "the selection budget's workgroup size differs from the plan's ETX_THREADS"
#endif

// The index of a phase key, or -1.
static inline int etx_sel_find(const char* phase) {
  for (int i = 0; i < ETX_SEL_N_PHASES; ++i)
    if (!strcmp(etx_sel_phase[i], phase)) return i;
  return -1;
}

// The entry of one plan instance: "<role>@0" for layer 0 when that key exists, else "<role>".
// The Python twin is etx.codegen.manifest.phase_key().
static inline int etx_sel_find_instance(const char* role, int layer) {
  if (layer == 0) {
    char key[64];
    const size_t n = strlen(role);
    if (n + 3 <= sizeof key) {
      memcpy(key, role, n); memcpy(key + n, "@0", 3);
      const int i = etx_sel_find(key);
      if (i >= 0) return i;
    }
  }
  return etx_sel_find(role);
}

// nullptr when the plan instance runs the export that the selection names for its phase, else the reason.
static inline const char* etx_sel_check(const char* role, int layer, const char* exp) {
  const int i = etx_sel_find_instance(role, layer);
  if (i < 0) return "the selection has no entry for this phase";
  if (strcmp(etx_sel_export[i], exp)) return "the plan instance's export differs from the selection";
  return nullptr;
}
