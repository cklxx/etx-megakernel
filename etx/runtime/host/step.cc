#include "step.h"
#include <cmath>
#include <limits>
namespace {
bool valid(const etx_step_descriptor* d) {
  return d && d->abi_version==ETX_STEP_ABI_VERSION &&
    d->struct_size==sizeof(*d) && !d->reserved && d->backend &&
    d->backend->reset && d->backend->launch && d->backend->wait && d->ranks &&
    std::isfinite(d->timeout_s) && d->timeout_s>0 && d->position>=0 &&
    d->position<std::numeric_limits<int64_t>::max();
}
}
extern "C" int etx_step(etx_step_descriptor* d) {
  if (!valid(d)) return ETX_INVALID;
  // Hooks see a copy: a failed prepare/finish cannot advance decode state.
  auto s=*d;
  s.next_token=s.token;
  try {
    if (s.prepare && s.prepare(s.context,&s)) return ETX_BACKEND_ERROR;
    // Hooks may update next_token, never the control contract.
    if (s.backend!=d->backend || s.context!=d->context || s.ranks!=d->ranks ||
        s.timeout_s!=d->timeout_s) return ETX_INVALID;
    for(uint32_t r=0;r<d->ranks;++r)
      if(d->backend->reset(d->context,r,d->timeout_s)) return ETX_BACKEND_ERROR;
    uint32_t launched=0; bool failed=false;
    try {
      for(;launched<d->ranks;++launched)
        if(d->backend->launch(d->context,launched,d->timeout_s)) {failed=true;break;}
    } catch (...) { failed=true; }
    // Drain the successful prefix, including when a launch callback throws.
    for(uint32_t r=0;r<launched;++r) {
      try { if(d->backend->wait(d->context,r,d->timeout_s)) failed=true; }
      catch (...) { failed=true; }
    }
    if(failed) return ETX_BACKEND_ERROR;
    if(s.finish && s.finish(s.context,&s)) return ETX_BACKEND_ERROR;
    d->next_token=s.next_token; d->token=s.next_token; ++d->position;
    return ETX_OK;
  } catch (...) { return ETX_BACKEND_ERROR; }
}
extern "C" int etx_decode(etx_step_descriptor* d,uint64_t n,int32_t eos,
                           int32_t* tokens,uint64_t* completed) {
  if(!completed) return ETX_INVALID;
  *completed=0;
  if(!valid(d) || (n && !tokens)) return ETX_INVALID;
  for(uint64_t i=0;i<n;++i) {
    int status=etx_step(d); if(status) return status;
    tokens[i]=d->token; ++*completed;
    if(eos>=0 && d->token==eos) break;
  }
  return ETX_OK;
}
extern "C" const char* etx_status_string(int s) {
  switch(s) {case ETX_OK:return "OK";case ETX_INVALID:return "invalid descriptor";
    case ETX_BACKEND_ERROR:return "backend failure";default:return "unknown status";}
}
