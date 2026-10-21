// Host-side native capture hook for the owning resident model process.
// Call only at a source-bound completed native role boundary while all selected
// buffers are still live. Capturing an HF hidden-state array is insufficient.
// This header does not enable capture in any model executable by default.
#pragma once
#include <hip/hip_runtime_api.h>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace etx_columns {
struct NativeView { const char* name; const void* device; size_t bytes; const char* dtype; };
inline void save_native_views(const std::string& directory, const std::vector<NativeView>& views,
                              hipStream_t completed_stream, const std::string& source_bound_metadata) {
 auto check=[](hipError_t code){if(code!=hipSuccess)throw std::runtime_error("native capture HIP failure");};
 // This synchronization belongs only to an explicitly separate capture run.
 // It is forbidden inside decisive candidate/control timings.
 check(hipStreamSynchronize(completed_stream));
 std::ifstream occupied(directory+"/payload.bin",std::ios::binary);
 if(occupied.good())throw std::runtime_error("refusing native capture overwrite");
 std::ofstream payload(directory+"/payload.partial",std::ios::binary);
 std::ofstream index(directory+"/segments.json");
 if(!payload||!index)throw std::runtime_error("capture directory missing/unwritable");
 size_t offset=0;index<<"[";bool comma=false;
 for(const auto& v:views) {
  if(!v.device||!v.bytes||v.bytes%4)throw std::runtime_error("invalid aligned native view");
  std::vector<char> bytes(v.bytes);check(hipMemcpy(bytes.data(),v.device,v.bytes,hipMemcpyDeviceToHost));
  payload.write(bytes.data(),bytes.size());if(!payload)throw std::runtime_error("native export incomplete");
  if(comma)index<<",";comma=true;
  index<<"{\"name\":\""<<v.name<<"\",\"offset\":"<<offset<<",\"bytes\":"<<v.bytes<<",\"dtype\":\""<<v.dtype<<"\"}";
  offset+=v.bytes;
 }
 index<<"]\n";payload.close();index.close();
 std::ofstream metadata(directory+"/source-bound-metadata.json");metadata<<source_bound_metadata;metadata.close();
 if(std::rename((directory+"/payload.partial").c_str(),(directory+"/payload.bin").c_str()))throw std::runtime_error("native capture export rename failed");
 // No correctness, source identity, placement or completion certificate is
 // synthesized here. The owning producer must supply its exact build/task
 // receipt; runner.py rejects missing certificates even if these files exist.
}
} // namespace etx_columns
