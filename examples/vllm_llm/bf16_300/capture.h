// Checking-only portable input capture. Included only by ETX_BF16_CAPTURE
// hosts. No code is added to the megakernel or ordinary host translation unit.
#pragma once
#include <iomanip>
struct Bf16Capture {
  std::string directory;
  const unsigned char* block = nullptr;
  void* output = nullptr;
  size_t output_offset = 0, output_bytes = 0;
  static void bytes(const std::string& name, const void* p, size_t n, bool device) {
    std::vector<unsigned char> h(n);
    if (p && device) ETX_CHECK(hipMemcpy(h.data(), p, n, hipMemcpyDeviceToHost));
    else if (p) memcpy(h.data(), p, n);
    std::ofstream f(name, std::ios::binary);
    f.write((const char*)h.data(), n);
    if (!f) die("capture write " + name);
  }
  template<class T> static T field(const unsigned char* a, size_t off) { T v; memcpy(&v, a+off, sizeof(v)); return v; }
  void before(const std::string& base, const std::string& role, int layer, int position,
              const unsigned char* a, const std::string& weights, size_t wo, size_t wn,
              int first, int count, int total_tasks, int topk) {
    directory = base + "/" + role;
    block = a;
    const int M = field<int>(a,48), K = field<int>(a,52), tpe = field<int>(a,60);
    if (field<int>(a,104) != 0 || field<int>(a,120) != 1 || first/tpe != (first+count-1)/tpe)
      die("capture supports one expert, unfolded batch-one DSL MoE only");
    for (int off : {64,72,80,88}) if (field<void*>(a,off)) die("capture norm-fold pointer");
    // _fold MoE with standalone top-k has exactly the plain silu_mul/scale
    // body (mode==0). Its logits pointer is present but never dereferenced.
    output = field<void*>(a,16);
    output_offset = (size_t)(first/tpe) * M * 2;
    output_bytes = (size_t)M * 2;
    bytes(directory+"/arguments.bin", a, 160, false);
    const size_t sizes[4] = {(size_t)(role == "moe_up" ? K : topk*K)*2, (size_t)topk*M*2,
                             (size_t)topk*4, (size_t)topk*4};
    const char* names[4] = {"input.bin", "output.bin", "ids.bin", "weights.bin"};
    for (int i=0;i<4;++i) bytes(directory+"/"+names[i], field<void*>(a,8+i*8), sizes[i], true);
    std::ofstream f(directory+"/capture.txt");
    f << layer << ' ' << position << ' ' << first << ' ' << count << ' ' << total_tasks << ' '
      << output_offset << ' ' << output_bytes << '\n' << std::quoted(weights) << ' ' << wo << ' ' << wn << '\n';
    if (!f) die("capture metadata write");
  }
  void after() {
    if (directory.empty()) return;
    bytes(directory+"/reference.bin", (const unsigned char*)output+output_offset, output_bytes, true);
    printf("BF16-CAPTURE independent original DSL launch directory=%s bytes=%zu\n", directory.c_str(), output_bytes);
    fflush(stdout);
    directory.clear();
  }
};
