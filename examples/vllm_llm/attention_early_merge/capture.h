// This header is inserted into a generated check-only copy of host.hip.
// It adds no code to the original host or to a timed megakernel.
#pragma once
#include <filesystem>
struct C6Buffer { std::string name; const void* ptr; size_t bytes; };
struct C6Capture {
  std::string directory;
  std::vector<C6Buffer> buffers;
  static void write(const std::string& path, const void* p, size_t n, bool device) {
    std::vector<char> data(n);
    if (device) ETX_CHECK(hipMemcpy(data.data(),p,n,hipMemcpyDeviceToHost));
    else memcpy(data.data(),p,n);
    std::ofstream f(path,std::ios::binary); f.write(data.data(),n);
    if (!f) die("C6 capture write " + path);
  }
  void begin(const char* base, int layer, int position, const std::vector<C6Buffer>& spans,
             const std::vector<std::pair<std::string,const unsigned char*>>& args) {
    if (!directory.empty()) die("C6 nested capture");
    directory=std::string(base)+"/p"+std::to_string(position)+"/L"+std::to_string(layer);
    if (!std::filesystem::create_directories(directory)) die("C6 capture exists: " + directory);
    buffers=spans;
    std::ofstream meta(directory+"/frame.txt");
    meta << "C6CAP1 " << layer << ' ' << position << ' ' << spans.size() << '\n';
    for (const auto& b: buffers) {
      meta << b.name << ' ' << (uint64_t)b.ptr << ' ' << b.bytes << '\n';
      write(directory+"/"+b.name+".before",b.ptr,b.bytes,true);
    }
    for (const auto& arg:args) {
      const auto* d=etx_imp_find(arg.first.c_str());
      if (!d) die("C6 missing argument descriptor");
      meta << arg.first << ' ' << d->arg_bytes << '\n';
      write(directory+"/"+arg.first+".args",arg.second,d->arg_bytes,false);
    }
    if (!meta) die("C6 metadata write");
  }
  void finish() {
    if (directory.empty()) die("C6 finish without prep");
    for (const auto& b:buffers) write(directory+"/"+b.name+".after",b.ptr,b.bytes,true);
    printf("C6-CAPTURE complete=%s\n",directory.c_str()); fflush(stdout);
    directory.clear(); buffers.clear();
  }
};
