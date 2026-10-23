// Model-independent, synchronous startup loader. No device work is launched.
#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
namespace etx {
// Injectable transport permits CPU contract tests without initializing a GPU.
struct loader_transport {
  virtual ~loader_transport() = default;
  virtual void peer(int reader, int owner) = 0;
  virtual void* allocate(int owner, size_t bytes, bool system) = 0;
  virtual void release(int owner, void*) noexcept = 0;
  virtual void zero(int owner, void*, size_t) = 0;
  virtual void copy(int owner, void*, const void*, size_t) = 0;
};
std::unique_ptr<loader_transport> make_loader_transport();
class loaded_plan {
 public:
  // Transport must outlive the result. Files use the schema documented in
  // manifest.py's I/O roles section. Validation precedes all allocations.
  loaded_plan(const std::string& plan, const std::string& manifest,
              const std::string& weight_map, loader_transport&);
  ~loaded_plan();
  loaded_plan(const loaded_plan&) = delete;
  loaded_plan& operator=(const loaded_plan&) = delete;
  void* argument(const std::string&) const;
  const std::vector<void*>& arguments() const { return pointers_; }
 private:
  loader_transport& transport_;
  std::vector<std::string> names_;
  std::vector<int> owners_;
  std::vector<void*> pointers_;
};
}
