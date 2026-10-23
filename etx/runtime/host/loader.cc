#include "loader.h"
#ifndef ETX_LOADER_CPU_ONLY
#include "api.h"
#endif
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <sstream>
#include <cstring>
namespace etx {
namespace {
[[noreturn]] void fail(const std::string& s) { throw std::runtime_error("loader: " + s); }
// Small strict JSON reader: retain integer text to avoid rounding byte offsets.
struct json {
  char kind = 0; std::string value; std::map<std::string,json> object; std::vector<json> array;
  const json& at(const std::string& key) const {
    auto i=object.find(key); if(kind!='{' || i==object.end()) fail("missing field " + key); return i->second;
  }
  std::string str() const { if(kind!='s') fail("expected string"); return value; }
  size_t number() const {
    if(kind!='n'||value.empty()||value.find_first_not_of("0123456789")!=std::string::npos) fail("expected unsigned integer");
    size_t n=0; for(char c:value) { if(n>(SIZE_MAX-(c-'0'))/10) fail("integer overflow"); n=n*10+c-'0'; } return n;
  }
};
class parser {
  std::string s; size_t p=0;
  void ws(){while(p<s.size() && (s[p]==' '||s[p]=='\n'||s[p]=='\r'||s[p]=='\t'))++p;}
  char take(){if(p==s.size())fail("truncated JSON");return s[p++];}
  unsigned hex4(){unsigned v=0;for(int i=0;i<4;++i){char c=take();unsigned d;
    if(c>='0'&&c<='9')d=c-'0';else if(c>='a'&&c<='f')d=c-'a'+10;
    else if(c>='A'&&c<='F')d=c-'A'+10;else fail("invalid unicode escape");v=v*16+d;}return v;}
  void unicode(std::string& v){unsigned u=hex4();
    if(u>=0xd800&&u<=0xdbff){if(take()!='\\'||take()!='u')fail("missing low surrogate");unsigned low=hex4();if(low<0xdc00||low>0xdfff)fail("invalid low surrogate");u=0x10000+((u-0xd800)<<10)+(low-0xdc00);}
    else if(u>=0xdc00&&u<=0xdfff)fail("unpaired surrogate");
    if(u<0x80)v+=char(u);
    else if(u<0x800){v+=char(0xc0|(u>>6));v+=char(0x80|(u&63));}
    else if(u<0x10000){v+=char(0xe0|(u>>12));v+=char(0x80|((u>>6)&63));v+=char(0x80|(u&63));}
    else {v+=char(0xf0|(u>>18));v+=char(0x80|((u>>12)&63));v+=char(0x80|((u>>6)&63));v+=char(0x80|(u&63));}}
  std::string string(){std::string v;for(;;){unsigned char c=take();if(c=='"')return v;if(c<32)fail("JSON control character");if(c=='\\'){c=take();switch(c){case '"':case '\\':case '/':break;case 'u':unicode(v);continue;case 'n':c='\n';break;case 'r':c='\r';break;case 't':c='\t';break;case 'b':c='\b';break;case 'f':c='\f';break;default:fail("unsupported JSON escape");}}v+=char(c);}}
  json read(unsigned depth){if(depth>128)fail("JSON nesting");ws();json j;char c=take();j.kind=c;
    if(c=='"'){j.kind='s';j.value=string();}
    else if(c=='{'||c=='['){ws();char end=c=='{'?'}':']';if(p<s.size()&&s[p]==end){++p;return j;}for(;;){ws();if(c=='{'){if(take()!='"')fail("expected object key");auto key=string();ws();if(take()!=':')fail("expected colon");if(!j.object.emplace(key,read(depth+1)).second)fail("duplicate JSON key");}else j.array.push_back(read(depth+1));ws();char sep=take();if(sep==end)break;if(sep!=',')fail("expected comma");}}
    else {j.kind='n';j.value+=c;while(p<s.size()&&s[p]!=','&&s[p]!=']'&&s[p]!='}'&&s[p]!=' '&&s[p]!='\n'&&s[p]!='\r'&&s[p]!='\t')j.value+=s[p++];
      if(j.value!="true"&&j.value!="false"&&j.value!="null"){std::istringstream check(j.value);double d;char extra;if(!(check>>d)||(check>>extra))fail("invalid JSON literal");}}
    return j;
  }
 public:
  explicit parser(const std::string& path){std::ifstream f(path);if(!f)fail("cannot open "+path);s.assign(std::istreambuf_iterator<char>(f),{});}
  json parse(){auto j=read(0);ws();if(p!=s.size())fail("trailing JSON");return j;}
};
std::string parent(const std::string& p){auto n=p.find_last_of('/');return n==std::string::npos?".":p.substr(0,n);}
std::string safe_path(const std::string& base,const std::string& file){if(file.empty()||file=="."||file==".."||file.find('/')!=std::string::npos||file.find('\\')!=std::string::npos||file.find('\0')!=std::string::npos)fail("unsafe source file");return base+"/"+file;}
struct tensor {std::string name;size_t bytes,index;int owner;bool weight,system;};
struct range {size_t arg,src,dst,rows,stride,width;std::string path;};
}
loaded_plan::loaded_plan(const std::string& plan_path,const std::string& manifest_path,const std::string& map_path,loader_transport& transport):transport_(transport){
  auto plan=parser(plan_path).parse(), manifest=parser(manifest_path).parse(), weights=parser(map_path).parse();
  size_t ranks=plan.at("devices").number();if(!ranks||ranks>size_t(std::numeric_limits<int>::max()))fail("invalid devices");
  const auto& args=plan.at("args");const auto& io=manifest.at("io_roles");
  if(args.kind!='{'||io.kind!='{'||args.object.size()!=io.object.size()||args.object.empty())fail("argument manifest mismatch");
  std::vector<tensor> tensors(args.object.size());std::vector<bool> seen(tensors.size());
  for(const auto& kv:args.object){auto i=kv.second.number();if(i>=tensors.size()||seen[i])fail("argument indices");seen[i]=true;
    auto& m=io.at(kv.first);auto bytes=m.at("bytes").number(), owner=m.at("owner").number();auto role=m.at("role").str(), memory=m.at("memory").str();
    if(!bytes||owner>=ranks)fail("invalid allocation");
    if(role!="weight"&&role!="activation"&&role!="runtime"&&role!="scratch")fail("invalid I/O role");
    if(memory!="device"&&memory!="system")fail("invalid memory policy");
    tensors[i]={kv.first,bytes,i,int(owner),role=="weight",memory=="system"};
  }
  std::vector<range> ranges;std::vector<std::vector<std::pair<size_t,size_t>>> spans(tensors.size());
  const auto& entries=weights.at("ranges");if(entries.kind!='[')fail("expected ranges array");
  for(const auto& e:entries.array){auto name=e.at("arg").str();auto found=args.object.find(name);if(found==args.object.end())fail("unknown weight argument");auto i=found->second.number();const auto& t=tensors[i];
    auto rank=e.at("rank").number(),src=e.at("src").number(),dst=e.at("dst").number(),rows=e.at("rows").number(),stride=e.at("stride").number(),width=e.at("width").number();
    if(!t.weight||rank!=size_t(t.owner)||!rows||!width||stride<width||dst>t.bytes||rows>(t.bytes-dst)/width||src>SIZE_MAX-width||(rows-1)>(SIZE_MAX-src-width)/stride)fail("invalid shard bounds or owner");
    auto path=safe_path(parent(map_path),e.at("file").str());std::ifstream f(path,std::ios::binary|std::ios::ate);auto size=f.tellg();
    if(!f||size<0||uint64_t(size)<src+(rows-1)*stride+width)fail("truncated shard source");
    ranges.push_back({i,src,dst,rows,stride,width,path});spans[i].push_back({dst,dst+rows*width});
  }
  for(const auto& t:tensors){auto& s=spans[t.index];std::sort(s.begin(),s.end());size_t end=0;for(auto r:s){if(r.first!=end)fail("overlapping or incomplete weight");end=r.second;}if(t.weight&&end!=t.bytes)fail("incomplete weight");}
  try {
    for(size_t d=0;d<ranks;++d)for(size_t o=0;o<ranks;++o)if(d!=o)transport_.peer(int(d),int(o));
    for(const auto& t:tensors){void* p=transport_.allocate(t.owner,t.bytes,t.system);if(!p)fail("null allocation");pointers_.push_back(p);owners_.push_back(t.owner);names_.push_back(t.name);transport_.zero(t.owner,p,t.bytes);}
    std::vector<char> chunk(4*1024*1024);
    for(const auto& r:ranges){std::ifstream f(r.path,std::ios::binary);size_t total=r.rows*r.width;
      for(size_t off=0;off<total;){size_t n=std::min(chunk.size(),total-off),packed=0;while(packed<n){size_t q=off+packed,part=std::min(n-packed,r.width-q%r.width);f.seekg(r.src+(q/r.width)*r.stride+q%r.width);if(!f.read(chunk.data()+packed,part))fail("shard read failed");packed+=part;}
        transport_.copy(tensors[r.arg].owner,static_cast<char*>(pointers_[r.arg])+r.dst+off,chunk.data(),n);off+=n;}
    }
  } catch(...) {for(size_t i=0;i<pointers_.size();++i)transport_.release(owners_[i],pointers_[i]);throw;}
}
loaded_plan::~loaded_plan(){for(size_t i=0;i<pointers_.size();++i)transport_.release(owners_[i],pointers_[i]);}
void* loaded_plan::argument(const std::string& name) const {for(size_t i=0;i<names_.size();++i)if(names_[i]==name)return pointers_[i];fail("unknown argument "+name);}
#ifndef ETX_LOADER_CPU_ONLY
namespace {
namespace b=host_backend;
void checked(b::error e){if(e!=b::success)fail(b::error_string(e));}
struct native_transport:loader_transport {
 void peer(int reader,int owner) override {checked(b::set_device(reader));int can=0;checked(b::device_can_access_peer(&can,reader,owner));if(!can)fail("peer access unavailable");
 int atomics=0;
#if defined(ETX_HOST_CUDA) || (defined(__CUDACC__) && !defined(__HIPCC__))
 checked(cudaDeviceGetP2PAttribute(&atomics,cudaDevP2PAttrNativeAtomicSupported,reader,owner));
#else
 checked(hipDeviceGetP2PAttribute(&atomics,hipDevP2PAttrNativeAtomicSupported,reader,owner));
#endif
 if(!atomics)fail("native peer atomics unavailable");
 auto e=b::enable_peer_access(owner);
#if defined(ETX_HOST_CUDA) || (defined(__CUDACC__) && !defined(__HIPCC__))
 if(e==cudaErrorPeerAccessAlreadyEnabled){(void)cudaGetLastError();return;}
#else
 if(e==hipErrorPeerAccessAlreadyEnabled){(void)hipGetLastError();return;}
#endif
 checked(e);}
 void* allocate(int owner,size_t n,bool system) override {checked(b::set_device(owner));void* p=nullptr;
 // CUDA SYSTEM control/events live on their owning device with peer access,
 // never in managed memory: concurrent polling must not trigger migration.
#if defined(ETX_HOST_CUDA) || (defined(__CUDACC__) && !defined(__HIPCC__))
 (void)system;checked(b::malloc(&p,n));
#else
 if(system)checked(b::malloc_fine_grained(&p,n));else checked(b::malloc(&p,n));
#endif
 return p;}
 void release(int owner,void* p) noexcept override {if(b::set_device(owner)==b::success)(void)b::free(p);}
 void zero(int owner,void* p,size_t n) override {checked(b::set_device(owner));checked(b::memset(p,0,n));}
 void copy(int owner,void* dst,const void* src,size_t n) override {checked(b::set_device(owner));checked(b::memcpy(dst,src,n,b::host_to_device));}
};
}
std::unique_ptr<loader_transport> make_loader_transport(){return std::unique_ptr<loader_transport>(new native_transport);}
#endif
}
