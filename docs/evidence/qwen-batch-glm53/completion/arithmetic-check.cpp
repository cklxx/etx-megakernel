#define GLM53_CPU 1
#include "examples/glm53/native.h"
extern "C" float decode(int i) { return glm_fp8((uint8_t)i); }
extern "C" float bf(float x) { return glm_bf(x); }
