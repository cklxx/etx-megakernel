#pragma once
// hipify maps <cub/cub.cuh> to hipcub (vLLM ROCm build); only kernels the import drops use it
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
