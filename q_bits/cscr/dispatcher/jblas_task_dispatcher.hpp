#pragma once

#include <torch/torch.h>
#include "jblas/jit_blas_weight_compression.h"

#define INTERFACE_TEMPLATE                                            \
  template <class _Launcher_T, template <class _T> class _Parallel_T> \
  class Interface
#define LAUNCHER_TEMPLATE                                                                              \
  template <JBLAS_ISA _RT_ISA_T, class _GemmCore_T, template <class _T, JBLAS_ISA> class _PrologueA_T, \
            template <class _T, JBLAS_ISA> class _PrologueB_T, template <JBLAS_ISA> class _Epilogue_T> \
  class Launcher

enum QBITS_TASK {
  QBITS_QUANTIZE,
  QBITS_DEQUANTIZE,
  QBITS_LINEAR,
};

enum QBITS_DT {
  QBITS_FP32,
  QBITS_BF16,
  QBITS_FP16,
};

inline bool check_amx() { return jblas::utils::parallel::CpuDevice::getInstance()->AMX_BF16(); }
inline bool check_vnni() { return jblas::utils::parallel::CpuDevice::getInstance()->AVX_VNNI(); }
inline bool check_avx512f() { return jblas::utils::parallel::CpuDevice::getInstance()->AVX512F(); }

struct jblas_config_param {
  std::string compute_type;  // determin gemm core template
  std::string weight_type;   // determin compress-weight template
  QBITS_DT src_dt;           // determin activation related template
  QBITS_DT dst_dt;           // determin write_back template
};

struct qbits_runtime_ctx {
  torch::Tensor *activation, *weight, *bias, *output;
  bool transpose;
  int64_t blocksize, m, n, k, lda, ldo;
};

void task_dispatcher(jblas_config_param* p, qbits_runtime_ctx* ctx, const std::string& task);