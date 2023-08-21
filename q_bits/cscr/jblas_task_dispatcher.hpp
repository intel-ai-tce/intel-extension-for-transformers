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

template <class KERNEL>
void jblas_quantize(jblas_config_param* p, qbits_runtime_ctx* ctx) {
  using PrologueB = typename KERNEL::WeightType;
  PrologueB compress_kernel;
}

template <QBITS_TASK TASK, class KERNEL>
void execute_task(jblas_config_param* p, qbits_runtime_ctx* ctx) {
  switch (TASK) {
    case QBITS_QUANTIZE:
      return jblas_quantize<KERNEL>(p, ctx);
      // case QBITS_DEQUANTIZE:
      //   return jblas_dequantize<KERNEL>(p, ctx);
      // case QBITS_LINEAR:
      //   return jblas_gemm<KERNEL>(p, ctx);
  }
}

template <QBITS_TASK TASK, INTERFACE_TEMPLATE, LAUNCHER_TEMPLATE, class Gemmcore, template <class _T> class Parallel,
          JBLAS_ISA ISA, template <class _T, JBLAS_ISA> class PrologueB, template <class _T, JBLAS_ISA> class PrologueA>
void parse_store(jblas_config_param* p, qbits_runtime_ctx* ctx) {
  if (p->dst_dt == QBITS_FP32) {
    using namespace jblas::epilogue::gemm;
    return execute_task<TASK,
                        Interface<Launcher<ISA, Gemmcore, PrologueA, PrologueB, AccumulatorWriteBackFp32>, Parallel>>(
        p, ctx);
  }
  TORCH_CHECK(false, "unsupported dst data type.");
}

template <QBITS_TASK TASK, INTERFACE_TEMPLATE, LAUNCHER_TEMPLATE, class Gemmcore, template <class _T> class Parallel,
          JBLAS_ISA ISA, template <class _T, JBLAS_ISA> class PrologueB>
void parse_activation(jblas_config_param* p, qbits_runtime_ctx* ctx) {
  using namespace jblas::prologue::gemm;
  if (p->compute_type == "int8" && p->src_dt == QBITS_FP32 && check_amx()) {
    return parse_store<TASK, Interface, Launcher, Gemmcore, Parallel, ISA, PrologueB, ActivationF32S8KBlockQuantize>(
        p, ctx);
  }
  TORCH_CHECK(false, "unsupported src data type.");
}

template <QBITS_TASK TASK, INTERFACE_TEMPLATE, LAUNCHER_TEMPLATE, class Gemmcore, template <class _T> class Parallel,
          JBLAS_ISA ISA>
void parse_weight(jblas_config_param* p, qbits_runtime_ctx* ctx) {
  using namespace jblas::prologue::weight_comp::gemm_kblcok;
  if (p->weight_type == "s8_scalef32") {
    return parse_activation<TASK, Interface, Launcher, Gemmcore, Parallel, ISA, WeightS8ScaleFp32>(p, ctx);
  }
  if (p->weight_type == "s4clip_scalef32") {
  }
  if (p->weight_type == "s4fullrange_scalef32") {
  }
  TORCH_CHECK(false, "unsupported weight_type, weight_type==" + p->weight_type);
}

template <QBITS_TASK TASK>
void parse_gemm_core(jblas_config_param* p, qbits_runtime_ctx* ctx) {
  if (p->compute_type == "int8") {
    if (check_amx()) {
      return parse_weight<TASK, jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight,
                          jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight,
                          jblas::gemm::kblock::GemmCore_Row_NN_16x48_AMX_INT8_KBLOCK,
                          jblas::utils::parallel::Parallel2DGemmKBlockFixed, JblasAMX_INT8>(p, ctx);
    }
    if (check_vnni()) {
      return parse_weight<TASK, jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight,
                          jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight,
                          jblas::gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK,
                          jblas::utils::parallel::Parallel2DGemmKBlockFixed, JblasAVX512_VNNI>(p, ctx);
    }
    TORCH_CHECK(false, "device ISA muster lagger than VNNI when compute_type==int8");
  }
  if (p->compute_type == "fp32") {
  }
  if (p->compute_type == "bf16") {
  }
  TORCH_CHECK(false, "unsupported compute_type, compute_type==" + p->compute_type);
}

template <QBITS_TASK TASK>
void task_dispatcher(jblas_config_param* p, qbits_runtime_ctx* ctx) {
  return parse_gemm_core<TASK>(p, ctx);
}