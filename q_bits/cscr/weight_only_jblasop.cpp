#include <torch/script.h>
#include <torch/torch.h>
#include "jblas_task_dispatcher.hpp"

template <QBITS_DT SRC_DT, QBITS_DT DST_DT>
static void inline init_jblas_config_param(jblas_config_param* p, const std::string& compute_type,
                                           const std::string& weight_type) {
  p->compute_type = compute_type;
  p->weight_type = weight_type;
  p->src_dt = SRC_DT;
  p->dst_dt = DST_DT;
}

static torch::Tensor jblas_quantize(const torch::Tensor& fp32_weight, bool transpose, int64_t block_size,
                                    const std::string& compute_type, const std::string& weight_type) {
  torch::Tensor output;
  jblas_config_param p;
  init_jblas_config_param<QBITS_FP32, QBITS_FP32>(&p, compute_type, weight_type);
  qbits_runtime_ctx ctx{nullptr, const_cast<torch::Tensor*>(&fp32_weight), nullptr, &output, transpose, block_size};
  task_dispatcher<QBITS_QUANTIZE>(&p, &ctx);
  return output;
}

static void jblas_dequantize_weight(torch::Tensor& compressed_weight, torch::Tensor& dequantize_weight, bool transpose,
                                    const std::string& compute_type, const std::string& weight_type) {}

static void jblas_f32in_f32out_linear_with_bias(const torch::Tensor& activation, const torch::Tensor& weight,
                                                const torch::Tensor& bias, torch::Tensor& output, int64_t lda,
                                                int64_t ldo, const std::string& compute_type,
                                                const std::string& weight_type) {}

static void jblas_f32in_f32out_linear_without_bias(const torch::Tensor& activation, const torch::Tensor& weight,
                                                   torch::Tensor& output, int64_t n, int64_t lda, int64_t ldo,
                                                   const std::string& compute_type, const std::string& weight_type) {}

TORCH_LIBRARY(weight_only_jblasop, m) {
  m.def("jblas_quantize", &jblas_quantize);
  m.def("jblas_quantweight_f32inf32out_linear_with_bias", &jblas_f32in_f32out_linear_with_bias);
  m.def("jblas_quantweight_f32inf32out_linear_without_bias", &jblas_f32in_f32out_linear_without_bias);
  m.def("jblas_dequantize_weight", &jblas_dequantize_weight);
}