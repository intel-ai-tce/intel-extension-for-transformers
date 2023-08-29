#include <ATen/core/TensorBody.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <torch/types.h>
#include <map>
#include "dispatcher/jblas_task_dispatcher.hpp"

static std::map<torch::ScalarType, QBITS_DT> qbits_dt_map{{torch::kFloat32, QBITS_FP32}};
static QBITS_DT get_qbits_dt(torch::Tensor* tensor) {
  TORCH_CHECK(qbits_dt_map.count(tensor->scalar_type()) != 0, "unsupported qbits data type.");
  return qbits_dt_map[tensor->scalar_type()];
}

template <QBITS_TASK TASK>
static void inline init_qbits_config_param(qbits_config_param* p, qbits_runtime_ctx* ctx,
                                           const std::string& compute_type, const std::string& weight_type) {
  p->compute_type = compute_type;
  p->weight_type = weight_type;
  switch (TASK) {
    case QBITS_QUANTIZE:
      p->src_dt = get_qbits_dt(ctx->weight);
      p->dst_dt = QBITS_FP32;  // jblas dosen't care about dst_dt in quantize-task, so we set fp32 as default.
      break;
    case QBITS_DEQUANTIZE:
      p->src_dt = QBITS_FP32;  // jblas dosen't care about src_dt in dequantize-task, so we set fp32 as default.
      p->dst_dt = get_qbits_dt(ctx->output);
      break;
    case QBITS_LINEAR:
      p->src_dt = get_qbits_dt(ctx->activation);
      p->dst_dt = get_qbits_dt(ctx->output);
      break;
  }
}

static torch::Tensor qbits_quantize(const torch::Tensor& fp32_weight, bool transpose, int64_t block_size,
                                    const std::string& compute_type, const std::string& weight_type) {
  torch::Tensor output;
  qbits_config_param p;
  qbits_runtime_ctx ctx{nullptr, const_cast<torch::Tensor*>(&fp32_weight), nullptr, &output, transpose, block_size};
  init_qbits_config_param<QBITS_QUANTIZE>(&p, &ctx, compute_type, weight_type);
  task_dispatcher(&p, &ctx, QBITS_QUANTIZE);
  return output;
}

static void qbits_dequantize(const torch::Tensor& compressed_weight, torch::Tensor& dequantize_weight, bool transpose,
                             const std::string& compute_type, const std::string& weight_type) {
  qbits_config_param p;
  qbits_runtime_ctx ctx{nullptr, const_cast<torch::Tensor*>(&compressed_weight), nullptr, &dequantize_weight,
                        transpose};
  init_qbits_config_param<QBITS_DEQUANTIZE>(&p, &ctx, compute_type, weight_type);
  task_dispatcher(&p, &ctx, QBITS_DEQUANTIZE);
}

static void qbits_linear_with_bias(const torch::Tensor& activation, const torch::Tensor& weight,
                                   const torch::Tensor& bias, torch::Tensor& output, int64_t lda, int64_t ldo,
                                   const std::string& compute_type, const std::string& weight_type) {
  qbits_config_param p;
  qbits_runtime_ctx ctx{
      const_cast<torch::Tensor*>(&activation),
      const_cast<torch::Tensor*>(&weight),
      const_cast<torch::Tensor*>(&bias),
      &output,
  };
  ctx.lda = lda;
  ctx.ldo = ldo;
  ctx.m = activation.sizes()[0];
  ctx.k = activation.sizes()[1];
  ctx.n = bias.sizes()[0];
  ctx.alpha = 1.f;
  ctx.beta = 1.f;
  init_qbits_config_param<QBITS_LINEAR>(&p, &ctx, compute_type, weight_type);
  task_dispatcher(&p, &ctx, QBITS_LINEAR);
}

static void qbits_linear_without_bias(const torch::Tensor& activation, const torch::Tensor& weight,
                                      torch::Tensor& output, int64_t n, int64_t lda, int64_t ldo,
                                      const std::string& compute_type, const std::string& weight_type) {
  qbits_config_param p;
  qbits_runtime_ctx ctx{
      const_cast<torch::Tensor*>(&activation),
      const_cast<torch::Tensor*>(&weight),
      &output,
      &output,
  };
  ctx.lda = lda;
  ctx.ldo = ldo;
  ctx.m = activation.sizes()[0];
  ctx.k = activation.sizes()[1];
  ctx.n = n;
  ctx.alpha = 1.f;
  ctx.beta = 0.f;
  init_qbits_config_param<QBITS_LINEAR>(&p, &ctx, compute_type, weight_type);
  task_dispatcher(&p, &ctx, QBITS_LINEAR);
}

static void qbits_set_weightonly_workspace(const torch::Tensor& workspace) {
  set_jblas_workspace(const_cast<torch::Tensor*>(&workspace));
}

TORCH_LIBRARY(weight_only_jblasop, m) {
  m.def("qbits_quantize", &qbits_quantize);
  m.def("qbits_linear_with_bias", &qbits_linear_with_bias);
  m.def("qbits_linear_without_bias", &qbits_linear_without_bias);
  m.def("qbits_dequantize", &qbits_dequantize);
  m.def("qbits_set_weightonly_workspace", &qbits_set_weightonly_workspace);
}