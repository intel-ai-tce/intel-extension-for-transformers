#include <torch/script.h>
#include <torch/torch.h>

static torch::Tensor jblas_quantize(const torch::Tensor& fp32_weight, bool transpose, int64_t block_size,
                                    const std::string& compute_type, const std::string& quant_type,
                                    const std::string& scale_type) {}

static void jblas_dequantize_weight(torch::Tensor& compressed_weight, torch::Tensor& dequantize_weight, bool transpose,
                                    const std::string& compute_type, const std::string& quant_type,
                                    const std::string& scale_type) {}

static void jblas_quantweight_f32_linear_with_bias(const torch::Tensor& activation, const torch::Tensor& weight,
                                                   const torch::Tensor& bias, torch::Tensor& output, int64_t m,
                                                   int64_t n, int64_t k, int64_t lda, int64_t ldo,
                                                   const std::string& compute_type, const std::string& quant_type,
                                                   const std::string& scale_type) {}

static void jblas_quantweight_f32_linear_without_bias(const torch::Tensor& activation, const torch::Tensor& weight,
                                                      torch::Tensor& output, int64_t m, int64_t n, int64_t k,
                                                      int64_t lda, int64_t ldo, const std::string& compute_type,
                                                      const std::string& quant_type, const std::string& scale_type) {}

TORCH_LIBRARY(weight_only_jblasop, m) {
  m.def("jblas_quantize", &jblas_quantize);
  m.def("jblas_quantweight_f32_linear_with_bias", &jblas_quantweight_f32_linear_with_bias);
  m.def("jblas_quantweight_f32_linear_without_bias", &jblas_quantweight_f32_linear_without_bias);
  m.def("jblas_dequantize_weight", &jblas_dequantize_weight);
}