
#include <cassert>
#include <cstddef>
#include <type_traits>
#include "jblas/jit_blas.h"
#include "jblas/jit_blas_epilogue.h"

template <JBLAS_ISA ISA_T, typename DST_T>
class DequantInt32AlphaBeta {
 public:
  struct Param {
    DST_T* C;
    int ldc;
    float* scalesA;
    int ldsa;
    float* scalesB;
    float* D;
    int ldd;
    float alpha, beta;
  };

  JBLAS_CODE forward(const int32_t* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& _param) {
    float* tmp_dst = reinterpret_cast<float*>(const_cast<int*>(cacheptr));
    jblas::kernel::wrapper::DequanS32Fp32::template forward<ISA_T>(cacheptr, cachestep, tmp_dst, cachestep, M, N,
                                                                   _param.scalesA + M_offset * _param.ldsa, _param.ldsa,
                                                                   _param.scalesB + N_offset);
    auto DOffset = M_offset * _param.ldd + N_offset;
    auto dptr = _param.D + DOffset;
    jblas::kernel::wrapper::AlphaBetaF32F32::template forward<ISA_T>(_param.alpha, tmp_dst, cachestep, _param.beta,
                                                                     dptr, _param.ldd, tmp_dst, cachestep, M, N);

    auto COffset = M_offset * _param.ldc + N_offset;
    auto cptr = _param.C + COffset;
    if constexpr (std::is_same_v<DST_T, float>) {
      return jblas::kernel::wrapper::Memcpy2D::template forward<ISA_T, float, DST_T>(
          (void*)tmp_dst, (void*)cptr, M, N * sizeof(DST_T), cachestep * sizeof(float), _param.ldc * sizeof(DST_T),
          NULL);
    }
    assert(false);
  }
};

template <JBLAS_ISA ISA_T>
using DequantInt32AlphaBetaStoreFp32 = DequantInt32AlphaBeta<ISA_T, float>;