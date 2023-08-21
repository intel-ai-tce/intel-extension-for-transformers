//  Copyright (c) 2023 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
#pragma once
#include <immintrin.h>

#include "jit_base.hpp"
#include "jit_blas.h"
#include "jit_blas_gemm.h"
#include "jit_blas_utils.h"
#include "kernel_wrapper.h"

namespace jblas {
namespace prologue {
enum class WeightPrologueType : int {
  Undef = -1,
  Begin = 0,
  WeightPack = Begin,
  End,
};
class PackedWeight {
 public:
  PackedWeight(jblas::gemm::GemmCoreType type) {
    mNPad = 0;
    mKPad = 0;
    mSize = 0;
    mCoreType = type;
  }

  virtual ~PackedWeight() {}

  void resize(int NPad, int KPad) {
    mNPad = NPad;
    mKPad = KPad;
  }

  virtual size_t getSerializedSize() {
    size_t totalsize = 0;
    totalsize += sizeof(mSize);
    totalsize += sizeof(mCoreType);
    totalsize += sizeof(mType);
    totalsize += sizeof(mNPad);
    totalsize += sizeof(mKPad);
    totalsize += getDataSerializedSize();
    return totalsize;
  }

  virtual void serializeToBuffer(void* buf) {
    auto wptr = reinterpret_cast<int8_t*>(buf);
    mSize = getSerializedSize();
    utils::serialize(wptr, mSize);
    utils::serialize(wptr, mCoreType);
    utils::serialize(wptr, mType);
    utils::serialize(wptr, mNPad);
    utils::serialize(wptr, mKPad);
    serializeDataToBuffer(wptr);
  }

  virtual void deserializeBuffer(void* buf, int memalloc) {
    auto rptr = reinterpret_cast<int8_t*>(buf);
    mSize = utils::deserialize<size_t>(rptr);
    mCoreType = utils::deserialize<jblas::gemm::GemmCoreType>(rptr);
    mType = utils::deserialize<int>(rptr);
    mNPad = utils::deserialize<int>(rptr);
    mKPad = utils::deserialize<int>(rptr);
    deserializeDataBuffer(rptr, memalloc);
  }
  size_t mSize = 0;
  jblas::gemm::GemmCoreType mCoreType = jblas::gemm::GemmCoreType::Undef;
  int mType = -1;
  int mNPad = 0, mKPad = 0;
  static int constexpr TypeOffset = sizeof(mSize) + sizeof(mCoreType);

 protected:
  virtual size_t getDataSerializedSize() = 0;
  virtual void serializeDataToBuffer(void* buf) = 0;
  virtual void deserializeDataBuffer(void* buf, int memalloc) = 0;
};

namespace gemm {

template <class _GemmCore_T, JBLAS_ISA ISA_T>
class ActivationBase {
 public:
  using AType = typename _GemmCore_T::AType;
  struct Param {
    const AType* A;
    int lda;
  };
  ActivationBase() {}

  JBLAS_CODE getActivation(AType** dstptr, int* dststep, const Param& _param, int m_size, int k_size, int m_offset,
                           int k_offset) {
    auto aptr = const_cast<AType*>(_param.A);
    if (k_size % _GemmCore_T::KTILE == 0) {
      *dstptr = aptr + m_offset * _param.lda + k_offset;
      *dststep = _param.lda;
      return JblasSuccess;
    } else {
      auto k_pad = utils::padto(k_size, _GemmCore_T::KTILE);
      *dststep = k_pad;
      return kernel::wrapper::Memcpy2D::forward<ISA_T, AType, AType>(aptr + m_offset * _param.lda + k_offset, *dstptr,
                                                                     m_size, k_size * sizeof(AType),
                                                                     _param.lda * sizeof(AType), k_pad * sizeof(AType));
    }
    return JblasSuccess;
  }
};

template <class _GemmCore_T, JBLAS_ISA ISA_T>
class ActivationConverterFp32 {
 public:
  using SrcType = float;
  using AType = typename _GemmCore_T::AType;
  struct Param {
    const SrcType* A;
    int lda;
  };
  ActivationConverterFp32() {}

  JBLAS_CODE getActivation(AType** dstptr, int* dststep, const Param& _param, int m_size, int k_size, int m_offset,
                           int k_offset) {
    auto aptr = const_cast<SrcType*>(_param.A);
    auto k_pad = utils::padto(k_size, _GemmCore_T::KTILE);
    *dststep = k_pad;
    if (std::is_same<AType, utils::bf16>::value) {
      return kernel::wrapper::Memcpy2DFp32CvtBf16::forward<ISA_T>(aptr + m_offset * _param.lda + k_offset, *dstptr,
                                                                  m_size, k_size, _param.lda * sizeof(SrcType),
                                                                  k_pad * sizeof(AType));
    }
    return JblasNotSupport;
  }
};

template <class _GemmCore_T, JBLAS_ISA ISA_T>
class ActivationF32U8KBlockQuantize {
 public:
  using AType = typename _GemmCore_T::AType;
  using AQType = uint8_t;
  using SType = float;
  struct Param {
    const float* A;
    int lda;
  };
  struct QuanParam {
    AQType* A = 0;
    AQType* zp = 0;
    int lda = 0;
    SType* scales = 0;
    int lds = 0;
    int kblock = 0;

    void resize(int m, int kpad, int _kblock) {
      kblock = _kblock;
      lda = kpad;
      mA.resize(m * lda);
      A = mA.data();
      lds = utils::updiv(kpad, _kblock);
      mScales.resize(m * lds);
      mZp.resize(m * lds);
      scales = mScales.data();
      zp = mZp.data();
    }
    utils::aligned_vector<AQType> mA;
    utils::aligned_vector<AQType> mZp;
    utils::aligned_vector<SType> mScales;
  };
  using Parallel = utils::parallel::Parallel2DRowMajorColBlock;

  Parallel createParallel(int m, int k, int kblock) {
    Parallel _paral;
    auto cb = utils::CpuBase();
    _paral.update(m, k, 1, 16, kblock, cb.mNumThreads);
    return _paral;
  }

  QuanParam createObj(int m, int k, int kblock) {
    QuanParam quan;
    int kpad = utils::padto(k, _GemmCore_T::KTILE);
    quan.resize(m, kpad, kblock);
    return quan;
  }

  void quantizeT(const Param& _param, int tidx, QuanParam& quan, Parallel& para) {
    int colidx, rowidx, rowsize, colsize;
    int blkidx, idxinblk;
    para.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize, &blkidx, &idxinblk);

    if (rowsize > 0 && colsize > 0) {
      // min max
      auto srcptr = _param.A + rowidx * _param.lda + colidx;
      int rowremain = utils::remainsize(rowidx, para.mRows, rowsize);
      int colremain = utils::remainsize(colidx, para.mCols, colsize);
      auto thdqptr = quan.A + rowidx * quan.lda + colidx;
      auto thdsptr = quan.scales + rowidx * quan.lds + blkidx;
      auto thdzptr = quan.zp + rowidx * quan.lds + blkidx;
      kernel::wrapper::QuantizeU8ColBlock::template forward<ISA_T>(
          rowremain, colremain, srcptr, _param.lda, thdqptr, quan.lda, thdsptr, quan.lds, thdzptr, para.mColBlock);
    }
  }

  QuanParam quantize(const Param& _param, int m, int k, int kblock) {
    utils::parallel::Parallel2DRowMajorColBlock paral = createParallel(m, k, kblock);
    QuanParam quan = createObj(m, k, kblock);
    if (paral.mThdsPerBlock == 1) {  // no barrier
#pragma omp parallel
      {
        int tidx = omp_get_thread_num();
        quantizeT(_param, tidx, quan, paral);
      }
    } else {
      assert(0);
    }
    return quan;
  }

  JBLAS_CODE getActivation(AType** dstptr, int* dststep, const QuanParam& _param, int m_size, int k_size, int m_offset,
                           int k_offset) {
    auto aptr = const_cast<AType*>(_param.A);
    *dstptr = aptr + m_offset * _param.lda + k_offset;
    *dststep = _param.lda;
    return JblasSuccess;
  }

  JBLAS_CODE getScale(SType** dstptr, int* dststep, const QuanParam& _param, int m_size, int k_size, int m_offset,
                      int k_offset) {
    auto ptr = const_cast<SType*>(_param.scales);
    *dstptr = ptr + m_offset * _param.lds + k_offset / _param.kblock;
    *dststep = _param.lds;
    return JblasSuccess;
  }

  static inline JBLAS_CODE getZp(AType** dstptr, int* dststep, const QuanParam& _param, int m_size, int k_size,
                                 int m_offset, int k_offset) {
    *dstptr = &_param.zp[(m_offset)*_param.lds + k_offset / _param.kblock];
    *dststep = _param.lds;
    return JblasSuccess;
  }

  static inline JBLAS_CODE getZpBroadcast(AType** dstptr, int* dststep, const QuanParam& _param, int m_size, int k_size,
                                          int m_offset, int k_offset) {
    for (size_t i = 0; i < m_size; i++) {
      auto zpval = _param.zp[(m_offset + i) * _param.lds + k_offset / _param.kblock];
      kernel::wrapper::Broadcast::template forward<ISA_T>(_param.kblock, zpval, *dstptr + i * _param.kblock);
    }
    return JblasSuccess;
  }
};

template <class _GemmCore_T, JBLAS_ISA ISA_T>
class ActivationF32S8KBlockQuantize {
 public:
  using AType = typename _GemmCore_T::AType;
  using AQType = int8_t;
  using SType = float;
  struct Param {
    const float* A;
    int lda;
  };
  struct QuanParam {
    AQType* A = 0;
    int lda = 0;
    SType* scales = 0;
    int lds = 0;
    int kblock = 0;

    void resize(int m, int kpad, int _kblock) {
      kblock = _kblock;
      lda = kpad;
      mA.resize(m * lda);
      A = mA.data();
      lds = utils::updiv(kpad, _kblock);
      mScales.resize(m * lds);
      scales = mScales.data();
    }
    utils::aligned_vector<AQType> mA;
    utils::aligned_vector<SType> mScales;
  };
  using Parallel = utils::parallel::Parallel2DRowMajorColBlock;

  Parallel createParallel(int m, int k, int kblock) {
    Parallel _paral;
    auto cb = utils::CpuBase();
    _paral.update(m, k, 1, 16, kblock, cb.mNumThreads);
    return _paral;
  }

  QuanParam createObj(int m, int k, int kblock) {
    QuanParam quan;
    int kpad = utils::padto(k, _GemmCore_T::KTILE);
    quan.resize(m, kpad, kblock);
    return quan;
  }

  void quantizeT(const Param& _param, int tidx, QuanParam& quan, Parallel& para) {
    int colidx, rowidx, rowsize, colsize;
    int blkidx, idxinblk;
    para.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize, &blkidx, &idxinblk);

    if (rowsize > 0 && colsize > 0) {
      // min max
      auto srcptr = _param.A + rowidx * _param.lda + colidx;
      int rowremain = utils::remainsize(rowidx, para.mRows, rowsize);
      int colremain = utils::remainsize(colidx, para.mCols, colsize);
      auto thdqptr = quan.A + rowidx * quan.lda + colidx;
      auto thdsptr = quan.scales + rowidx * quan.lds + blkidx;
      kernel::wrapper::QuantizeS8ColBlock::template forward<ISA_T>(rowremain, colremain, srcptr, _param.lda, thdqptr,
                                                                   quan.lda, thdsptr, quan.lds, para.mColBlock);
    }
  }

  QuanParam quantize(const Param& _param, int m, int k, int kblock) {
    utils::parallel::Parallel2DRowMajorColBlock paral = createParallel(m, k, kblock);
    QuanParam quan = createObj(m, k, kblock);
    if (paral.mThdsPerBlock == 1) {  // no barrier
#pragma omp parallel
      {
        int tidx = omp_get_thread_num();
        quantizeT(_param, tidx, quan, paral);
      }
    } else {
      assert(0);
    }
    return quan;
  }

  JBLAS_CODE getActivation(AType** dstptr, int* dststep, const QuanParam& _param, int m_size, int k_size, int m_offset,
                           int k_offset) {
    auto aptr = const_cast<AType*>(_param.A);
    *dstptr = aptr + m_offset * _param.lda + k_offset;
    *dststep = _param.lda;
    return JblasSuccess;
  }

  JBLAS_CODE getScale(SType** dstptr, int* dststep, const QuanParam& _param, int m_size, int k_size, int m_offset,
                      int k_offset) {
    auto ptr = const_cast<SType*>(_param.scales);
    *dstptr = ptr + m_offset * _param.lds + k_offset / _param.kblock;
    *dststep = _param.lds;
    return JblasSuccess;
  }

  static inline JBLAS_CODE getZp(AType** dstptr, int* dststep, const QuanParam& _param, int m_size, int k_size,
                                 int m_offset, int k_offset) {
    return JblasSuccess;
  }
};

template <typename T, JBLAS_ISA ISA_T>
class WeightBase {
 public:
  static void transposeWeight(const int Row, const int Col, const T* src, const int ld_src, T* dst, const int ld_dst) {
    utils::parallel::Parallel2DRowMajor _para;
    utils::CpuBase cb;
    _para.update(Row, Col, 16, 16, cb.mNumThreads);
    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      int colidx, rowidx, rowsize, colsize;
      _para.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
      if (rowsize > 0 && colsize > 0) {
        // rowremain: src valid size. rowsize: padded size
        int rowremain = utils::remainsize(rowidx, Row, rowsize);
        int colremain = utils::remainsize(colidx, Col, colsize);
        kernel::wrapper::Transpose2D<T>::template forward<ISA_T>(
            src + rowidx * ld_src + colidx, dst + rowidx + colidx * ld_dst, rowremain, colremain, ld_src, ld_dst);
      }
    }
  }
};

// Storage class has real weight memory, PackedWeight provides interface
class StorageWeight : public prologue::PackedWeight {
 public:
  StorageWeight(jblas::gemm::GemmCoreType _type) : PackedWeight(_type) {
    mRawPtr = NULL;
    mRawSize = 0;
    mType = int(WeightPrologueType::WeightPack);
  }

  void resize(int NPad, int KPad) {
    mNPad = NPad;
    mKPad = KPad;
    mWeights.resize((size_t)NPad * KPad * jblas::gemm::getWeightSize(mCoreType));
    mRawPtr = mWeights.data();
    mRawSize = mWeights.size();
  }

  template <typename WT>
  inline WT* getPtr() const {
    return reinterpret_cast<WT*>(mRawPtr);
  }

  template <typename WT>
  inline size_t getSize() const {
    return mRawSize / sizeof(WT);
  }

  int8_t* mRawPtr;
  size_t mRawSize;

 protected:
  virtual size_t getDataSerializedSize() override {
    size_t totalsize = 0;
    totalsize += sizeof(mRawSize);
    totalsize += mRawSize;
    return totalsize;
  }
  virtual void serializeDataToBuffer(void* buf) override {
    auto wptr = reinterpret_cast<int8_t*>(buf);
    utils::serialize(wptr, mRawSize);
    std::memcpy(mRawPtr, wptr, mRawSize);
  }
  virtual void deserializeDataBuffer(void* buf, int memalloc) override {
    auto rptr = reinterpret_cast<int8_t*>(buf);
    size_t rsize = utils::deserialize<size_t>(rptr);
    if (memalloc) {
      mWeights.resize(rsize);
      std::memcpy(mWeights.data(), rptr, rsize);
      mRawPtr = mWeights.data();
      mRawSize = mWeights.size();
    } else {
      mRawPtr = (int8_t*)rptr;
      mRawSize = rsize;
    }
  }
  utils::aligned_vector<int8_t> mWeights;
};

template <class _GemmCore_T, JBLAS_ISA ISA_T>
class WeightPack : public WeightBase<typename _GemmCore_T::BType, ISA_T> {
 public:
  using WType = typename _GemmCore_T::BType;
  struct Param {
    const prologue::PackedWeight* packedW;
  };

  PackedWeight* packTranspose(const int N, const int K, const WType* B, const int ldb) {
    utils::aligned_vector<float> B_NT(N * K);
    transposeWeight(N, K, B, ldb, B_NT.data(), N);
    return packWeight(N, K, B_NT.data(), N);
  }

  PackedWeight* pack(const int N, const int K, const WType* B, const int ldb) { return packWeight(N, K, B, N); }

  inline JBLAS_CODE getWeight(WType** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                              const PackedWeight* ptr) {
    auto wptr = dynamic_cast<const StorageWeight*>(ptr);
    if (wptr) {
      auto NPad = wptr->mNPad;
      auto KPad = wptr->mKPad;
      auto bptr = wptr->getPtr<WType>() + n_offset * KPad + k_offset * _GemmCore_T::NTILE;
      *dstptr = bptr;
      *dststep = KPad;
      return JblasSuccess;
    }
    return JblasInvalidParam;
  }

 protected:
  // from KxN int8 symmetric weight to packed N//NtilexKPadxNTile int4 weight
  PackedWeight* packWeight(const int N, const int K, const WType* B, const int ldb) {
    int KPad = utils::padto(K, _GemmCore_T::KTILE);
    int NPad = utils::padto(N, _GemmCore_T::NTILE);
    auto ptr = new StorageWeight(_GemmCore_T::TYPE);
    ptr->resize(NPad, KPad);
    WType* wptr = ptr->getPtr<WType>();
    reorder(N, K, B, ldb, wptr);
    return ptr;
  }

  void reorder(const int N, const int K, const WType* B, const int ldb, WType* dstptr) {
    utils::parallel::Parallel2DRowMajor _para;
    utils::CpuBase cb;
    _para.update(K, N, _GemmCore_T::KTILE, _GemmCore_T::NTILE, cb.mNumThreads);
    int KPad = utils::padto(K, _GemmCore_T::KTILE);
    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      int colidx, rowidx, rowsize, colsize;
      _para.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
      if (rowsize > 0 && colsize > 0) {
        int rowremain = utils::remainsize(rowidx, K,
                                          rowsize);  // rowremain: src valid size. rowsize: padded size
        int colremain = utils::remainsize(colidx, N, colsize);
        const auto src = B + rowidx * ldb + colidx;
        const auto dst = dstptr + rowidx * _GemmCore_T::NTILE + colidx * KPad;
        using PaddingInterleaveMNWType =
            kernel::wrapper::PaddingInterleaveMN<_GemmCore_T::NTILE, _GemmCore_T::PACK_ROW>;
        auto ret = PaddingInterleaveMNWType::template forward<ISA_T>(  //
            src, dst, rowremain, colremain, rowsize, colsize, ldb, KPad);
        assert(ret == JblasSuccess);
      }
    }
  }
};

}  // namespace gemm
class PackedWeightParser {
 public:
  static PackedWeight* deserialBuffer(void* serialized_buf, int memalloc = 0) {
    auto rptr = reinterpret_cast<int8_t*>(serialized_buf);
    rptr += PackedWeight::TypeOffset;
    int mType = utils::deserialize<int>(rptr);
    if (mType >= int(WeightPrologueType::Begin) && mType < int(WeightPrologueType::End)) {
      rptr = reinterpret_cast<int8_t*>(serialized_buf);
      auto type = static_cast<WeightPrologueType>(mType);
      switch (type) {
        case jblas::prologue::WeightPrologueType::WeightPack: {
          auto ptr = new gemm::StorageWeight(jblas::gemm::GemmCoreType::Undef);
          ptr->deserializeBuffer(rptr, memalloc);
          return ptr;
        }
        default:
          return nullptr;
      }
    }
    return nullptr;
  }
};

}  // namespace prologue
}  // namespace jblas
