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
#include "jit_blas_wrapper.h"
#include "kernel_wrapper.h"

namespace jblas {
namespace prologue {
namespace weight_comp {
class PackedWeightKBlock : public prologue::PackedWeight {
 public:
  PackedWeightKBlock(jblas::gemm::GemmCoreType _type) : PackedWeight(_type) {}
  int mBlockSize = 1;
};
namespace gemm {
enum class WeightCompType : int {
  Undef = 0,
  S8_F32,
  S4_F32,
  S4_Bf16,
};

class PackedWeightS4F32 : public prologue::weight_comp::PackedWeightKBlock {
 public:
  PackedWeightS4F32(jblas::gemm::GemmCoreType type) : PackedWeightKBlock(type) {
    mWPtr = NULL;
    mWSize = 0;
    mCoreType = type;
    mSPtr = NULL;
    mSSize = 0;
    mBlockSize = 0;
    mType = static_cast<int>(WeightCompType::S4_F32);
  }

  void resize(int NPad, int KPad, int Block) {
    mNPad = NPad;
    mKPad = KPad;
    mWeights.resize((size_t)NPad * KPad / 2);
    mBlockSize = Block;
    int nk_scale = utils::updiv(KPad, Block);
    mScales.resize(nk_scale * NPad);
    mWPtr = mWeights.data();
    mWSize = mWeights.size();
    mSPtr = mScales.data();
    mSSize = mScales.size();
  }

  utils::int4x2* mWPtr;
  size_t mWSize;
  float* mSPtr;
  size_t mSSize;

 protected:
  virtual size_t getDataSerializedSize() override {
    size_t totalsize = 0;
    totalsize += sizeof(mBlockSize);
    totalsize += sizeof(mWSize);
    totalsize += mWSize * sizeof(mWPtr[0]);
    totalsize += sizeof(mSSize);
    totalsize += mSSize * sizeof(mSPtr[0]);
    return totalsize;
  }
  virtual void serializeDataToBuffer(void* buf) override {
    auto wptr = reinterpret_cast<int8_t*>(buf);
    utils::serialize(wptr, mBlockSize);
    utils::serialize(wptr, mWSize);
    for (size_t i = 0; i < mWSize; i++) {
      utils::serialize(wptr, mWPtr[i]);
    }
    utils::serialize(wptr, mSSize);
    for (size_t i = 0; i < mSSize; i++) {
      utils::serialize(wptr, mSPtr[i]);
    }
  }
  virtual void deserializeDataBuffer(void* buf, int memalloc) override {
    auto rptr = reinterpret_cast<int8_t*>(buf);
    mBlockSize = utils::deserialize<int>(rptr);
    size_t rsize = utils::deserialize<size_t>(rptr);
    if (memalloc) {
      mWeights.resize(rsize);
      std::memcpy(mWeights.data(), rptr, rsize * sizeof(mWeights[0]));
      mWPtr = mWeights.data();
      mWSize = mWeights.size();
    } else {
      mWPtr = (utils::int4x2*)rptr;
      mWSize = rsize;
    }
    rptr += rsize * sizeof(mWeights[0]);
    rsize = utils::deserialize<size_t>(rptr);
    if (memalloc) {
      mScales.resize(rsize);
      std::memcpy(mScales.data(), rptr, rsize * sizeof(mScales[0]));
      mSPtr = mScales.data();
      mSSize = mScales.size();
    } else {
      mSPtr = (float*)rptr;
      mSSize = rsize;
    }
    rptr += rsize * sizeof(mScales[0]);
  }
  utils::aligned_vector<utils::int4x2> mWeights;
  utils::aligned_vector<float> mScales;
};

class PackedWeightS4Bf16 : public prologue::weight_comp::PackedWeightKBlock {
 public:
  PackedWeightS4Bf16(jblas::gemm::GemmCoreType _type) : PackedWeightKBlock(_type) {
    mWPtr = NULL;
    mWSize = 0;
    mSPtr = NULL;
    mSSize = 0;
    mBlockSize = 0;
    mType = static_cast<int>(WeightCompType::S4_Bf16);
  }

  void resize(int NPad, int KPad, int Block) {
    mNPad = NPad;
    mKPad = KPad;
    mWeights.resize((size_t)NPad * KPad / 2);
    mBlockSize = Block;
    int nk_scale = utils::updiv(KPad, Block);
    mScales.resize(nk_scale * NPad);
    mWPtr = mWeights.data();
    mWSize = mWeights.size();
    mSPtr = mScales.data();
    mSSize = mScales.size();
  }

  utils::int4x2* mWPtr;
  size_t mWSize;
  utils::bf16* mSPtr;
  size_t mSSize;

 protected:
  virtual size_t getDataSerializedSize() override {
    size_t totalsize = 0;
    totalsize += sizeof(mBlockSize);
    totalsize += sizeof(mWSize);
    totalsize += mWSize * sizeof(mWPtr[0]);
    totalsize += sizeof(mSSize);
    totalsize += mSSize * sizeof(mSPtr[0]);
    return totalsize;
  }
  virtual void serializeDataToBuffer(void* buf) override {
    auto wptr = reinterpret_cast<int8_t*>(buf);
    utils::serialize(wptr, mBlockSize);
    utils::serialize(wptr, mWSize);
    for (size_t i = 0; i < mWSize; i++) {
      utils::serialize(wptr, mWPtr[i]);
    }
    utils::serialize(wptr, mSSize);
    for (size_t i = 0; i < mSSize; i++) {
      utils::serialize(wptr, mSPtr[i]);
    }
  }
  virtual void deserializeDataBuffer(void* buf, int memalloc) override {
    auto rptr = reinterpret_cast<int8_t*>(buf);
    mBlockSize = utils::deserialize<int>(rptr);
    size_t rsize = utils::deserialize<size_t>(rptr);
    if (memalloc) {
      mWeights.resize(rsize);
      std::memcpy(mWeights.data(), rptr, rsize * sizeof(mWeights[0]));
      mWPtr = mWeights.data();
      mWSize = mWeights.size();
    } else {
      mWPtr = (utils::int4x2*)rptr;
      mWSize = rsize;
    }
    rptr += rsize * sizeof(mWeights[0]);
    rsize = utils::deserialize<size_t>(rptr);
    if (memalloc) {
      mScales.resize(rsize);
      std::memcpy(mScales.data(), rptr, rsize * sizeof(mScales[0]));
      mSPtr = mScales.data();
      mSSize = mScales.size();
    } else {
      mSPtr = (utils::bf16*)rptr;
      mSSize = rsize;
    }
    rptr += rsize * sizeof(mScales[0]);
  }
  utils::aligned_vector<utils::int4x2> mWeights;
  utils::aligned_vector<utils::bf16> mScales;
};

class PackedWeightS8F32 : public prologue::weight_comp::PackedWeightKBlock {
 public:
  PackedWeightS8F32(jblas::gemm::GemmCoreType _type) : PackedWeightKBlock(_type) {
    mWPtr = NULL;
    mWSize = 0;
    mSPtr = NULL;
    mSSize = 0;
    mBlockSize = 0;
    mType = static_cast<int>(WeightCompType::S8_F32);
  }

  void resize(int NPad, int KPad, int Block) {
    mNPad = NPad;
    mKPad = KPad;
    mWeights.resize((size_t)NPad * KPad);
    mBlockSize = Block;
    int nk_scale = utils::updiv(KPad, Block);
    mScales.resize(nk_scale * NPad);
    mWPtr = mWeights.data();
    mWSize = mWeights.size();
    mSPtr = mScales.data();
    mSSize = mScales.size();
  }

  int8_t* mWPtr;
  size_t mWSize;
  float* mSPtr;
  size_t mSSize;

 protected:
  virtual size_t getDataSerializedSize() override {
    size_t totalsize = 0;
    totalsize += sizeof(mBlockSize);
    totalsize += sizeof(mWSize);
    totalsize += mWSize * sizeof(mWPtr[0]);
    totalsize += sizeof(mSSize);
    totalsize += mSSize * sizeof(mSPtr[0]);
    return totalsize;
  }
  virtual void serializeDataToBuffer(void* buf) override {
    auto wptr = reinterpret_cast<int8_t*>(buf);
    utils::serialize(wptr, mBlockSize);
    utils::serialize(wptr, mWSize);
    for (size_t i = 0; i < mWSize; i++) {
      utils::serialize(wptr, mWPtr[i]);
    }
    utils::serialize(wptr, mSSize);
    for (size_t i = 0; i < mSSize; i++) {
      utils::serialize(wptr, mSPtr[i]);
    }
  }
  virtual void deserializeDataBuffer(void* buf, int memalloc) override {
    auto rptr = reinterpret_cast<int8_t*>(buf);
    mBlockSize = utils::deserialize<int>(rptr);
    size_t rsize = utils::deserialize<size_t>(rptr);
    if (memalloc) {
      mWeights.resize(rsize);
      std::memcpy(mWeights.data(), rptr, rsize * sizeof(mWeights[0]));
      mWPtr = mWeights.data();
      mWSize = mWeights.size();
    } else {
      mWPtr = (int8_t*)rptr;
      mWSize = rsize;
    }
    rptr += rsize * sizeof(mWeights[0]);
    rsize = utils::deserialize<size_t>(rptr);
    if (memalloc) {
      mScales.resize(rsize);
      std::memcpy(mScales.data(), rptr, rsize * sizeof(mScales[0]));
      mSPtr = mScales.data();
      mSSize = mScales.size();
    } else {
      mSPtr = (float*)rptr;
      mSSize = rsize;
    }
    rptr += rsize * sizeof(mScales[0]);
  }
  utils::aligned_vector<int8_t> mWeights;
  utils::aligned_vector<float> mScales;
};

template <class _GemmCore_T, JBLAS_ISA ISA_T>
class WeightS8_KBlock {
 public:
  struct Param {
    const prologue::PackedWeight* packedW;
  };

  void quantizeWeight(const int N, const int K, const float* B, const int ldb, int blocksize, int8_t* qB,
                      float* scales) {
    utils::parallel::Parallel2DRowMajor _para;
    utils::CpuBase cb;
    _para.update(K, N, blocksize, 16, cb.mNumThreads);
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
        kernel::wrapper::QuantizeS8RowBlock::forward<ISA_T>(B + rowidx * ldb + colidx, qB + rowidx * N + colidx,
                                                            rowremain, colremain, ldb, N,
                                                            scales + rowidx / blocksize * N + colidx, blocksize);
      }
    }
  }

  void transposeWeight(const int N, const int K, const float* src, const int ld_src, float* dst, const int ld_dst) {
    utils::parallel::Parallel2DRowMajor _para;
    utils::CpuBase cb;
    _para.update(N, K, 16, 16, cb.mNumThreads);
    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      int colidx, rowidx, rowsize, colsize;
      _para.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
      if (rowsize > 0 && colsize > 0) {
        int rowremain = utils::remainsize(rowidx, N,
                                          rowsize);  // rowremain: src valid size. rowsize: padded size
        int colremain = utils::remainsize(colidx, K, colsize);
        kernel::wrapper::Transpose2D<float>::forward<ISA_T>(
            src + rowidx * ld_src + colidx, dst + rowidx + colidx * ld_dst, rowremain, colremain, ld_src, ld_dst);
      }
    }
  }

  // from KxN int8 symmetric weight to packed N//NtilexKPadxNTile int4 weight

  PackedWeight* compressWeight(const int N, const int K, const int8_t* B, const int ldb, const float* scales,
                               int blocksize, WeightCompType type) {
    int KPad = utils::padto(K, _GemmCore_T::KTILE);
    int NPad = utils::padto(N, _GemmCore_T::NTILE);
    int nk_scale = utils::updiv(KPad, blocksize);
    PackedWeight* ptr = NULL;
    int8_t* wptr = NULL;
    if (type == WeightCompType::S8_F32) {
      auto tmp = new PackedWeightS8F32(_GemmCore_T::TYPE);
      tmp->resize(NPad, KPad, blocksize);
      wptr = tmp->mWPtr;
      ptr = tmp;
#pragma omp parallel for
      for (int i = 0; i < nk_scale; i++) {
        std::memcpy(tmp->mSPtr + i * NPad, scales + i * N, N * sizeof(scales[0]));
      }
    }
    if (ptr == NULL) {
      return ptr;
    }
    reorderCompress(N, K, B, ldb, scales, wptr, blocksize);
    return ptr;
  }

  void reorderCompress(const int N, const int K, const int8_t* B, const int ldb, const float* scales, int8_t* dstptr,
                       int blocksize) {
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
        auto ret = kernel::wrapper::PaddingInterleaveMN<_GemmCore_T::NTILE, sizeof(B[0]), _GemmCore_T::PACK_ROW>::
            template forward<ISA_T>((void*)(B + rowidx * ldb + colidx),
                                    dstptr + rowidx * _GemmCore_T::NTILE + colidx * KPad, rowremain, colremain, rowsize,
                                    colsize, ldb * sizeof(B[0]), KPad * sizeof(dstptr[0]));
        assert(ret == JblasSuccess);
      }
    }
  }

  PackedWeight* compressWeightTranspose(const int N, const int K, const float* B, const int ldb, int blocksize,
                                        WeightCompType type) {
    utils::aligned_vector<float> B_NT(N * K);
    transposeWeight(N, K, B, ldb, B_NT.data(), N);
    return compressWeight(N, K, B_NT.data(), N, blocksize, type);
  }

  PackedWeight* compressWeight(const int N, const int K, const float* B, const int ldb, int blocksize,
                               WeightCompType type) {
    int nk_scale = utils::updiv(K, blocksize);
    utils::aligned_vector<int8_t> quanW(N * K);
    utils::aligned_vector<float> scales(nk_scale * N);
    quantizeWeight(N, K, B, ldb, blocksize, quanW.data(), scales.data());
    return compressWeight(N, K, quanW.data(), N, scales.data(), blocksize, type);
  }

  template <typename _T>
  inline JBLAS_CODE getWeight(_T** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                              const PackedWeight* ptr) {
    return JblasNotSupport;
  }

  inline JBLAS_CODE getWeight(float** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                              const PackedWeight* ptr) {
    static_assert(_GemmCore_T::PACK_ROW == 1);  // float PackRow==1
    {
      auto wptr = dynamic_cast<const PackedWeightS8F32*>(ptr);
      if (wptr) {
        auto NPad = wptr->mNPad;
        auto KPad = wptr->mKPad;
        auto bptr = wptr->mWPtr + n_offset * KPad + k_offset * _GemmCore_T::NTILE;
        for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
          kernel::wrapper::DecompressKBlockS8F32::forward<ISA_T, float>(
              bptr + i * KPad, *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW,
              _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW, _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW,
              _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW, wptr->mSPtr + n_offset + i, k_offset, wptr->mBlockSize, NPad);
        }
        *dststep = k_size;
        return JblasSuccess;
      }
    }
    return JblasInvalidParam;
  }
};

class CompressedPackedWeight {
 public:
  static PackedWeight* deserialBuffer(void* serialized_buf, int memalloc = 0) {
    auto rptr = reinterpret_cast<int8_t*>(serialized_buf);
    rptr += PackedWeight::TypeOffset;
    int mType = utils::deserialize<int>(rptr);
    rptr = reinterpret_cast<int8_t*>(serialized_buf);
    auto type = static_cast<WeightCompType>(mType);
    if (type == WeightCompType::S4_F32) {
      auto ptr = new PackedWeightS4F32(jblas::gemm::GemmCoreType::Undef);
      ptr->deserializeBuffer(rptr, memalloc);
      return ptr;
    }
    if (type == WeightCompType::S4_Bf16) {
      auto ptr = new PackedWeightS4Bf16(jblas::gemm::GemmCoreType::Undef);
      ptr->deserializeBuffer(rptr, memalloc);
      return ptr;
    }
    if (type == WeightCompType::S8_F32) {
      auto ptr = new PackedWeightS8F32(jblas::gemm::GemmCoreType::Undef);
      ptr->deserializeBuffer(rptr, memalloc);
      return ptr;
    }
    return NULL;
  }
};

template <class _GemmCore_T, JBLAS_ISA ISA_T>
class WeightFp4_BNB_KBlock;

template <class _GemmCore_T, JBLAS_ISA ISA_T>
class WeightNf4_KBlock;
template <class _GemmCore_T, JBLAS_ISA ISA_T>
class WeightS4_Clip_KBlock;
template <class _GemmCore_T, JBLAS_ISA ISA_T>
class WeightBit4_KBlock {
 public:
  struct Param {
    const prologue::PackedWeight* packedW;
  };

  virtual void quantRowBlock(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src, int ld_dst,
                             float* scales, int blocksize) = 0;
  void quantizeWeight(const int N, const int K, const float* B, const int ldb, int blocksize, int8_t* qB,
                      float* scales) {
    utils::parallel::Parallel2DRowMajor _para;
    utils::CpuBase cb;
    _para.update(K, N, blocksize, 16, cb.mNumThreads);
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
        quantRowBlock(B + rowidx * ldb + colidx, qB + rowidx * N + colidx, rowremain, colremain, ldb, N,
                      scales + rowidx / blocksize * N + colidx, blocksize);
      }
    }
  }

  virtual void fp32_qdq(int bits, float* srcptr, float* dstptr, int row, int col, int ld_src, int ld_dst,
                        int blocksize) = 0;

  void qdqWeight(const std::string& quant_type, const int N, const int K, float* B, const int ldb, int blocksize,
                 bool transpose) {
    std::vector<float> trans_wei;
    if (transpose) {
      trans_wei.resize(N * K);
      for (int i = 0; i < N; i++)
        for (int j = 0; j < K; j++) trans_wei[j * N + i] = B[i * K + j];
    }
    float* tmp = transpose ? trans_wei.data() : B;
    int bits = quant_type == "s8" ? 8 : 4;
    fp32_qdq(bits, tmp, tmp, K, N, ldb, N, blocksize);
    if (transpose) {
      for (int i = 0; i < N; i++)
        for (int j = 0; j < K; j++) B[i * K + j] = trans_wei[j * N + i];
    }
  }

  void transposeWeight(const int N, const int K, const float* src, const int ld_src, float* dst, const int ld_dst) {
    utils::parallel::Parallel2DRowMajor _para;
    utils::CpuBase cb;
    _para.update(N, K, 16, 16, cb.mNumThreads);
    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      int colidx, rowidx, rowsize, colsize;
      _para.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
      if (rowsize > 0 && colsize > 0) {
        int rowremain = utils::remainsize(rowidx, N,
                                          rowsize);  // rowremain: src valid size. rowsize: padded size
        int colremain = utils::remainsize(colidx, K, colsize);
        kernel::wrapper::Transpose2D<float>::forward<ISA_T>(
            src + rowidx * ld_src + colidx, dst + rowidx + colidx * ld_dst, rowremain, colremain, ld_src, ld_dst);
      }
    }
  }

  // from KxN int8 symmetric weight to packed N//NtilexKPadxNTile int4 weight
  PackedWeight* compressWeight(const int N, const int K, const int8_t* B, const int ldb, const float* scales,
                               int blocksize, WeightCompType type) {
    int KPad = utils::padto(K, _GemmCore_T::KTILE);
    int NPad = utils::padto(N, _GemmCore_T::NTILE);
    int nk_scale = utils::updiv(KPad, blocksize);
    PackedWeight* ptr = NULL;
    utils::bit4x2* wptr = NULL;
    if (type == WeightCompType::S4_F32) {
      auto tmp = new PackedWeightS4F32(_GemmCore_T::TYPE);
      tmp->resize(NPad, KPad, blocksize);
      wptr = tmp->mWPtr;
      ptr = tmp;
#pragma omp parallel for
      for (int i = 0; i < nk_scale; i++) {
        std::memcpy(tmp->mSPtr + i * NPad, scales + i * N, N * sizeof(scales[0]));
      }
    } else if (type == WeightCompType::S4_Bf16) {
      auto tmp = new PackedWeightS4Bf16(_GemmCore_T::TYPE);
      tmp->resize(NPad, KPad, blocksize);
      wptr = tmp->mWPtr;
      ptr = tmp;
#pragma omp parallel for
      for (int i = 0; i < nk_scale; i++) {
        for (int j = 0; j < N; j++) {
          *(tmp->mSPtr + i * NPad + j) = utils::cast<float, utils::bf16>(*(scales + i * N + j));
        }
      }
    }
    if (ptr == NULL) {
      return ptr;
    }
    reorderCompress(N, K, B, ldb, scales, wptr, blocksize);
    return ptr;
  }

  virtual JBLAS_CODE doCompress(int8_t* srcptr, jblas::utils::bit4x2* dstptr, int row, int col, int ld_src,
                                int ld_dst) = 0;

  void reorderCompress(const int N, const int K, const int8_t* B, const int ldb, const float* scales,
                       utils::bit4x2* dstptr, int blocksize) {
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
        utils::aligned_vector<int8_t> tmp;
        tmp.resize(colsize * rowsize);
        auto ret = kernel::wrapper::PaddingInterleaveMN<_GemmCore_T::NTILE, sizeof(B[0]), _GemmCore_T::PACK_ROW>::
            template forward<ISA_T>((void*)(B + rowidx * ldb + colidx), tmp.data(), rowremain, colremain, rowsize,
                                    colsize, ldb * sizeof(B[0]), rowsize * sizeof(tmp[0]));
        assert(ret == JblasSuccess);
        ret = doCompress(tmp.data(), dstptr + rowidx * _GemmCore_T::NTILE / 2 + colidx * KPad / 2, rowsize, colsize,
                         rowsize, KPad);
        assert(ret == JblasSuccess);
      }
    }
  }

  PackedWeight* compressWeightTranspose(const int N, const int K, const float* B, const int ldb, int blocksize,
                                        WeightCompType type) {
    utils::aligned_vector<float> B_NT(N * K);
    transposeWeight(N, K, B, ldb, B_NT.data(), N);
    return compressWeight(N, K, B_NT.data(), N, blocksize, type);
  }

  PackedWeight* compressWeight(const int N, const int K, const float* B, const int ldb, int blocksize,
                               WeightCompType type) {
    int nk_scale = utils::updiv(K, blocksize);
    utils::aligned_vector<int8_t> quanW(N * K);
    utils::aligned_vector<float> scales(nk_scale * N);
    quantizeWeight(N, K, B, ldb, blocksize, quanW.data(), scales.data());
    return compressWeight(N, K, quanW.data(), N, scales.data(), blocksize, type);
  }

  virtual void DecompressKblockF32DstF32Scale(utils::bit4x2* srcptr, float* dstptr, int row, int col, int ld_src,
                                              int ld_dst, float* scales, int k_offset, int kblock, int NPad) = 0;

  virtual void DecompressKblockF32DstBf16Scale(utils::bit4x2* srcptr, float* dstptr, int row, int col, int ld_src,
                                               int ld_dst, utils::bf16* scales, int k_offset, int kblock, int NPad) = 0;

  virtual void DecompressKblockBf16DstF32Scale(utils::bit4x2* srcptr, utils::bf16* dstptr, int row, int col, int ld_src,
                                               int ld_dst, float* scales, int k_offset, int kblock, int NPad) = 0;

  virtual void DecompressKblockBf16DstBf16Scale(utils::bit4x2* srcptr, utils::bf16* dstptr, int row, int col,
                                                int ld_src, int ld_dst, utils::bf16* scales, int k_offset, int kblock,
                                                int NPad) = 0;

  inline JBLAS_CODE getWeight(float** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                              const PackedWeight* ptr) {
    static_assert(_GemmCore_T::PACK_ROW == 1);  // float PackRow==1
    {
      auto wptr = dynamic_cast<const PackedWeightS4F32*>(ptr);
      if (wptr) {
        auto NPad = wptr->mNPad;
        auto KPad = wptr->mKPad;
        auto bptr = wptr->mWPtr + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2;
        for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
          DecompressKblockF32DstF32Scale(
              bptr + i * KPad / 2, *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW,
              _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW, _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW / 2,
              _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW, wptr->mSPtr + n_offset + i, k_offset, wptr->mBlockSize, NPad);
        }
        *dststep = k_size;
        return JblasSuccess;
      }
    }
    {
      auto wptr = dynamic_cast<const PackedWeightS4Bf16*>(ptr);
      if (wptr) {
        auto NPad = wptr->mNPad;
        auto KPad = wptr->mKPad;
        auto bptr = wptr->mWPtr + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2;
        for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
          DecompressKblockF32DstBf16Scale(
              bptr + i * KPad / 2, *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW,
              _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW, _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW / 2,
              _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW, wptr->mSPtr + n_offset + i, k_offset, wptr->mBlockSize, NPad);
        }
        *dststep = k_size;
        return JblasSuccess;
      }
    }
    return JblasInvalidParam;
  }

  inline JBLAS_CODE getWeight(utils::bf16** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                              const PackedWeight* ptr) {
    static_assert(_GemmCore_T::PACK_ROW == 2);  // bf16 PackRow==1
    {
      auto wptr = dynamic_cast<const PackedWeightS4F32*>(ptr);
      if (wptr) {
        auto NPad = wptr->mNPad;
        auto KPad = wptr->mKPad;
        auto bptr = wptr->mWPtr + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2;
        for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
          DecompressKblockBf16DstF32Scale(
              bptr + i * KPad / 2, *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW,
              _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW, _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW / 2,
              _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW, wptr->mSPtr + n_offset + i, k_offset / _GemmCore_T::PACK_ROW,
              wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad);
        }
        *dststep = k_size;
        return JblasSuccess;
      }
    }
    {
      auto wptr = dynamic_cast<const PackedWeightS4Bf16*>(ptr);
      if (wptr) {
        auto NPad = wptr->mNPad;
        auto KPad = wptr->mKPad;
        auto bptr = wptr->mWPtr + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2;
        for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
          DecompressKblockBf16DstBf16Scale(
              bptr + i * KPad / 2, *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW,
              _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW, _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW / 2,
              _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW, wptr->mSPtr + n_offset + i, k_offset, wptr->mBlockSize, NPad);
        }
        *dststep = k_size;
        return JblasSuccess;
      }
    }
    return JblasInvalidParam;
  }

  inline JBLAS_CODE getWeight(int8_t** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                              const PackedWeight* ptr) {
    utils::int4x2* bptr = NULL;
    int NPad = 0, KPad = 0;
    {
      auto wptr = dynamic_cast<const PackedWeightS4F32*>(ptr);
      if (wptr) {
        NPad = wptr->mNPad;
        KPad = wptr->mKPad;
        bptr = wptr->mWPtr + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2;
      }
    }
    {
      auto wptr = dynamic_cast<const PackedWeightS4Bf16*>(ptr);
      if (wptr) {
        NPad = wptr->mNPad;
        KPad = wptr->mKPad;
        bptr = wptr->mWPtr + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2;
      }
    }
    if (bptr == NULL) {
      return JblasInvalidParam;
    }
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      kernel::wrapper::DecompressKBlockS4S8::forward<ISA_T>(
          bptr + i * KPad / 2, *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW,
          _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW, _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW,
          _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW);
    }
    *dststep = k_size;
    return JblasSuccess;
  }

  template <typename _T>
  JBLAS_CODE getScale(_T** dstptr, int* dststep, int n_size, int k_size, int n_offset, int k_offset,
                      const PackedWeight* ptr) {
    return JblasNotSupport;
  }

  JBLAS_CODE getScale(float** dstptr, int* dststep, int n_size, int k_size, int n_offset, int k_offset,
                      const PackedWeight* ptr) {
    auto wptr = dynamic_cast<const PackedWeightS4F32*>(ptr);
    if (wptr) {
      auto NPad = wptr->mNPad;
      auto KPad = wptr->mKPad;
      *dstptr = wptr->mSPtr + n_offset + k_offset / wptr->mBlockSize * NPad;
      *dststep = NPad;
      return JblasSuccess;
    }
    return JblasInvalidParam;
  }

  JBLAS_CODE getScale(utils::bf16** dstptr, int* dststep, int n_size, int k_size, int n_offset, int k_offset,
                      const PackedWeight* ptr) {
    auto wptr = dynamic_cast<const PackedWeightS4Bf16*>(ptr);
    if (wptr) {
      auto NPad = wptr->mNPad;
      auto KPad = wptr->mKPad;
      *dstptr = wptr->mSPtr + n_offset + k_offset / wptr->mBlockSize * NPad;
      *dststep = NPad;
      return JblasSuccess;
    }
    return JblasInvalidParam;
  }
};

template <class _GemmCore_T, JBLAS_ISA ISA_T>
class WeightS4_Clip_KBlock : public WeightBit4_KBlock<_GemmCore_T, ISA_T> {
 public:
  JBLAS_CODE doCompress(int8_t* srcptr, jblas::utils::bit4x2* dstptr, int row, int col, int ld_src,
                        int ld_dst) override {
    return kernel::wrapper::CompressS8S4<_GemmCore_T::NTILE>::template forward<ISA_T>(
        srcptr, reinterpret_cast<utils::int4x2*>(dstptr), row, col, ld_src,
        ld_dst);  // ld_dst here not stride
  }

  virtual void fp32_qdq(int bits, float* srcptr, float* dstptr, int row, int col, int ld_src, int ld_dst,
                        int blocksize) override {
    auto process_row = [&](int internal_blksize, int process_row, int done_row, int col_offset) {
      for (size_t j = 0; j < process_row; j += internal_blksize) {
        float maxval = std::numeric_limits<float>::min();
        for (size_t ij = 0; ij < internal_blksize; ij++) {
          maxval = std::max(maxval, std::abs(srcptr[(done_row + j + ij) * ld_src + col_offset]));
        }
        float scale = maxval / 127;
        uint32_t* bf16scale = reinterpret_cast<uint32_t*>(&scale);
        if (bits == 4) *bf16scale = *bf16scale & 0xffff0000;
        float rscale = 1.f / scale;
        for (size_t ij = 0; ij < internal_blksize; ij++) {
          int8_t tmp = utils::cast<float, int8_t>(srcptr[(done_row + j + ij) * ld_src + col_offset] * rscale);
          if (bits == 4) tmp = jblas::utils::int4x2::convert(tmp) << 4;
          dstptr[(done_row + j + ij) * ld_dst + col_offset] = (float)tmp * scale;
        }
      }
    };
    for (int i = 0; i < col; i++) {
      if (row < blocksize) {
        process_row(row, row, 0, i);
      } else {
        process_row(blocksize, row / blocksize * blocksize, 0, i);
        if (row % blocksize > 0) {
          process_row(row % blocksize, row % blocksize, row / blocksize * blocksize, i);
        }
      }
    }
  }

  virtual void quantRowBlock(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src, int ld_dst,
                             float* scales, int blocksize) override {
    kernel::wrapper::QuantizeS8RowBlock::forward<ISA_T>(srcptr, dstptr, row, col, ld_src, ld_dst, scales, blocksize);
  }

  void DecompressKblockF32DstF32Scale(utils::bit4x2* srcptr, float* dstptr, int row, int col, int ld_src, int ld_dst,
                                      float* scales, int k_offset, int kblock, int NPad) override {
    kernel::wrapper::DecompressKBlockS4FP<float>::forward<ISA_T, float, S4_TYPE>(
        reinterpret_cast<utils::int4x2*>(srcptr), dstptr, row, col, ld_src, ld_dst, scales, k_offset, kblock, NPad);
  }

  void DecompressKblockF32DstBf16Scale(utils::bit4x2* srcptr, float* dstptr, int row, int col, int ld_src, int ld_dst,
                                       utils::bf16* scales, int k_offset, int kblock, int NPad) override {
    kernel::wrapper::DecompressKBlockS4FP<float>::forward<ISA_T, utils::bf16, S4_TYPE>(
        reinterpret_cast<utils::int4x2*>(srcptr), dstptr, row, col, ld_src, ld_dst, scales, k_offset, kblock, NPad);
  }

  void DecompressKblockBf16DstF32Scale(utils::bit4x2* srcptr, utils::bf16* dstptr, int row, int col, int ld_src,
                                       int ld_dst, float* scales, int k_offset, int kblock, int NPad) override {
    kernel::wrapper::DecompressKBlockS4FP<utils::bf16>::forward<ISA_T, float, S4_TYPE>(
        reinterpret_cast<utils::int4x2*>(srcptr), dstptr, row, col, ld_src, ld_dst, scales, k_offset, kblock, NPad);
  }

  void DecompressKblockBf16DstBf16Scale(utils::bit4x2* srcptr, utils::bf16* dstptr, int row, int col, int ld_src,
                                        int ld_dst, utils::bf16* scales, int k_offset, int kblock, int NPad) override {
    kernel::wrapper::DecompressKBlockS4FP<utils::bf16>::forward<ISA_T, utils::bf16, S4_TYPE>(
        reinterpret_cast<utils::int4x2*>(srcptr), dstptr, row, col, ld_src, ld_dst, scales, k_offset, kblock, NPad);
  }

 private:
  static const JBLAS_S4_TYPE S4_TYPE = S4_CLIP;
};

template <class _GemmCore_T, JBLAS_ISA ISA_T>
class WeightS4_FullRange_KBlock : public WeightS4_Clip_KBlock<_GemmCore_T, ISA_T> {
  static const JBLAS_S4_TYPE S4_TYPE = S4_FULLRANGE;
  void quantRowBlock(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src, int ld_dst, float* scales,
                     int blocksize) override {
    kernel::wrapper::QuantizeS4FullRangeRowBlock::forward<ISA_T>(srcptr, dstptr, row, col, ld_src, ld_dst, scales,
                                                                 blocksize);
  }

  void DecompressKblockF32DstF32Scale(utils::bit4x2* srcptr, float* dstptr, int row, int col, int ld_src, int ld_dst,
                                      float* scales, int k_offset, int kblock, int NPad) override {
    kernel::wrapper::DecompressKBlockS4FP<float>::forward<ISA_T, float, S4_TYPE>(
        reinterpret_cast<utils::int4x2*>(srcptr), dstptr, row, col, ld_src, ld_dst, scales, k_offset, kblock, NPad);
  }

  void DecompressKblockF32DstBf16Scale(utils::bit4x2* srcptr, float* dstptr, int row, int col, int ld_src, int ld_dst,
                                       utils::bf16* scales, int k_offset, int kblock, int NPad) override {
    kernel::wrapper::DecompressKBlockS4FP<float>::forward<ISA_T, utils::bf16, S4_TYPE>(
        reinterpret_cast<utils::int4x2*>(srcptr), dstptr, row, col, ld_src, ld_dst, scales, k_offset, kblock, NPad);
  }

  void DecompressKblockBf16DstF32Scale(utils::bit4x2* srcptr, utils::bf16* dstptr, int row, int col, int ld_src,
                                       int ld_dst, float* scales, int k_offset, int kblock, int NPad) override {
    kernel::wrapper::DecompressKBlockS4FP<utils::bf16>::forward<ISA_T, float, S4_TYPE>(
        reinterpret_cast<utils::int4x2*>(srcptr), dstptr, row, col, ld_src, ld_dst, scales, k_offset, kblock, NPad);
  }

  void DecompressKblockBf16DstBf16Scale(utils::bit4x2* srcptr, utils::bf16* dstptr, int row, int col, int ld_src,
                                        int ld_dst, utils::bf16* scales, int k_offset, int kblock, int NPad) override {
    kernel::wrapper::DecompressKBlockS4FP<utils::bf16>::forward<ISA_T, utils::bf16, S4_TYPE>(
        reinterpret_cast<utils::int4x2*>(srcptr), dstptr, row, col, ld_src, ld_dst, scales, k_offset, kblock, NPad);
  }

  void fp32_qdq(int bits, float* srcptr, float* dstptr, int row, int col, int ld_src, int ld_dst,
                int blocksize) override {
    auto process_row = [&](int internal_blksize, int process_row, int done_row, int col_offset) {
      for (size_t j = 0; j < process_row; j += internal_blksize) {
        float amax = 0.f, max = 0.f;
        for (size_t ij = 0; ij < internal_blksize; ij++) {
          auto v = srcptr[(done_row + j + ij) * ld_src + col_offset];
          if (amax < std::abs(v)) {
            amax = std::abs(v);
            max = v;
          }
        }
        float scale = max / -8.f;
        uint32_t* bf16scale = reinterpret_cast<uint32_t*>(&scale);
        if (bits == 4) *bf16scale = *bf16scale & 0xffff0000;
        float rscale = scale != 0.f ? 1.f / scale : 0.f;
        for (size_t ij = 0; ij < internal_blksize; ij++) {
          auto quant_v = srcptr[(done_row + j + ij) * ld_src + col_offset] * rscale;
          int8_t x = MIN(15, (int8_t)(quant_v + 8.5f));
          dstptr[(done_row + j + ij) * ld_dst + col_offset] = (x - 8) * scale;
        }
      }
    };
    for (int i = 0; i < col; i++) {
      if (row < blocksize) {
        process_row(row, row, 0, i);
      } else {
        process_row(blocksize, row / blocksize * blocksize, 0, i);
        if (row % blocksize > 0) {
          process_row(row % blocksize, row % blocksize, row / blocksize * blocksize, i);
        }
      }
    }
  }
};

#define F4_BASE_FUNC(F4_TYPE)                                                                                          \
  void quantRowBlock(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src, int ld_dst, float* scales,     \
                     int blocksize) override {                                                                         \
    kernel::wrapper::QuantizeF4RowBlock::forward<ISA_T, F4_TYPE>(srcptr, dstptr, row, col, ld_src, ld_dst, scales,     \
                                                                 blocksize);                                           \
  }                                                                                                                    \
  JBLAS_CODE doCompress(int8_t* srcptr, jblas::utils::bit4x2* dstptr, int row, int col, int ld_src, int ld_dst)        \
      override {                                                                                                       \
    return kernel::wrapper::CompressF4<_GemmCore_T::NTILE>::template forward<ISA_T>(                                   \
        srcptr, reinterpret_cast<utils::f4x2*>(dstptr), row, col, ld_src, ld_dst);                                     \
  }                                                                                                                    \
  void DecompressKblockF32DstF32Scale(utils::bit4x2* srcptr, float* dstptr, int row, int col, int ld_src, int ld_dst,  \
                                      float* scales, int k_offset, int kblock, int NPad) override {                    \
    kernel::wrapper::DecompressKBlockF4Fp<float>::forward<ISA_T, float, F4_TYPE>(                                      \
        reinterpret_cast<utils::f4x2*>(srcptr), dstptr, row, col, ld_src, ld_dst, scales, k_offset, kblock, NPad);     \
  }                                                                                                                    \
  void DecompressKblockF32DstBf16Scale(utils::bit4x2* srcptr, float* dstptr, int row, int col, int ld_src, int ld_dst, \
                                       utils::bf16* scales, int k_offset, int kblock, int NPad) override {             \
    kernel::wrapper::DecompressKBlockF4Fp<float>::forward<ISA_T, utils::bf16, F4_TYPE>(                                \
        reinterpret_cast<utils::f4x2*>(srcptr), dstptr, row, col, ld_src, ld_dst, scales, k_offset, kblock, NPad);     \
  }                                                                                                                    \
  void DecompressKblockBf16DstF32Scale(utils::bit4x2* srcptr, utils::bf16* dstptr, int row, int col, int ld_src,       \
                                       int ld_dst, float* scales, int k_offset, int kblock, int NPad) override {       \
    kernel::wrapper::DecompressKBlockF4Fp<utils::bf16>::forward<ISA_T, float, F4_TYPE>(                                \
        reinterpret_cast<utils::f4x2*>(srcptr), dstptr, row, col, ld_src, ld_dst, scales, k_offset, kblock, NPad);     \
  }                                                                                                                    \
  void DecompressKblockBf16DstBf16Scale(utils::bit4x2* srcptr, utils::bf16* dstptr, int row, int col, int ld_src,      \
                                        int ld_dst, utils::bf16* scales, int k_offset, int kblock, int NPad)           \
      override {                                                                                                       \
    kernel::wrapper::DecompressKBlockF4Fp<utils::bf16>::forward<ISA_T, utils::bf16, F4_TYPE>(                          \
        reinterpret_cast<utils::f4x2*>(srcptr), dstptr, row, col, ld_src, ld_dst, scales, k_offset, kblock, NPad);     \
  }                                                                                                                    \
  virtual void fp32_qdq(int bits, float* srcptr, float* dstptr, int row, int col, int ld_src, int ld_dst,              \
                        int blocksize) override {                                                                      \
    jblas::kernel::ref::f4_fp32_qdq<F4_TYPE>(bits, srcptr, dstptr, row, col, ld_src, ld_dst, blocksize);               \
  }

template <class _GemmCore_T, JBLAS_ISA ISA_T>
class WeightFp4_BNB_KBlock : public WeightBit4_KBlock<_GemmCore_T, ISA_T> {
 public:
  static constexpr JBLAS_F4_TYPE F4_TYPE = FP4_BNB;
  F4_BASE_FUNC(FP4_BNB);
};

template <class _GemmCore_T, JBLAS_ISA ISA_T>
class WeightNf4_KBlock : public WeightBit4_KBlock<_GemmCore_T, ISA_T> {
 public:
  static constexpr JBLAS_F4_TYPE F4_TYPE = NF4;
  F4_BASE_FUNC(NF4);
};

}  // namespace gemm
}  // namespace weight_comp
}  // namespace prologue
namespace wrapper {
namespace gemm_kblock {

template <JBLAS_ISA _RT_ISA_T, class _GemmCore_T, template <class _T, JBLAS_ISA> class _PrologueA_T,
          template <class _T, JBLAS_ISA> class _PrologueB_T, template <JBLAS_ISA> class _Epilogue_T>
class GemmLauncherKBlockPackWeight {
 public:
  static JBLAS_ISA constexpr RT_ISA = _RT_ISA_T;
  using GemmCore = _GemmCore_T;
  using PrologueA = _PrologueA_T<GemmCore, _RT_ISA_T>;
  using PrologueB = _PrologueB_T<GemmCore, _RT_ISA_T>;
  using Epilogue = _Epilogue_T<_RT_ISA_T>;
  using AType = typename GemmCore::AType;
  using AParam = typename PrologueA::Param;
  using QuanAParam = typename PrologueA::QuanParam;
  using BType = typename GemmCore::BType;
  using BParam = typename PrologueB::Param;
  using CType = typename GemmCore::CType;
  using EpiParam = typename Epilogue::Param;
  static_assert(GemmCore::ISA <= _RT_ISA_T, "RunTime ISA should cover GEMM's ISA");
  struct Param {
    const int M, N, K;
    const AParam paramA;
    const BParam paramB;
    const EpiParam paramC;
    void* workspace;
  };
  struct ParallelConfig {
    const int rowidx, colidx;
    const int rowsize, colsize;
    const int MStep, NStep, KStep;
    const size_t StackSize;
  };
  GemmCore mGemmCore;
  PrologueA mProA;
  PrologueB mProB;
  Epilogue mEpilogue;

  void launch(const ParallelConfig& _config, const Param& _param, const QuanAParam& _quan) {
    int rowremain = utils::remainsize(_config.rowidx, _param.M, _config.rowsize);
    int colremain = utils::remainsize(_config.colidx, _param.N, _config.colsize);
    auto StackTmp = alloca(_config.StackSize);
    auto tmpB = (BType*)(StackTmp);
    auto tmpA = (AType*)(tmpB + _config.NStep * _config.KStep);
    auto tmpC = (CType*)(tmpA + GemmCore::MTILE * _config.KStep);
    for (int itern = 0; itern < colremain; itern += _config.NStep) {
      int n_remain = utils::remainsize(itern, colremain, _config.NStep);
      for (int iterm = 0; iterm < rowremain; iterm += _config.MStep) {
        int m_remain = utils::remainsize(iterm, rowremain, _config.MStep);
        run_block(_config, _param, _quan, iterm, itern, m_remain, n_remain, tmpA, tmpB, tmpC);
      }
    }
  }

 protected:
  void run_block(const ParallelConfig& _config, const Param& _param, const QuanAParam& _quan, int blk_m, int blk_n,
                 int blk_msize, int blk_nsize, AType* tmpA, BType* tmpB, CType* tmpC) {
    int n_padded = utils::padto(blk_nsize, GemmCore::NTILE);
    auto c_tile_ptr = tmpC;
    auto c_block_ptr = (float*)(c_tile_ptr + GemmCore::NTILE * GemmCore::MTILE);
    for (int iterk = 0; iterk < _param.K; iterk += _config.KStep) {
      int k_remain = utils::remainsize(iterk, _param.K, _config.KStep);
      int k_padded = utils::padto(k_remain, GemmCore::KTILE);
      auto bptr_cache = tmpB;
      int bcache_step = 0;
      mProB.getWeight(&bptr_cache, &bcache_step, k_padded, n_padded, iterk, _config.colidx + blk_n,
                      _param.paramB.packedW);
      int bcache_stride = bcache_step * sizeof(BType);
      for (int i = 0; i < blk_msize; i += GemmCore::MTILE) {
        int m_remain = utils::remainsize(i, blk_msize, GemmCore::MTILE);
        auto cptr_cache = c_block_ptr + i * _config.NStep;
        int ccache_stride = _config.NStep * sizeof(CType);

        AType* aptr_cache = nullptr;
        int acache_step = 0;
        mProA.getActivation(&aptr_cache, &acache_step, _quan, m_remain, k_padded, (blk_m + i + _config.rowidx), iterk);
        float* ascale_ptr = nullptr;
        int ascale_step = 0;
        mProA.getScale(&ascale_ptr, &ascale_step, _quan, m_remain, k_padded, (blk_m + i + _config.rowidx), iterk);
        AType* azp_ptr = tmpA;
        int azp_step = _config.KStep;
        mProA.getZpBroadcast(&azp_ptr, &azp_step, _quan, m_remain, k_padded, (blk_m + i + _config.rowidx), iterk);
        for (int itern = 0; itern < n_padded; itern += GemmCore::NTILE) {
          mGemmCore.forward(aptr_cache, bptr_cache + itern * bcache_step, c_tile_ptr, m_remain, GemmCore::NTILE,
                            k_padded, acache_step * sizeof(AType), bcache_stride, GemmCore::NTILE * sizeof(CType), 0);
          float* wscale_ptr = nullptr;
          int wscale_step = 0;
          mProB.getScale(&wscale_ptr, &wscale_step, GemmCore::NTILE, k_padded, (blk_n + itern + _config.colidx), iterk,
                         _param.paramB.packedW);
          kernel::wrapper::AccumulateDequantizeS32F32::template forward<_RT_ISA_T>(
              c_tile_ptr, cptr_cache + itern, 1.f, iterk == 0 ? 0.f : 1.f, m_remain, GemmCore::NTILE, GemmCore::NTILE,
              _config.NStep, ascale_ptr, ascale_step, wscale_ptr);
          mGemmCore.forward(azp_ptr, bptr_cache + itern * bcache_step, c_tile_ptr, m_remain, GemmCore::NTILE, k_padded,
                            azp_step * sizeof(AType), bcache_stride, GemmCore::NTILE * sizeof(CType), 0);
          kernel::wrapper::AccumulateDequantizeS32F32::template forward<_RT_ISA_T>(
              c_tile_ptr, cptr_cache + itern, -1.f, 1.f, m_remain, GemmCore::NTILE, GemmCore::NTILE, _config.NStep,
              ascale_ptr, ascale_step, wscale_ptr);
        }
      }
    }
    mEpilogue.forward(c_block_ptr, _config.NStep, (_config.rowidx + blk_m), _config.colidx + blk_n, blk_msize,
                      blk_nsize, _param.paramC);
  }
};

template <JBLAS_ISA _RT_ISA_T, class _GemmCore_T, template <class _T, JBLAS_ISA> class _PrologueA_T,
          template <class _T, JBLAS_ISA> class _PrologueB_T, template <JBLAS_ISA> class _Epilogue_T>
class GemmSLauncherKBlockPackWeight {
 public:
  static JBLAS_ISA constexpr RT_ISA = _RT_ISA_T;
  using GemmCore = _GemmCore_T;
  using PrologueA = _PrologueA_T<GemmCore, _RT_ISA_T>;
  using PrologueB = _PrologueB_T<GemmCore, _RT_ISA_T>;
  using Epilogue = _Epilogue_T<_RT_ISA_T>;
  using AType = typename GemmCore::AType;
  using AParam = typename PrologueA::Param;
  using QuanAParam = typename PrologueA::QuanParam;
  using BType = typename GemmCore::BType;
  using BParam = typename PrologueB::Param;
  using CType = typename GemmCore::CType;
  using EpiParam = typename Epilogue::Param;
  static_assert(GemmCore::ISA <= _RT_ISA_T, "RunTime ISA should cover GEMM's ISA");
  struct Param {
    const int M, N, K;
    const AParam paramA;
    const BParam paramB;
    const EpiParam paramC;
    void* workspace;
  };
  struct ParallelConfig {
    const int rowidx, colidx;
    const int rowsize, colsize;
    const int MStep, NStep, KStep;
    const size_t StackSize;
  };
  GemmCore mGemmCore;
  PrologueA mProA;
  PrologueB mProB;
  Epilogue mEpilogue;

  void launch(const ParallelConfig& _config, const Param& _param, const QuanAParam& _quan) {
    auto blkptr = dynamic_cast<const prologue::weight_comp::PackedWeightKBlock*>(_param.paramB.packedW);
    if (blkptr == nullptr) {
      return;
    }
    int rowremain = utils::remainsize(_config.rowidx, _param.M, _config.rowsize);
    int colremain = utils::remainsize(_config.colidx, _param.N, _config.colsize);
    auto StackTmp = alloca(_config.StackSize);
    auto tmpB = (BType*)(StackTmp);
    auto tmpA = (AType*)(tmpB + _config.NStep * _config.KStep);
    auto tmpC = (CType*)(tmpA + GemmCore::MTILE * _config.KStep);
    for (int itern = 0; itern < colremain; itern += _config.NStep) {
      int n_remain = utils::remainsize(itern, colremain, _config.NStep);
      for (int iterm = 0; iterm < rowremain; iterm += _config.MStep) {
        int m_remain = utils::remainsize(iterm, rowremain, _config.MStep);
        run_block(_config, _param, blkptr, _quan, iterm, itern, m_remain, n_remain, tmpA, tmpB, tmpC);
      }
    }
  }

 protected:
  void run_block(const ParallelConfig& _config, const Param& _param,
                 const prologue::weight_comp::PackedWeightKBlock* blkptr, const QuanAParam& _quan, int blk_m, int blk_n,
                 int blk_msize, int blk_nsize, AType* tmpA, BType* tmpB, CType* tmpC) {
    int n_padded = utils::padto(blk_nsize, GemmCore::NTILE);
    auto c_tile_ptr = tmpC;
    auto c_block_ptr = (float*)(c_tile_ptr + GemmCore::NTILE * GemmCore::MTILE);
    for (int iterk = 0; iterk < _param.K; iterk += _config.KStep) {
      int k_remain = utils::remainsize(iterk, _param.K, _config.KStep);
      int k_padded = utils::padto(k_remain, GemmCore::KTILE);
      auto bptr_cache = tmpB;
      int bcache_step = 0;
      mProB.getWeight(&bptr_cache, &bcache_step, k_padded, n_padded, iterk, _config.colidx + blk_n,
                      _param.paramB.packedW);
      float* wscale_ptr = nullptr;
      utils::bf16* wscale_bf16ptr = nullptr;
      int wscale_step = 0;
      if (blkptr->mType == int(prologue::weight_comp::gemm::WeightCompType::S4_F32)) {
        mProB.getScale(&wscale_ptr, &wscale_step, n_padded, k_padded, (blk_n + _config.colidx), iterk,
                       _param.paramB.packedW);
      } else if (blkptr->mType == int(prologue::weight_comp::gemm::WeightCompType::S4_Bf16)) {
        mProB.getScale(&wscale_bf16ptr, &wscale_step, n_padded, k_padded, (blk_n + _config.colidx), iterk,
                       _param.paramB.packedW);
      }

      int bcache_stride = bcache_step * sizeof(BType);
      for (int i = 0; i < blk_msize; i += GemmCore::MTILE) {
        int m_remain = utils::remainsize(i, blk_msize, GemmCore::MTILE);
        auto cptr_cache = c_block_ptr + i * _config.NStep;
        int ccache_stride = _config.NStep * sizeof(CType);

        AType* aptr_cache = nullptr;
        int acache_step = 0;
        mProA.getActivation(&aptr_cache, &acache_step, _quan, m_remain, k_padded, (blk_m + i + _config.rowidx), iterk);
        float* ascale_ptr = nullptr;
        int ascale_step = 0;
        mProA.getScale(&ascale_ptr, &ascale_step, _quan, m_remain, k_padded, (blk_m + i + _config.rowidx), iterk);
        AType* azp_ptr = tmpA;
        int azp_step = _config.KStep;
        mProA.getZp(&azp_ptr, &azp_step, _quan, m_remain, k_padded, (blk_m + i + _config.rowidx), iterk);
        if (blkptr->mType == int(prologue::weight_comp::gemm::WeightCompType::S4_F32)) {
          mGemmCore.forward(aptr_cache, bptr_cache, cptr_cache, azp_ptr, ascale_ptr, ascale_step, wscale_ptr,
                            wscale_step, m_remain, n_padded, k_padded, blkptr->mBlockSize, acache_step * sizeof(AType),
                            bcache_stride, _config.NStep * sizeof(CType), iterk);
        } else if (blkptr->mType == int(prologue::weight_comp::gemm::WeightCompType::S4_Bf16)) {
          mGemmCore.forward(aptr_cache, bptr_cache, cptr_cache, azp_ptr, ascale_ptr, ascale_step, wscale_bf16ptr,
                            wscale_step, m_remain, n_padded, k_padded, blkptr->mBlockSize, acache_step * sizeof(AType),
                            bcache_stride, _config.NStep * sizeof(CType), iterk);
        }
      }
    }
    mEpilogue.forward(c_block_ptr, _config.NStep, (_config.rowidx + blk_m), _config.colidx + blk_n, blk_msize,
                      blk_nsize, _param.paramC);
  }
};

template <class _Launcher_T, template <class _T> class _Parallel_T>
class GemmInterfaceKBlockPackWeight {
 public:
  using Arguments = typename _Launcher_T::Param;
  using Config = typename _Launcher_T::ParallelConfig;
  using ActivationType = typename _Launcher_T::PrologueA;
  using WeightType = typename _Launcher_T::PrologueB;
  using GemmCore = typename _Launcher_T::GemmCore;
  using Parallel = _Parallel_T<GemmCore>;
  Parallel createParallel(int M = 0, int N = 0, int K = 0, int KBlock = 0) {
    Parallel _paral;
    auto cb = utils::CpuBase();
    _paral.update(M, N, K, KBlock, cb.mNumThreads);
    return _paral;
  }
  ActivationType* getActivationPtr() { return &mLauncher.mProA; }
  WeightType* getWeightPtr() { return &mLauncher.mProB; }
  // forward=packB+compute
  JBLAS_CODE compute(const Arguments& _param, Parallel _paral = Parallel()) {
    auto bptr = dynamic_cast<const prologue::weight_comp::PackedWeightKBlock*>(_param.paramB.packedW);
    if (bptr == nullptr) {
      return JblasInvalidParam;
    }
    auto paraA = mLauncher.mProA.createParallel(_param.M, _param.K, bptr->mBlockSize);
    auto quanA = mLauncher.mProA.createObj(_param.M, _param.K, bptr->mBlockSize);
    auto cb = utils::CpuBase();
    if (_paral.update(_param.M, _param.N, _param.K, bptr->mBlockSize, cb.mNumThreads)) {
      static bool dbgprint = false;
      if (dbgprint) {
        _paral.print();
        dbgprint = false;
      }
    }
    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      mLauncher.mProA.quantizeT(_param.paramA, tidx, quanA, paraA);
#pragma omp barrier
      int colidx, rowidx, rowsize, colsize;
      _paral.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
      if (rowsize > 0 && colsize > 0) {
        Config _config{rowidx,     colidx, rowsize, colsize, _paral.getMStep(), _paral.getNStep(), _paral.getKStep(),
                       cb.mL2Cache};
        mLauncher.launch(_config, _param, quanA);
      }
    }
    return JblasSuccess;
  }

 protected:
  _Launcher_T mLauncher;
};

}  // namespace gemm_kblock
namespace gemm_default {
namespace weight_comp {
namespace avx512f {
JBLAS_ISA constexpr DefaultISA = JblasAVX512F;
using GemmKernelS4ClipKBlock = jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
        DefaultISA, jblas::gemm::GemmCore_Row_NN_8x48_AVX512F, jblas::prologue::gemm::ActivationBase,
        jblas::prologue::weight_comp::gemm::WeightS4_Clip_KBlock, jblas::epilogue::gemm::AlphaBetaProcessFp32>,
    DefaultParallel>;
using GemmKernelS4FullRangeKBlock = jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
        DefaultISA, jblas::gemm::GemmCore_Row_NN_8x48_AVX512F, jblas::prologue::gemm::ActivationBase,
        jblas::prologue::weight_comp::gemm::WeightS4_FullRange_KBlock, jblas::epilogue::gemm::AlphaBetaProcessFp32>,
    DefaultParallel>;
using GemmKernelNf4KBlock = jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
        DefaultISA, jblas::gemm::GemmCore_Row_NN_8x48_AVX512F, jblas::prologue::gemm::ActivationBase,
        jblas::prologue::weight_comp::gemm::WeightNf4_KBlock, jblas::epilogue::gemm::AlphaBetaProcessFp32>,
    DefaultParallel>;
using GemmKernelS8KBlock = jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
        DefaultISA, jblas::gemm::GemmCore_Row_NN_8x48_AVX512F, jblas::prologue::gemm::ActivationBase,
        jblas::prologue::weight_comp::gemm::WeightS8_KBlock, jblas::epilogue::gemm::AlphaBetaProcessFp32>,
    DefaultParallel>;
}  // namespace avx512f
namespace avx512_vnni {
JBLAS_ISA constexpr DefaultISA = JblasAVX512_VNNI;
using GemmKernelDynamicQuantS4KBlockSimple = jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight<
    jblas::wrapper::gemm_kblock::GemmLauncherKBlockPackWeight<
        DefaultISA, jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI, jblas::prologue::gemm::ActivationF32U8KBlockQuantize,
        jblas::prologue::weight_comp::gemm::WeightS4_Clip_KBlock, jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
    jblas::utils::parallel::Parallel2DGemmKBlock>;
using GemmKernelDynamicQuantS4KBlock = jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight<
    jblas::wrapper::gemm_kblock::GemmLauncherKBlockPackWeight<
        DefaultISA, jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI, jblas::prologue::gemm::ActivationF32U8KBlockQuantize,
        jblas::prologue::weight_comp::gemm::WeightS4_Clip_KBlock, jblas::epilogue::gemm::AlphaBetaProcessFp32>,
    jblas::utils::parallel::Parallel2DGemmKBlock>;

using GemmSKernelDynamicS4ClipKBlock = jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight<
    jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
        DefaultISA, jblas::gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK,
        jblas::prologue::gemm::ActivationF32U8KBlockQuantize, jblas::prologue::weight_comp::gemm::WeightS4_Clip_KBlock,
        jblas::epilogue::gemm::AlphaBetaProcessFp32>,
    jblas::utils::parallel::Parallel2DGemmKBlockFixed>;
using GemmSKernelDynamicS4FullRangeKBlock = jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight<
    jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
        DefaultISA, jblas::gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK,
        jblas::prologue::gemm::ActivationF32U8KBlockQuantize,
        jblas::prologue::weight_comp::gemm::WeightS4_FullRange_KBlock, jblas::epilogue::gemm::AlphaBetaProcessFp32>,
    jblas::utils::parallel::Parallel2DGemmKBlockFixed>;
}  // namespace avx512_vnni
namespace amx_bf16 {
JBLAS_ISA constexpr DefaultISA = JblasAMX_BF16;
using GemmKernelS4ClipKBlock = jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
        DefaultISA, jblas::gemm::GemmCore_Row_NN_16x64_AMX_BF16,
        jblas::prologue::gemm::ActivationConverterFp32,  // activation fp32->bf16
        jblas::prologue::weight_comp::gemm::WeightS4_Clip_KBlock,
        jblas::epilogue::gemm::AlphaBetaProcessFp32>,  // output fp32->fp32
    DefaultParallel>;
using GemmKernelS4FullRangeKBlock = jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
        DefaultISA, jblas::gemm::GemmCore_Row_NN_16x64_AMX_BF16,
        jblas::prologue::gemm::ActivationConverterFp32,  // activation fp32->bf16
        jblas::prologue::weight_comp::gemm::WeightS4_FullRange_KBlock,
        jblas::epilogue::gemm::AlphaBetaProcessFp32>,  // output fp32->fp32
    DefaultParallel>;
using GemmKernelFp4KBlock = jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
        DefaultISA, jblas::gemm::GemmCore_Row_NN_16x64_AMX_BF16,
        jblas::prologue::gemm::ActivationConverterFp32,  // activation fp32->bf16
        jblas::prologue::weight_comp::gemm::WeightFp4_BNB_KBlock,
        jblas::epilogue::gemm::AlphaBetaProcessFp32>,  // output fp32->fp32
    DefaultParallel>;
using GemmKernelNf4KBlock = jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
        DefaultISA, jblas::gemm::GemmCore_Row_NN_16x64_AMX_BF16,
        jblas::prologue::gemm::ActivationConverterFp32,  // activation fp32->bf16
        jblas::prologue::weight_comp::gemm::WeightNf4_KBlock,
        jblas::epilogue::gemm::AlphaBetaProcessFp32>,  // output fp32->fp32
    DefaultParallel>;
}  // namespace amx_bf16
namespace amx_int8 {
JBLAS_ISA constexpr DefaultISA = JblasAMX_INT8;
using GemmSKernelDynamicS4ClipKBlock = jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight<
    jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
        DefaultISA, jblas::gemm::kblock::GemmCore_Row_NN_16x48_AMX_INT8_KBLOCK,
        jblas::prologue::gemm::ActivationF32S8KBlockQuantize, jblas::prologue::weight_comp::gemm::WeightS4_Clip_KBlock,
        jblas::epilogue::gemm::AlphaBetaProcessFp32>,
    jblas::utils::parallel::Parallel2DGemmKBlockFixed>;
using GemmSKernelDynamicS4FullRangeKBlock = jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight<
    jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
        DefaultISA, jblas::gemm::kblock::GemmCore_Row_NN_16x48_AMX_INT8_KBLOCK,
        jblas::prologue::gemm::ActivationF32S8KBlockQuantize,
        jblas::prologue::weight_comp::gemm::WeightS4_FullRange_KBlock, jblas::epilogue::gemm::AlphaBetaProcessFp32>,
    jblas::utils::parallel::Parallel2DGemmKBlockFixed>;
}  // namespace amx_int8
}  // namespace weight_comp
}  // namespace gemm_default
}  // namespace wrapper
}  // namespace jblas