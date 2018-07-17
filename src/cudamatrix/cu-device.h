// cudamatrix/cu-device.h

// Copyright 2009-2012  Karel Vesely
//           2012-2015  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.



#ifndef KALDI_CUDAMATRIX_CU_DEVICE_H_
#define KALDI_CUDAMATRIX_CU_DEVICE_H_

#if HAVE_CUDA == 1

#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include <cublas_v2.h>
#include <map>
#include <string>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "base/kaldi-common.h"
#include "cudamatrix/cu-allocator.h"

#include <mpi.h>

namespace kaldi {

typedef struct GpuInfo_{
        int32 gpuid;
        float d2h_bandwidth;
        float h2d_bandwidth;
        size_t mem_free;
        size_t mem_total;
        float  mem_ratio;
        bool   used;
}GpuInfo;

typedef struct MPIGpuInfo_{
	char hostname[256];
	int gpuid;
	int myid;
}MPIGpuInfo;

/**
 * Singleton object which represents the CUDA device
 * responsible for CUBLAS initilalisation, collects profiling info
 */
class CuDevice {
 // Singleton object (there should only be one instantiated per program)
 public:
  ~CuDevice();
  static inline CuDevice& Instantiate() { return global_device_; }

  inline cublasHandle_t GetHandle() { return handle_; }

  // We provide functions Malloc, MallocPitch and Free which replace cudaMalloc,
  // cudaMallocPitch and cudaFree.  Their function is to cache the results of
  // previous allocations to avoid the very large overhead that CUDA's
  // allocation seems to give for some setups.
  inline void* Malloc(size_t size) { return allocator_.MallocLocal(size); }

  inline void* MallocPitch(size_t row_bytes, size_t num_rows, size_t *pitch) {
    return allocator_.MallocPitchLocal(row_bytes, num_rows, pitch);
  }
  inline void Free(void *ptr) { allocator_.FreeLocal(ptr); }

  /// Select a GPU for computation, the 'use_gpu' modes are:
  ///  "yes"      -- Select GPU automatically and die if this fails.
  ///  "optional" -- Do as above, but if it fails, back off to CPU.
  ///  "no"       -- Run on CPU.
  ///  (more comments in cu-device.cc)


  void SelectGpuId(std::string use_gpu);

  /// wd007

  void CreateHandle(cublasHandle_t *handle);
  void DestroyHandle(cublasHandle_t handle);

  void SelectGpu(int32 gpu_idx);
  int  SelectGpu();
  /// select recommend gpu card
  void SelectPreferGpu(int32 gpu_idx);

  void GetBandwidth(int32 gpu_idx, float &d2h, float &h2d);

  bool Initialize();

  int GetDeviceId();

  int MPISelectGpu(MPIGpuInfo *gpuinfo, MPI_Win &win, int thread_idx, int num_threads);
  ////

  /// Check if the CUDA GPU is selected for use
  bool Enabled() const {
    return (active_gpu_id_ > -1);
  }

  /// Get the active GPU id
  int32 ActiveGpuId() {
    return active_gpu_id_;
  }

  /// Returns true if either we have no GPU, or we have a GPU
  /// and it supports double precision.
  bool DoublePrecisionSupported();

  void SetVerbose(bool verbose) {  verbose_ = verbose; }

  /// Sum the IO time
  void AccuProfile(const std::string &key, double time);
  void PrintProfile();

  void PrintMemoryUsage() const;

  void ResetProfile() {
    profile_map_.clear();
  }

  /// Get the actual GPU memory use stats
  std::string GetFreeMemory(int64* free = NULL, int64* total = NULL) const;
  /// Get the name of the GPU
  void DeviceGetName(char* name, int32 len, int32 dev);

  /// Check if GPU is in good condition by multiplying small matrices on GPU+CPU.
  /// Overheated GPUs may give inaccurate results, which we want to detect.
  void CheckGpuHealth();

  /// If Enabled(), returns the number n of bytes such that the matrix stride
  /// will always be a multiple of n (from properties_.textureAlignment).
  /// Otherwise, return 16, which is the stride used for CPU matrices.
  int32 GetMatrixAlignment() const;

 private:
  CuDevice();
  CuDevice(CuDevice&); // Disallow.
  CuDevice &operator=(CuDevice&);  // Disallow.


  static CuDevice global_device_;
  cublasHandle_t handle_;

  /// Check if the GPU run in compute exclusive mode Returns true if it is
  /// running in compute exclusive mode and we have a GPU.  Returns false
  /// otherwise.  Sets error to true if there was some error, such as that we
  /// were running in compute exclusive modes but no GPUs available; otherwise
  /// sets it to false.
  bool IsComputeExclusive();

  /// Automatically select GPU and get CUDA context.  Returns true on success.
  bool SelectGpuIdAuto();

  /// Try to get CUDA context on manually selected GPU.  Return true on success.
  bool SelectGpuIdManual(int32 gpu_id);

  void FinalizeActiveGpu();

  /// Should only be called if Enabled() == true.
  int32 MajorDeviceVersion();

  /// Should only be called if Enabled() == true.
  int32 MinorDeviceVersion();

  unordered_map<std::string, double, StringHasher> profile_map_;

  /// active_gpu_id_ values:
  /// -3 default (default, the SelectGpuId was not called, we did not want to use GPU)
  /// -2 SelectGpuId was called, but no GPU was present
  /// -1 SelectGpuId was called, but the GPU was manually disabled
  /// 0..N Normal GPU IDs
  int32 active_gpu_id_;

  int64 free_memory_at_startup_;

  cudaDeviceProp properties_;

  bool verbose_;

  CuMemoryAllocator allocator_;

  std::vector<GpuInfo> gpuinfo_;

}; // class CuDevice

// This function is declared as a more convenient way to get the CUDA device handle for use
// in the CUBLAS v2 API, since we so frequently need to access it.
inline cublasHandle_t GetCublasHandle() { return CuDevice::Instantiate().GetHandle(); }

inline void CreateCublasHandle(cublasHandle_t *handle) { CuDevice::Instantiate().CreateHandle(handle);}
inline void DestroyCublasHandle(cublasHandle_t handle) { CuDevice::Instantiate().DestroyHandle(handle);}

}  // namespace

#endif // HAVE_CUDA


#endif
