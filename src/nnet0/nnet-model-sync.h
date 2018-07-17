// nnet0/nnet-model-sync.h

// Copyright 2015-2016   Shanghai Jiao Tong University (author: Wei Deng)

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

#ifndef NNET_NNET_MODEL_SYNC_H_
#define NNET_NNET_MODEL_SYNC_H_

#include "thread/kaldi-semaphore.h"
#include "thread/kaldi-mutex.h"
#include "nnet0/nnet-nnet.h"

#include "cudamatrix/cu-device.h"
#include <mpi.h>

namespace kaldi {
namespace nnet0 {

struct NnetParallelOptions{
	int32 num_threads;
	int merge_size;
	int num_merge;
	int num_procs;
	int myid;
	int thread_level;
	BaseFloat global_momentum;
	BaseFloat global_learnrate;
	bool asgd_lock;
	std::string merge_func;
	std::string log_file;


	NnetParallelOptions():
									 num_threads(1),
									 merge_size(120000),
									 num_merge(0),
									 num_procs(-1),
									 myid(0),
									 thread_level(0),
									 global_momentum(-1.0),
									 global_learnrate(1.0),
									 asgd_lock(true),
									 merge_func("globalgradient"),
									 log_file("")
									 { }

	  void Register(OptionsItf *po) {
		  po->Register("num-threads", &num_threads, "Number of threads(GPUs) to use");
		  po->Register("global-momentum", &global_momentum, "Global momentum used in multi-machine paralization.");
		  po->Register("global-learnrate", &global_learnrate, "Global learning rate used in multi-machine paralization.");
		  po->Register("asgd-lock", &asgd_lock, "Apply lock on asgd training.");

	      if (this->num_procs >= 1)
	      {
	          po->Register("merge-size",&merge_size, "Multi-machine merge size");
	          po->Register("merge-function", &merge_func, "Multi-machine merge function");
	          po->Register("log-file", &log_file, "Each job log.");
	      }

	  }
};

class ModelAverageMerge;
class ModelGlobalSumMerge;
class ModelGlobalGradientMerge;
class ModelGlobalAdagradMerge;
class ModelMergeFunction;

class NnetModelSync{
public:
	typedef enum {
		kDstAddress = 0x0,
		kSrcAddress = 0x1,
	} AddressType;

	typedef enum {
		kCudaMemcpyHostToHost = 0x0,
		kCudaMemcpyHostToDevice,
		kCudaMemcpyDeviceToHost,
		kCudaMemcpyDeviceToDevice,
	} cudaMemcpyKind;

	NnetModelSync(Nnet *nnet, const NnetParallelOptions *opts=NULL):
		initialized_(false),data_(NULL),free_data_(NULL),dim_(0),nnet(nnet),
		opts_(opts),p_merge_func_(NULL)
	{
		//Init(nnet);
		MultiMachineInit();
	}

	~NnetModelSync()
	{
		Destory();
		delete[] isfinished_;
	}

	void LockModel() {
		model_mutex_.Lock();
	}
	void UnlockModel(){
		model_mutex_.Unlock();
	}

	void LockStates() {
		stats_mutex_.Lock();
	}
	void UnlockStates(){
		stats_mutex_.Unlock();
	}

	void GetWeight(Nnet *nnet);

	void SetWeight(Nnet *nnet);

	void Destory();

	int32 Dim(){return this->dim_;};

	void CopyToHost(Nnet *nnet)
	{
		*(this->nnet) = *nnet;
	}

	ModelMergeFunction *GetModelMergeFunction()
	{
		return p_merge_func_;
	}

	void MultiMachineInit();

    void ResetGradient()
    {
        for (int i = 0; i < opts_->num_threads; i++)
            reset_gradient_[i] = true;
    }

	void Initialize(Nnet *nnet)
	{
		model_mutex_.Lock();
		if (!initialized_)
		{
			isfinished_ = new bool[opts_->num_threads];
			reset_gradient_ = new bool[opts_->num_threads];
			for (int i = 0; i < opts_->num_threads; i++)
            {
				isfinished_[i] = false;
                reset_gradient_[i] = false;
            }
			this->GetWeight(nnet);
			InitMergeFunction();
			initialized_ = true;
		}
		model_mutex_.Unlock();
	}

	bool	*isfinished_;
    bool    *reset_gradient_;

private:
	friend class ModelAverageMerge;
	friend class ModelGlobalSumMerge;
	friend class ModelGlobalGradientMerge;
	friend class ModelGlobalAdagradMerge;


	int32 GetDim(Nnet *nnet);
	void Init(Nnet *nnet);
	void InitMergeFunction();

	bool	initialized_;

	Mutex model_mutex_;
	Mutex stats_mutex_;
	BaseFloat *data_;
	BaseFloat *free_data_;
	int32 dim_;
	Nnet *nnet;
	const NnetParallelOptions *opts_;
	ModelMergeFunction *p_merge_func_;

public:

#if HAVE_CUDA == 1
  kaldi::MPIGpuInfo *gpuinfo_;
  MPI_Win win;
#endif
};



class NnetParallelUtil{
public:
	std::string AddSuffix(std::string filename, int idx);
	std::string FAddSuffix(std::string filename, int idx);
	std::string GetFilename(std::string filename);
	int NumofMerge(std::string fn, int merge_size);
	int NumofCEMerge(std::string fn, int merge_size);
};



} // namespace nnet
} // namespace kaldi

#endif /* NNET_NNET_MODEL_SYNC_H_ */
