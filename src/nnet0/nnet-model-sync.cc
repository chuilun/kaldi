// nnet0/nnet-model-sync.cc

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

#include "nnet0/nnet-utils.h"

#include "nnet0/nnet-model-sync.h"
#include "nnet0/nnet-model-merge-function.h"

namespace kaldi {
namespace nnet0 {

void
NnetModelSync::Init(Nnet *nnet)
{
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
	if (NULL != this->free_data_)
		return;

	size_t size = 0;
	void *data = NULL;
	void *free_data = NULL;
	int32 dim = 0;

	dim = this->GetDim(nnet);

	size = dim * sizeof(BaseFloat)+16;
	CU_SAFE_CALL(cudaHostAlloc((void**) &free_data, size, cudaHostAllocPortable)); //cudaHostAllocDefault
	data = (free_data ? (void *)( (((unsigned long)*(&free_data)) + 15) & ~0xFUL ) : NULL) ;

	if (NULL != data)
	{
		this->data_ = static_cast<BaseFloat*> (data);
		this->free_data_ = static_cast<BaseFloat*> (free_data);
		this->dim_ = dim;
	}
	else
	{
	    throw std::bad_alloc();
	}
 }else
#endif
        {
                // not implemented for CPU yet
                // return 0;
        }

}

void
NnetModelSync::MultiMachineInit()
{
    if (opts_->num_procs > 1)
    {
        //p_merge_func_ = ModelMergeFunction::Factory(opts_, this);

#if HAVE_CUDA == 1
        gpuinfo_ = (MPIGpuInfo*)malloc(opts_->num_procs * opts_->num_threads * sizeof(MPIGpuInfo));
        std::memset(gpuinfo_, 0, opts_->num_procs * opts_->num_threads * sizeof(MPIGpuInfo));
#endif
    }
}

void
NnetModelSync::InitMergeFunction()
{
	if (opts_->num_procs > 1 && NULL == p_merge_func_)
	{
		p_merge_func_ = ModelMergeFunction::Factory(opts_, this);
	}
}
void
NnetModelSync::Destory()
{
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
	if (NULL != this->free_data_)
	{
		CU_SAFE_CALL(cudaFreeHost(this->free_data_));
		this->free_data_ = NULL;
		this->data_ = NULL;
		this->dim_ = 0;
	}
 }else
#endif
        {
                // not implemented for CPU yet
                // return 0;
        }

}

int32
NnetModelSync::GetDim(Nnet *nnet)
{
	return nnet->GetDim();
}

void
NnetModelSync::GetWeight(Nnet *nnet)
{
	if (NULL == this->data_)
		this->Init(nnet);

	void *host_data_ = (void*)this->data_;
	// device to host
	nnet->WeightCopy(host_data_, NnetModelSync::kDstAddress, NnetModelSync::kCudaMemcpyDeviceToHost);
}

void
NnetModelSync::SetWeight(Nnet *nnet)
{
	KALDI_ASSERT(this->data_ != NULL);

	void *host_data_ = (void*)this->data_;
	// host to device
	nnet->WeightCopy(host_data_, NnetModelSync::kSrcAddress, NnetModelSync::kCudaMemcpyHostToDevice);
}

/*
 * 'ark,o:copy-feats scp:exp/tri_dnn_mmi/scplist/train.scp ark:- |'
 */

std::string NnetParallelUtil::AddSuffix(std::string filename, int idx)
{
  char buf[1024];
  char suf[1024], ext[1024], fn[1024];
  int  len;

  const char *pfn = filename.c_str();
  len = strlen(pfn);
  const char *p1, *p2;
  p1 = strstr(pfn,"scp:");
  if (NULL == p1) return "";
  p2 = strchr(p1, ' ');
  if (NULL == p2) p2 = pfn+len;

  strncpy(fn, pfn, p2-pfn); fn[p2-pfn] = '\0';
  int l1 = strlen(fn);
  char *p3 = strrchr(fn, '.');
  *p3='\0';

  strncpy(suf,p3+1, fn+l1-p3); suf[fn+l1-p3]='\0';

  strncpy(ext, p2, pfn+len-p2); ext[pfn+len-p2]='\0';

  sprintf(buf,"%s.%d.%s%s",fn,idx,suf, ext);

  return buf;
}

std::string NnetParallelUtil::FAddSuffix(std::string filename, int idx)
{
  char buf[1024];
  char ext[128], fn[128];
  int  len;

  const char *pfn = filename.c_str();
  len = strlen(pfn);
  const char *p2;

  p2 = strchr(pfn, '.');

  strncpy(fn,pfn, p2-pfn); fn[p2-pfn]='\0';
  strncpy(ext, p2+1, pfn+len-p2); ext[pfn+len-p2]='\0';

  sprintf(buf,"%s.%d.%s",fn,idx,ext);

  return buf;
}

std::string NnetParallelUtil::GetFilename(std::string filename)
{
  char fn[128];

  const char *pfn = filename.c_str();
  const char *p1, *p2;
  p1 = strstr(pfn,"scp:");
  p2 = strchr(p1, ' ');


  strncpy(fn,p1+4, p2-p1-4); fn[p2-p1-4]='\0';

  return fn;
}

int NnetParallelUtil::NumofMerge(std::string fn, int merge_size)
{
	std::string sfn = fn+".len";
	std::ifstream in(sfn.c_str());
	std::string str, featname;
	int len, piece = 0;
	size_t frames = 0;
	while(std::getline(in, str))
	{
		std::istringstream ss(str);
		ss>>featname>>len;

		if (frames + len > merge_size)
		{
			piece++;
			frames = 0;
		}
		frames += len;
	}

	if (frames > merge_size/2)
		piece++;

	return piece;
}

int NnetParallelUtil::NumofCEMerge(std::string fn, int merge_size)
{
	std::string sfn = fn+".len";
	std::ifstream in(sfn.c_str());
	std::string str, featname;
	int len, piece = 0;
	size_t frames = 0;
	while(std::getline(in, str))
	{
		std::istringstream ss(str);
		ss>>featname>>len;

		frames += len;
	}

	piece = frames/merge_size + 1;

	return piece;
}

} // namespace nnet
} // namespace kaldi
