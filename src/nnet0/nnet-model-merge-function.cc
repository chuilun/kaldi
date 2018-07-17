// nnet0/nnet-model-merge-function.cc

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


#include "matrix/cblas-wrappers.h"

#include "nnet0/nnet-model-merge-function.h"


namespace kaldi {
namespace nnet0 {


ModelMergeFunction*
ModelMergeFunction::Factory(const NnetParallelOptions *opts, NnetModelSync *model_sync)
{
	ModelMergeFunction* ret = NULL;
	MerFunType type;
	if (opts->merge_func == "average")
		type = AVERAGE;
	else if (opts->merge_func == "globalsum")
		type = GLOBAL_SUM;
	else if (opts->merge_func == "globalgradient")
		type = GLOBAL_GRADIENT;
	else if (opts->merge_func == "globaladagrad")
		type = GLOBAL_ADAGRAD;

	switch(type) {
	case AVERAGE:  ret = new ModelAverageMerge(opts, model_sync);  break;
	case GLOBAL_SUM:      ret = new ModelGlobalSumMerge(opts, model_sync);     break;
	case GLOBAL_GRADIENT:      ret = new ModelGlobalGradientMerge(opts, model_sync);     break;
	case GLOBAL_ADAGRAD:		   ret = new ModelGlobalAdagradMerge(opts, model_sync); break;
	default: KALDI_ERR<< "Unknown MergeFunction type";
	break;
  }
  return ret;
}

int ModelMergeFunction:: MergeStatus(int status)
{
	MPI_Barrier(MPI_COMM_WORLD);
	int total_status = 0;
	MPI_Allreduce(&status, (void*)(&total_status), 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	if (total_status < opts->num_procs)
		this->misLastMerge = true;
	return	total_status;
}

/**
 * Model average.
 */
/*
void ModelAverageMerge::Merge(int root)
{

	//cblas_Xscal(model_sync_->Dim(), 1.0/opts->num_procs, model_sync_->data_, 1);

	void *srcaddr = (void *) (opts->myid==root ? MPI_IN_PLACE : this->model_sync_->data_);

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(srcaddr, (void*)(this->model_sync_->data_),
			this->model_sync_->dim_, MPI_FLOAT, MPI_SUM, root, MPI_COMM_WORLD);

	if (opts->myid == root)
	{
		cblas_Xscal(model_sync_->Dim(), 1.0/opts->num_procs, model_sync_->data_, 1);
	}


	//std::cout<<"Reduce finished!"<<std::endl;

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast((void*)(model_sync_->data_), model_sync_->Dim(), MPI_FLOAT, root, MPI_COMM_WORLD);
	//std::cout<<"Bcast finished!"<<std::endl;
	this->mLeftMerge--;
}
*/

void ModelAverageMerge::Merge(int root)
{

    void *srcaddr = (void *) MPI_IN_PLACE;
    void *dstaddr = (void *) this->model_sync_->data_;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(srcaddr, dstaddr, this->model_sync_->dim_, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    cblas_Xscal(model_sync_->Dim(), 1.0/opts->num_procs, model_sync_->data_, 1);

    this->mLeftMerge--;
}

void
ModelGlobalSumMerge::Init()
{
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {

	if (NULL != this->nnet_free_data_)
		return;

	size_t size = 0;
	void *nnet_data_ = NULL;
	void *nnet_free_data_ = NULL;

	this->dim_ = this->model_sync_->Dim();

	size = dim_ * sizeof(BaseFloat)+16;
	CU_SAFE_CALL(cudaHostAlloc((void**) &nnet_free_data_, size, cudaHostAllocPortable)); //cudaHostAllocDefault
	nnet_data_ = (nnet_free_data_ ? (void *)( (((unsigned long)*(&nnet_free_data_)) + 15) & ~0xFUL ) : NULL) ;

	if (NULL != nnet_data_)
	{
		this->nnet_data_ = static_cast<BaseFloat*> (nnet_data_);
		this->nnet_free_data_ = static_cast<BaseFloat*> (nnet_free_data_);

		CU_SAFE_CALL(cudaMemcpy(this->nnet_data_, this->model_sync_->data_, dim_*sizeof(BaseFloat), cudaMemcpyHostToHost));
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


void ModelGlobalSumMerge::Merge(int root)
{

	NnetModelSync *model_sync = this->model_sync_;

	float eta = this->mLearningRate;

	cblas_Xaxpy(this->dim_, -1, this->nnet_data_, 1, model_sync->data_, 1);

	void *addr = (void *) (opts->myid==root ? MPI_IN_PLACE : model_sync->data_);

	MPI_Reduce(addr, (void*)(model_sync->data_), model_sync->Dim(), MPI_FLOAT, MPI_SUM, root, MPI_COMM_WORLD);

	if (opts->myid==root)
	{
		//cblas_Xscal(dim_, 1.0/opts->num_procs, model_sync->data_, 1);
		cblas_Xaxpy(this->dim_, eta/opts->num_procs, model_sync->data_, 1, this->nnet_data_, 1);
	}

	//std::cout<<"Adagrad Reduce finished!"<<std::endl;
			//t1 = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast((void*)(this->nnet_data_), dim_, MPI_FLOAT, root, MPI_COMM_WORLD);

	//std::memcpy(model_sync->data_, this->nnet_data_, dim_ * sizeof(BaseFloat));
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
	    CU_SAFE_CALL(cudaMemcpy(model_sync->data_, this->nnet_data_, dim_ * sizeof(BaseFloat), cudaMemcpyHostToHost));
    }else
#endif
    {
        std::memcpy(model_sync->data_, this->nnet_data_, dim_ * sizeof(BaseFloat));
    }

	//t2 = MPI_Wtime();
			//c = (t2-t1)*1000;
			//std::cout<<"Bcast finished!"<<std::endl;
			//printf("ModelAdagrad ---- Reduce, Adagrad, Bcast, total time: %.2lf %.2lf %.2lf %.2lf ms.\n",a,b,c,a+b+c);

	this->mLeftMerge--;

}

/**
 * Model global gradient sum merge.
 */

void ModelGlobalGradientMerge::Init()
{
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {

	if (NULL != this->nnet_free_data_)
		return;

	size_t size = 0;
	void *nnet_data_ = NULL;
	void *nnet_free_data_ = NULL;
	void *gradient_data_ = NULL;
	void *gradient_free_data_ = NULL;

	this->dim_ = this->model_sync_->Dim();

	size = dim_ * sizeof(BaseFloat)+16;
	CU_SAFE_CALL(cudaHostAlloc((void**) &nnet_free_data_, size, cudaHostAllocPortable)); //cudaHostAllocDefault
	nnet_data_ = (nnet_free_data_ ? (void *)( (((unsigned long)*(&nnet_free_data_)) + 15) & ~0xFUL ) : NULL) ;

	if (NULL != nnet_data_)
	{
		this->nnet_data_ = static_cast<BaseFloat*> (nnet_data_);
		this->nnet_free_data_ = static_cast<BaseFloat*> (nnet_free_data_);

		CU_SAFE_CALL(cudaMemcpy(this->nnet_data_, this->model_sync_->data_, dim_*sizeof(BaseFloat), cudaMemcpyHostToHost));
	}
	else
	{
	    throw std::bad_alloc();
	}

	CU_SAFE_CALL(cudaHostAlloc((void**) &gradient_free_data_, size, cudaHostAllocPortable)); //cudaHostAllocDefault
	gradient_data_ = (gradient_free_data_ ? (void *)( (((unsigned long)*(&gradient_free_data_)) + 15) & ~0xFUL ) : NULL) ;

	if (NULL != gradient_data_)
	{
		this->gradient_data_ = static_cast<BaseFloat*> (gradient_data_);
		this->gradient_free_data_ = static_cast<BaseFloat*> (gradient_free_data_);

		CU_SAFE_CALL(cudaMemset(this->gradient_data_, 0, dim_*sizeof(BaseFloat)));
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


void ModelGlobalGradientMerge::Merge(int root)
{
    double t1, t2, tk;
    Timer tm;
	void *srcaddr = (void *) (opts->myid==root ? MPI_IN_PLACE : this->model_sync_->data_);

	t1 = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(srcaddr, (void*)(this->model_sync_->data_),
			this->model_sync_->dim_, MPI_FLOAT, MPI_SUM, root, MPI_COMM_WORLD);
	t2 = MPI_Wtime();
    KALDI_VLOG(2) << "MPI_Reduce: " << t2-t1;

    tm.Reset();
	if (opts->myid == root)
	{
		// average W(t)
		cblas_Xscal(model_sync_->Dim(), 1.0/opts->num_procs, model_sync_->data_, 1);
		// global gradient G(t) = average W(t) - W(t-1)
		cblas_Xaxpy(this->dim_, -1, this->nnet_data_, 1, model_sync_->data_, 1);
		// delta(t) = mmt * delta_(t-1) + lr * G(t)
		if (mmt < 0.0) mmt = 1.0 - 1.0/this->opts->num_procs;
		cblas_Xscal(this->dim_, mmt, this->gradient_data_, 1);
		cblas_Xaxpy(this->dim_, this->mLearningRate, model_sync_->data_, 1, this->gradient_data_, 1);

	}
    tk = tm.Elapsed();
    KALDI_VLOG(2) << "MKL merge: " << tk;


	//std::cout<<"Adagrad Reduce finished!"<<std::endl;
	t1 = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast((void*)(this->nnet_data_), dim_, MPI_FLOAT, root, MPI_COMM_WORLD);
	t2 = MPI_Wtime();
    KALDI_VLOG(2) << "MPI_Bcast: " << t2-t1;

	//std::memcpy(model_sync->data_, this->nnet_data_, dim_ * sizeof(BaseFloat));
#if HAVE_CUDA == 1
    if (CuDevice::Instantiate().Enabled()) {
	    CU_SAFE_CALL(cudaMemcpy(this->model_sync_->data_, this->nnet_data_, dim_ * sizeof(BaseFloat), cudaMemcpyHostToHost));
	}else 
#endif
    {
	    std::memcpy(this->model_sync_->data_, this->nnet_data_, dim_ * sizeof(BaseFloat));
	}
    
	this->mLeftMerge--;

}


/**
 * Model global adagrad merge.
 */


void ModelGlobalAdagradMerge::AdaGrad(int32 dim, BaseFloat eta, BaseFloat K, const BaseFloat *gradient)
{
	  for(size_t i=0; i<dim; i++)
	  {
	        nnet_data_[i] += (eta/sqrt(K+(gradient[i])*(gradient[i])))*(gradient[i]);
	  }
}


void ModelGlobalAdagradMerge::Merge(int root)
{

	NnetModelSync *model_sync = this->model_sync_;

	float eta = this->mLearningRate;

	cblas_Xaxpy(this->dim_, -1, this->nnet_data_, 1, model_sync->data_, 1);

	void *addr = (void *) (opts->myid==root ? MPI_IN_PLACE : model_sync->data_);

	MPI_Reduce(addr, (void*)(model_sync->data_), model_sync->Dim(), MPI_FLOAT, MPI_SUM, root, MPI_COMM_WORLD);

	if (opts->myid==root)
	{
		//KALDI_VLOG(1) << "ModelGlobalAdagradMerge::AdaGrad" << " eta: " << eta << " dim: " << dim_;
		cblas_Xscal(dim_, 1.0/opts->num_procs, model_sync->data_, 1);
		this->AdaGrad(dim_, eta, 1, model_sync->data_);
	}

	//std::cout<<"Adagrad Reduce finished!"<<std::endl;
			//t1 = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast((void*)(this->nnet_data_), dim_, MPI_FLOAT, root, MPI_COMM_WORLD);

	//std::memcpy(model_sync->data_, this->nnet_data_, dim_ * sizeof(BaseFloat));
#if HAVE_CUDA == 1
    if (CuDevice::Instantiate().Enabled()) {
	    CU_SAFE_CALL(cudaMemcpy(model_sync->data_, this->nnet_data_, dim_ * sizeof(BaseFloat), cudaMemcpyHostToHost));
    }else
#endif
    {
        std::memcpy(model_sync->data_, this->nnet_data_, dim_ * sizeof(BaseFloat));
    }

	//t2 = MPI_Wtime();
			//c = (t2-t1)*1000;
			//std::cout<<"Bcast finished!"<<std::endl;
			//printf("ModelAdagrad ---- Reduce, Adagrad, Bcast, total time: %.2lf %.2lf %.2lf %.2lf ms.\n",a,b,c,a+b+c);

	this->mLeftMerge--;

}


} // namespace nnet
} // namespace kaldi

