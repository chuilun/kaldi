// nnet0/nnet-model-merge-function.h

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

#ifndef NNET_NNET_MODEL_MERGE_FUNCTION_H_
#define NNET_NNET_MODEL_MERGE_FUNCTION_H_

#include <cassert>
#include <limits>
#include <cmath>
#include <sstream>

#include "nnet0/nnet-model-sync.h"

namespace kaldi {
namespace nnet0 {

class ModelMergeFunction
{
public:
	/// Enum with merge function types
	typedef enum {
		MER_FUN_I = 0x0400,
		AVERAGE,
		GLOBAL_ADAGRAD,
		GLOBAL_SUM,
		GLOBAL_GRADIENT
	} MerFunType;

	/// Factory for creating objective function instances
	static ModelMergeFunction* Factory(const NnetParallelOptions *opts, NnetModelSync *model_sync);
	////////////////////////////////////////////////////////////
	 /// Interface specification
public:
	 ModelMergeFunction(const NnetParallelOptions *opts, NnetModelSync *model_sync)
		: mLeftMerge(opts->num_merge), mCurrentSamples(0),misLastMerge(false), opts(opts), model_sync_(model_sync)
	 { }

	 virtual ~ModelMergeFunction()
	 { }

	 virtual MerFunType GetTypeId() = 0;

	 /// evaluates the data, calculate global error
	 virtual void Merge(int root) = 0;

	 int leftMerge()
	 {
		 return mLeftMerge;
	 }

	 int CurrentMergeCache()
	 {
		 return mCurrentSamples;
	 }

	 void MergeCacheReset()
	 {
		 mCurrentSamples = 0;
	 }

	 void AddMergeCache(int frames)
	 {
		 mCurrentSamples += frames;
	 }

	 int MergeStatus(int status);

	 bool isLastMerge()
	 {
		 return misLastMerge;
	 }

protected:
	 int mLeftMerge;
	 int mCurrentSamples;
	 bool misLastMerge;
	 const NnetParallelOptions *opts;
	 NnetModelSync *model_sync_;
};

/**
 * Model average.
 */
class ModelAverageMerge : public ModelMergeFunction
{

public:
		ModelAverageMerge(const NnetParallelOptions *opts, NnetModelSync *model_sync)
			:ModelMergeFunction(opts, model_sync)
		{
			//std::cout<<"ModelAverageMerge"<<std::endl;
		}

		virtual ~ModelAverageMerge()
		{ }


		MerFunType GetTypeId()
		{ return ModelAverageMerge::AVERAGE; }

		void Merge(int root);

};

/**
 * Model global gradient sum merge.
 */
class ModelGlobalSumMerge : public ModelMergeFunction
{

public:
		ModelGlobalSumMerge(const NnetParallelOptions *opts, NnetModelSync *model_sync)
		  	  :ModelMergeFunction(opts, model_sync), mLearningRate(1.0),nnet_data_(NULL),nnet_free_data_(NULL),dim_(0)
			{
				Init();
			}

		virtual ~ModelGlobalSumMerge()
		{ }


		virtual MerFunType GetTypeId()
		{ return ModelMergeFunction::GLOBAL_SUM; }

		virtual void Merge(int root);

		 float LearningRate()
		 {
		 	return mLearningRate;
		 }

		 void SetLearningRate(float lrate=1.0)
		 {
		 	mLearningRate = lrate;
		 }

protected:
		 void Init();

		float mLearningRate;
		BaseFloat *nnet_data_;
		BaseFloat *nnet_free_data_;
		int32 dim_;
};

/**
 * Model global gradient sum merge.
 */
class ModelGlobalGradientMerge : public ModelMergeFunction
{

public:
	ModelGlobalGradientMerge(const NnetParallelOptions *opts, NnetModelSync *model_sync)
		  	  : ModelMergeFunction(opts, model_sync), mLearningRate(1.0),mmt(opts->global_momentum),nnet_data_(NULL),nnet_free_data_(NULL),gradient_data_(NULL),gradient_free_data_(NULL),dim_(0)
			{
				Init();
			}

		virtual ~ModelGlobalGradientMerge()
		{ }


		virtual MerFunType GetTypeId()
		{ return ModelGlobalGradientMerge::GLOBAL_GRADIENT; }

		virtual void Merge(int root);

		 float LearningRate()
		 {
		 	return mLearningRate;
		 }

		 void SetLearningRate(float lrate=1.0)
		 {
		 	mLearningRate = lrate;
		 }

protected:
		 void Init();

		float mLearningRate;
		float mmt;
		BaseFloat *nnet_data_;
		BaseFloat *nnet_free_data_;
		BaseFloat *gradient_data_;
		BaseFloat *gradient_free_data_;
		int32 dim_;
};

/**
 * Model global adagrad merge.
 */
class ModelGlobalAdagradMerge : public ModelGlobalSumMerge
{

public:
		ModelGlobalAdagradMerge(const NnetParallelOptions *opts, NnetModelSync *model_sync)
	  	  : ModelGlobalSumMerge(opts, model_sync)
		{

		}

		virtual ~ModelGlobalAdagradMerge()
		{ }


		MerFunType GetTypeId()
		{ return ModelMergeFunction::GLOBAL_ADAGRAD; }

		virtual void Merge(int root);

		void AdaGrad(int32 dim, BaseFloat eta, BaseFloat K, const BaseFloat *gradient);


};


} // namespace nnet
} // namespace kaldi




#endif /* NNET_NNET_MODEL_MERGE_FUNCTION_H_ */
