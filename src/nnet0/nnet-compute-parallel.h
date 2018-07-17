// nnet0/nnet-compute-parallel.h

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

#ifndef KALDI_NNET_NNET_COMPUTE_H_
#define KALDI_NNET_NNET_COMPUTE_H_

#include "nnet2/am-nnet.h"
#include "hmm/transition-model.h"

#include <string>
#include <iomanip>
#include <mpi.h>

#include "nnet-trnopts.h"
#include "nnet0/nnet-randomizer.h"
#include "nnet0/nnet-loss.h"
#include "nnet0/nnet-nnet.h"
#include "nnet0/nnet-model-sync.h"

#include "cudamatrix/cu-device.h"

namespace kaldi {
namespace nnet0 {

struct NnetUpdateOptions {
    bool binary,
         crossvalidate,
         randomize;

    bool use_psgd;

    BaseFloat kld_scale;
    int32 skip_frames;
    int32 sweep_time;
    int32 dump_time;

    std::string feature_transform;
    std::string objective_function;
    std::string frame_weights;
    std::string use_gpu;
    std::string si_model_filename;
    std::string sweep_frames_str;
    bool  sweep_loop;
    bool  skip_inner;


    int32 length_tolerance;
    int32 update_frames;
    double dropout_retention;

    const NnetTrainOptions *trn_opts;
    const NnetDataRandomizerOptions *rnd_opts;
    const NnetParallelOptions *parallel_opts;

    NnetUpdateOptions(const NnetTrainOptions *trn_opts, const NnetDataRandomizerOptions *rnd_opts, const NnetParallelOptions *parallel_opts)
    	: binary(true),crossvalidate(false),randomize(true),use_psgd(false),kld_scale(-1.0),skip_frames(1),sweep_time(1), dump_time(0),
		  objective_function("xent"),frame_weights(""),use_gpu("yes"),sweep_frames_str("0"),sweep_loop(false), skip_inner(false),
		  length_tolerance(5),update_frames(-1),dropout_retention(0.0),
		  trn_opts(trn_opts),rnd_opts(rnd_opts),parallel_opts(parallel_opts){ }

  	  void Register(OptionsItf *po)
  	  {

  		  po->Register("binary", &binary, "Write output in binary mode");
  		  po->Register("cross-validate", &crossvalidate, "Perform cross-validation (don't backpropagate)");
	      po->Register("randomize", &randomize, "Perform the frame-level shuffling within the Cache::");


	      po->Register("feature-transform", &feature_transform, "Feature transform in Nnet format");

	      po->Register("objective-function", &objective_function, "Objective function : xent|mse");

	      po->Register("length-tolerance", &length_tolerance, "Allowed length difference of features/targets (frames)");

	      po->Register("frame-weights", &frame_weights, "Per-frame weights to scale gradients (frame selection/weighting).");

	      po->Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");

	      po->Register("dropout-retention", &dropout_retention, "number between 0..1, saying how many neurons to preserve (0.0 will keep original value");

	      po->Register("si-model",&si_model_filename, "kld speaker independent model filename");

	      po->Register("kld-scale", &kld_scale, "KLD regularization weight to the original training criterion");

	      po->Register("skip-frames", &skip_frames, "Compute model on selected frames(one frame out of every skip frames)");

	      po->Register("sweep-time", &sweep_time, "Sweep times for each utterance in skip frames training(Deprecated, use --sweep-frames instead)");
	      po->Register("sweep-frames", &sweep_frames_str, "Sweep frames indexes for each utterance in skip frames training, e.g. 0:1 for skip_frames = 2");
	      po->Register("sweep-loop", &sweep_loop, "Sweep all frames indexes for each utterance in skip frames training if true, "
	    		  "e.g. utt1:frame1, utt1:frame2, utt1:frame3 ...; otherwise sweep one frames index, e.g. utt1:frame1, utt2:frame2, utt3:frame3 ...");
	      po->Register("skip-inner", &skip_inner, "Skip frame in neural network inner or input");

	      po->Register("update-frames",&update_frames, "Every update-frames frames each client exchange gradient");

	      po->Register("use-psgd",&use_psgd, "use preconditional sgd instead of sgd, it always true while training with multi-machine");

	      po->Register("dump-time", &dump_time, "num hours frames between model dumping [ 0 == disabled ]");
  	  }
};


struct NnetStats {

    int32 num_done, num_no_tgt_mat, num_other_error;

    kaldi::int64 total_frames;
    Xent xent;
    Mse mse;
    MultiTaskLoss multitask;

    NnetStats():num_done(0),num_no_tgt_mat(0),num_other_error(0),total_frames(0){} //{ std::memset(this, 0, sizeof(*this)); }

    virtual ~NnetStats(){}

    virtual void  MergeStats(NnetUpdateOptions *opts, int root)
    {
    	int myid = opts->parallel_opts->myid;
    	MPI_Barrier(MPI_COMM_WORLD);

    	void *addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&this->total_frames));
    	MPI_Reduce(addr, (void*)(&this->total_frames), 1, MPI_UNSIGNED_LONG, MPI_SUM, root, MPI_COMM_WORLD);

    	addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&this->num_done));
    	MPI_Reduce(addr, (void*)(&this->num_done), 1, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);

    	addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&this->num_no_tgt_mat));
    	MPI_Reduce(addr, (void*)(&this->num_no_tgt_mat), 1, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);

    	addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&this->num_other_error));
    	MPI_Reduce(addr, (void*)(&this->num_other_error), 1, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);

        if (opts->objective_function == "xent") {
        		xent.Merge(myid, 0);
        } else if (opts->objective_function == "mse") {
        		mse.Merge(myid, 0);
        } else if (0 == opts->objective_function.compare(0, 9, "multitask")) {
        		multitask.Merge(myid, 0);
        } else {
        		KALDI_ERR << "Unknown objective function code : " << opts->objective_function;
        }

    }

    virtual void  Print(NnetUpdateOptions *opts, double time_now)
    {
        KALDI_LOG << "Done " << num_done << " files, " << num_no_tgt_mat
                  << " with no tgt_mats, " << num_other_error
                  << " with other errors. "
                  << "[" << (opts->crossvalidate?"CROSS-VALIDATION":"TRAINING")
                  << ", " << (opts->randomize?"RANDOMIZED":"NOT-RANDOMIZED")
                  << ", " << time_now/60 << " min, " << total_frames/time_now << " fps"
                  << "]";

        if (opts->objective_function == "xent") {
        	KALDI_LOG << xent.Report();
        } else if (opts->objective_function == "mse") {
        	KALDI_LOG << mse.Report();
        } else if (0 == opts->objective_function.compare(0, 9, "multitask")) {
    		KALDI_LOG << multitask.Report();
        } else {
        	KALDI_ERR << "Unknown objective function code : " << opts->objective_function;
        }

    }
};


void NnetUpdateParallel(const NnetUpdateOptions *opts,
		std::string	model_filename,
		std::string feature_rspecifier,
		std::string targets_rspecifier,
		Nnet *nnet,
		NnetStats *stats);


} // namespace nnet0
} // namespace kaldi

#endif // KALDI_NNET_NNET_COMPUTE_H_
