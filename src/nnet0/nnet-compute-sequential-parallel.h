// nnet0/nnet-compute-sequential-parallel.h

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

#ifndef KALDI_NNET_NNET_COMPUTE_SEQUENTIAL_H_
#define KALDI_NNET_NNET_COMPUTE_SEQUENTIAL_H_

#include "nnet2/am-nnet.h"
#include "hmm/transition-model.h"

#include <string>
#include <iomanip>
#include <mpi.h>

#include "nnet-trnopts.h"
#include "nnet-pdf-prior.h"
#include "nnet0/nnet-nnet.h"
#include "nnet0/nnet-model-sync.h"

#include "cudamatrix/cu-device.h"

namespace kaldi {
namespace nnet0 {
class SequentialTrainingSync;
class ModelMergeFunction;

struct NnetSequentialUpdateOptions {
  std::string criterion; // "mmi" or "mpfe" or "smbr"
  BaseFloat acoustic_scale,
          	  lm_scale,
	old_acoustic_scale,
    kld_scale,
  	  	  	  frame_smooth;

  bool drop_frames;
  bool one_silence_class;  // Affects MPE/SMBR>
  BaseFloat boost; // for MMI, boosting factor (would be Boosted MMI)... e.g. 0.1.

  std::string silence_phones_str; // colon-separated list of integer ids of silence phones,
                                  // for MPE/SMBR only.
  std::string sweep_frames_str;
  std::string sweep_frames_filename;

  int32 update_frames;
  int32 max_frames; // Allow segments maximum of one minute by default
  std::string use_gpu;
  std::string si_model_filename;
  bool use_psgd;

  //lstm
  int32 targets_delay;
  int32 batch_size;
  int32 num_stream;
  int32 dump_interval;
  int32 frame_limit;
  int32 skip_frames;
  int32 dump_time;
  //lstm

  bool  sweep_loop;
  bool  skip_inner;

  const NnetTrainOptions *trn_opts;
  const PdfPriorOptions *prior_opts;
  const NnetParallelOptions *parallel_opts;


  NnetSequentialUpdateOptions(const NnetTrainOptions *trn_opts, const PdfPriorOptions *prior_opts, const NnetParallelOptions *parallel_opts): criterion("mmi"),
		  	  	  	  	  	  	 acoustic_scale(0.1), lm_scale(0.1), old_acoustic_scale(0.0), kld_scale(-1.0), frame_smooth(-1.0),
		  	  	  	  	  	  	 drop_frames(true), one_silence_class(false), boost(0.0), sweep_frames_str("0"), sweep_frames_filename(""),
								 update_frames(-1),
				                 max_frames(6000),
  	  	  	  	  	  	  	  	 use_gpu("yes"),
								 si_model_filename(""),
								 use_psgd(false),
								 targets_delay(0), batch_size(0), num_stream(0), dump_interval(0),frame_limit(10000),skip_frames(1),dump_time(0),
								 sweep_loop(false), skip_inner(false),
								 trn_opts(trn_opts),
								 prior_opts(prior_opts),
								 parallel_opts(parallel_opts){ }

  void Register(OptionsItf *po) {

	  po->Register("criterion", &criterion, "Criterion, 'mmi'|'mpfe'|'smbr', "
	                   "determines the objective function to use.  Should match "
	                   "option used when we created the examples.");

      po->Register("acoustic-scale", &acoustic_scale,
                  "Scaling factor for acoustic likelihoods");
      po->Register("lm-scale", &lm_scale,
                  "Scaling factor for \"graph costs\" (including LM costs)");
      po->Register("old-acoustic-scale", &old_acoustic_scale,
                  "Add in the scores in the input lattices with this scale, rather "
                  "than discarding them.");
      po->Register("kld-scale", &kld_scale,
                  "KLD regularization weight to the original training criterion");

      po->Register("frame-smooth", &frame_smooth,
                  "making the sequence- discriminative training criterion closer to the frame-discriminative training criterion");

      po->Register("max-frames",&max_frames, "Maximum number of frames a segment can have to be processed");
      po->Register("drop-frames", &drop_frames,
                        "Drop frames, where is zero den-posterior under numerator path "
                        "(ie. path not in lattice)");
      po->Register("boost", &boost, "Boosting factor for boosted MMI (e.g. 0.1)");
      po->Register("one-silence-class", &one_silence_class, "If true, newer "
                     "behavior which will tend to reduce insertions.");
      po->Register("silence-phones", &silence_phones_str,
                     "For MPFE or SMBR, colon-separated list of integer ids of "
                     "silence phones, e.g. 1:2:3");
      po->Register("sweep-frames", &sweep_frames_str, "Sweep frames index for each utterance in skip frames training, e.g. 0");
      po->Register("sweep-frames-filename", &sweep_frames_filename, "Sweep frames index in skip frames training which can be custom setting for each utterance, this would cover sweep-frames option.");

      po->Register("update-frames",&update_frames, "Every update-frames frames each client exchange gradient");
      po->Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");

      po->Register("use-psgd",&use_psgd, "use preconditional sgd instead of sgd, it always true while training with multi-machine");

      po->Register("si-model",&si_model_filename, "kld speaker independent model filename");

      //<jiayu>
      po->Register("targets-delay", &targets_delay, "---LSTM--- BPTT targets delay");

      po->Register("batch-size", &batch_size, "---LSTM--- BPTT batch size");

      po->Register("num-stream", &num_stream, "---LSTM--- BPTT multi-stream training");

      po->Register("dump-interval", &dump_interval, "---LSTM--- num utts between model dumping [ 0 == disabled ]");
      
      po->Register("frame-limit", &frame_limit, "Max number of frames to be processed in lstm");

      po->Register("skip-frames", &skip_frames, "LSTM based model skip frames for next input");

      po->Register("dump-time", &dump_time, "num hours frames between model dumping [ 0 == disabled ]");
      //</jiayu>

      po->Register("sweep-loop", &sweep_loop, "Sweep all frames indexes for each utterance in skip frames training if true, "
      	    		  "e.g. utt1:frame1, utt1:frame2, utt1:frame3 ...; otherwise sweep one frames index, e.g. utt1:frame1, utt2:frame2, utt3:frame3 ...");
      po->Register("skip-inner", &skip_inner, "Skip frame in neural network inner or input");
  }
};


struct NnetSequentialStats {

    int32 num_done, num_no_num_ali, num_no_den_lat,
          num_other_error, num_frm_drop;

    kaldi::int64 total_frames;
    double lat_like; // total likelihood of the lattice
    double lat_ac_like; // acoustic likelihood weighted by posterior.
    double total_mmi_obj, mmi_obj;
    double total_post_on_ali, post_on_ali;
    double total_frame_acc;


    void MergeStats(NnetSequentialUpdateOptions *opts, int root)
    {
    		int myid = opts->parallel_opts->myid;
    		MPI_Barrier(MPI_COMM_WORLD);

    		void *addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&total_frames));
    		MPI_Reduce(addr, (void*)(total_frames), 1, MPI_UNSIGNED_LONG, MPI_SUM, root, MPI_COMM_WORLD);

    		addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&num_no_num_ali));
    		MPI_Reduce(addr, (void*)(&num_no_num_ali), 1, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);

    		addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&num_no_den_lat));
    		MPI_Reduce(addr, (void*)(&num_no_den_lat), 1, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);

    		addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&num_other_error));
    		MPI_Reduce(addr, (void*)(&num_other_error), 1, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);

    		addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&num_done));
    		MPI_Reduce(addr, (void*)(&num_done), 1, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);

    		addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&num_frm_drop));
    		MPI_Reduce(addr, (void*)(&num_frm_drop), 1, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);

    		addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&total_mmi_obj));
    		MPI_Reduce(addr, (void*)(&total_mmi_obj), 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

    		addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&total_frame_acc));
    		MPI_Reduce(addr, (void*)(&total_frame_acc), 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
    }

  NnetSequentialStats() { std::memset(this, 0, sizeof(*this)); }

  void Print(std::string criterion, double time_now)
  {
	  	  KALDI_LOG << "Time taken = " << time_now/60 << " min; processed "
	                << (total_frames/time_now) << " frames per second.";

	      KALDI_LOG << "Done " << num_done << " files, "
	                << num_no_num_ali << " with no numerator alignments, "
	                << num_no_den_lat << " with no denominator lattices, "
	                << num_other_error << " with other errors.";

	      if (criterion == "mmi")
	      KALDI_LOG << "Overall MMI-objective/frame is "
	                << std::setprecision(8) << (total_mmi_obj/total_frames)
	                << " over " << total_frames << " frames."
	                << " (average den-posterior on ali " << (total_post_on_ali/total_frames) << ","
	                << " dropped " << num_frm_drop << " frames with num/den mismatch)";
	      else
	      KALDI_LOG << "Overall average frame-accuracy is "
	                << (total_frame_acc/total_frames) << " over " << total_frames
	                << " frames.";
  }
  //void Add(const NnetSequentialStats &other);
};


void NnetSequentialUpdateParallel(const NnetSequentialUpdateOptions *opts,
		std::string feature_transform,
		std::string	model_filename,
		std::string transition_model_filename,
		std::string feature_rspecifier,
		std::string den_lat_rspecifier,
		std::string num_ali_rspecifier,
		std::string sweep_frames_rspecifier,
		Nnet *nnet,
		NnetSequentialStats *stats);

} // namespace nnet0
} // namespace kaldi

#endif // KALDI_NNET_NNET_COMPUTE_SEQUENTIAL_H_
