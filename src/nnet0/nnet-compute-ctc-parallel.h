// nnet0/nnet-compute-ctc-parallel.h

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

#ifndef KALDI_NNET_NNET_COMPUTE_CTC_PARALLEL_H_
#define KALDI_NNET_NNET_COMPUTE_CTC_PARALLEL_H_

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

#include "nnet0/nnet-compute-parallel.h"

namespace kaldi {
namespace nnet0 {

struct NnetCtcUpdateOptions : public NnetUpdateOptions {


    int32 num_stream;
    int32 max_frames;
    int32 batch_size;
    int32 targets_delay;


    NnetCtcUpdateOptions(const NnetTrainOptions *trn_opts, const NnetDataRandomizerOptions *rnd_opts, const NnetParallelOptions *parallel_opts)
    	: NnetUpdateOptions(trn_opts, rnd_opts, parallel_opts), num_stream(4), max_frames(25000), batch_size(0), targets_delay(0) { }

  	  void Register(OptionsItf *po)
  	  {
  	  	NnetUpdateOptions::Register(po);

	      	po->Register("num-stream", &num_stream, "---CTC--- BPTT multi-stream training");
	      	po->Register("max-frames", &max_frames, "Max number of frames to be processed");
	        po->Register("batch-size", &batch_size, "---LSTM--- BPTT batch size");
	        po->Register("targets-delay", &targets_delay, "---LSTM--- BPTT targets delay");
  	  }
};


struct NnetCtcStats: NnetStats {

    Ctc ctc;

    NnetCtcStats() { }

    void MergeStats(NnetUpdateOptions *opts, int root)
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
        }
        else if (opts->objective_function == "ctc") {
        		ctc.Merge(myid, 0);
        }
        else {
        		KALDI_ERR << "Unknown objective function code : " << opts->objective_function;
        }

    }

    void Print(NnetUpdateOptions *opts, double time_now)
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
        }
        else if (opts->objective_function == "ctc") {
        	KALDI_LOG << ctc.Report();
        } else {
        	KALDI_ERR << "Unknown objective function code : " << opts->objective_function;
        }
    }
};


void NnetCtcUpdateParallel(const NnetCtcUpdateOptions *opts,
		std::string	model_filename,
		std::string feature_rspecifier,
		std::string targets_rspecifier,
		Nnet *nnet,
		NnetCtcStats *stats);

void NnetCEUpdateParallel(const NnetCtcUpdateOptions *opts,
		std::string	model_filename,
		std::string feature_rspecifier,
		std::string targets_rspecifier,
		Nnet *nnet,
		NnetStats *stats);

} // namespace nnet0
} // namespace kaldi

#endif // KALDI_NNET_NNET_COMPUTE_CTC_PARALLEL_H_
