// nnet0/nnet-compute-lstm-lm-parallel.h

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

#ifndef KALDI_NNET_NNET_COMPUTE_LSTM_LM_PARALLEL_H_
#define KALDI_NNET_NNET_COMPUTE_LSTM_LM_PARALLEL_H_

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

#include "nnet0/nnet-compute-lstm-asgd.h"

namespace kaldi {
namespace nnet0 {

struct NnetLstmLmUpdateOptions : public NnetLstmUpdateOptions {

	std::string class_boundary;
	int32 num_class;
	BaseFloat var_penalty;

    NnetLstmLmUpdateOptions(const NnetTrainOptions *trn_opts, const NnetDataRandomizerOptions *rnd_opts, const NnetParallelOptions *parallel_opts)
    	: NnetLstmUpdateOptions(trn_opts, rnd_opts, parallel_opts), class_boundary(""), num_class(0), var_penalty(0) { }

  	  void Register(OptionsItf *po)
  	  {
  		  NnetLstmUpdateOptions::Register(po);

	      //lm
  		  po->Register("class-boundary", &class_boundary, "The fist index of each class(and final class class) in class based language model");
  		  po->Register("num-class", &num_class, "The number of class that the language model output");
  		  po->Register("var-penalty", &var_penalty, "The penalty of the variance regularization approximation item");
  	  }
};


struct NnetLmStats: NnetStats {

    CBXent cbxent;
    Xent xent;

    NnetLmStats() { }

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
        else if (opts->objective_function == "cbxent") {
                        cbxent.Merge(myid, 0);
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
        else if (opts->objective_function == "cbxent") {
                KALDI_LOG << cbxent.Report();
        }
        else {
        	KALDI_ERR << "Unknown objective function code : " << opts->objective_function;
        }
    }
};

typedef struct Word_
{
	  int32  idx;
	  int32	 wordid;
	  int32  classid;
}Word;

class NnetLmUtil
{
public:
	static bool compare_classid(const Word &a, const Word &b)
	{
		return a.classid < b.classid;
	}

	static bool compare_wordid(const Word &a, const Word &b)
	{
		return a.wordid < b.wordid;
	}

	void SortUpdateClass(const std::vector<int32>& update_id, std::vector<int32>& sorted_id,
				std::vector<int32>& sortedclass_id, std::vector<int32>& sortedclass_id_index, std::vector<int32>& sortedclass_id_reindex,
					const Vector<BaseFloat>& frame_mask, Vector<BaseFloat>& sorted_frame_mask, const std::vector<int32> &word2class);

	void SortUpdateWord(const Vector<BaseFloat>& update_id,
				std::vector<int32>& sortedword_id, std::vector<int32>& sortedword_id_index);

	void SetClassBoundary(const Vector<BaseFloat>& classinfo,
			std::vector<int32> &class_boundary, std::vector<int32> &word2class);

};


void NnetLstmLmUpdateParallel(const NnetLstmLmUpdateOptions *opts,
		std::string	model_filename,
		std::string feature_rspecifier,
		Nnet *nnet,
		NnetLmStats *stats);


} // namespace nnet0
} // namespace kaldi

#endif // KALDI_NNET_NNET_COMPUTE_LSTM_H_
