// nnetbin/nnet-train-sequential-parallel.cc

// Copyright 2014-2015  Shanghai Jiao Tong University (author: Wei Deng)

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


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/faster-decoder.h"
#include "decoder/decodable-matrix.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"

#include "nnet0/nnet-trnopts.h"
#include "nnet0/nnet-component.h"
#include "nnet0/nnet-activation.h"
#include "nnet0/nnet-nnet.h"
#include "nnet0/nnet-pdf-prior.h"
#include "nnet0/nnet-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"

#include <iomanip>

#include "nnet0/nnet-compute-sequential-parallel.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet0;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Perform one iteration of DNN-MMI training by stochastic "
        "gradient descent.\n"
        "The network weights are updated on each utterance.\n"
        "Usage:  nnet-compute-sequential-parallel [options] <model-in> <transition-model-in>(optional) "
        "<feature-rspecifier> <sweep_frames_rspecifier>(optional) <den-lat-rspecifier> <ali-rspecifier> [<model-out>]\n"
        "e.g.: \n"
        " nnet-compute-sequential-parallel nnet.init trans.mdl(optional) scp:train.scp scp:sweep.scp(optional) scp:denlats.scp ark:train.ali "
        "nnet.iter1\n";

    ParseOptions po(usage);

    NnetTrainOptions trn_opts; trn_opts.learn_rate=0.00001;
    trn_opts.Register(&po);

    bool binary = true;
    po.Register("binary", &binary, "Write output in binary mode");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in Nnet format");

    PdfPriorOptions prior_opts;
    prior_opts.Register(&po);

    NnetParallelOptions parallel_opts;
    parallel_opts.Register(&po);

    NnetSequentialUpdateOptions opts(&trn_opts, &prior_opts, &parallel_opts);
    opts.Register(&po);

    po.Read(argc, argv);


    std::string model_filename, transition_model_filename,
				feature_rspecifier, den_lat_rspecifier,
				num_ali_rspecifier, sweep_frames_rspecifier, target_model_filename;
                transition_model_filename = "", sweep_frames_rspecifier = "";

    if (po.NumArgs() == 6)
    {
		model_filename = po.GetArg(1),
		transition_model_filename = po.GetArg(2),
		feature_rspecifier = po.GetArg(3),
		den_lat_rspecifier = po.GetArg(4),
		num_ali_rspecifier = po.GetArg(5);
		target_model_filename = po.GetArg(6);
    }
    else if (po.NumArgs() == 7)
    {
    		model_filename = po.GetArg(1),
    		transition_model_filename = po.GetArg(2),
    		feature_rspecifier = po.GetArg(3),
    		sweep_frames_rspecifier = po.GetArg(4);
    		den_lat_rspecifier = po.GetArg(5),
    		num_ali_rspecifier = po.GetArg(6);
    		target_model_filename = po.GetArg(7);
    }
    else if (po.NumArgs() == 5)
    {
		model_filename = po.GetArg(1),
		feature_rspecifier = po.GetArg(2),
		den_lat_rspecifier = po.GetArg(3),
		num_ali_rspecifier = po.GetArg(4);
		target_model_filename = po.GetArg(5);
    }
    else
    {
        po.PrintUsage();
        exit(1);
    }

    using namespace kaldi;
    using namespace kaldi::nnet0;
    typedef kaldi::int32 int32;


    // Initialize GPU
    #if HAVE_CUDA == 1
        CuDevice::Instantiate().Initialize();
        //CuDevice::Instantiate().DisableCaching();
    #endif

    Nnet nnet;

    NnetSequentialStats stats;

    Timer time;
    double time_now = 0;
    KALDI_LOG << "TRAINING STARTED";

    NnetSequentialUpdateParallel(&opts,
								feature_transform,
								model_filename,
								transition_model_filename,
								feature_rspecifier,
								den_lat_rspecifier,
								num_ali_rspecifier,
								sweep_frames_rspecifier,
								&nnet,
								&stats);

    //add back the softmax
    KALDI_LOG << "Appending the softmax " << target_model_filename;
    nnet.AppendComponent(new Softmax(nnet.OutputDim(),nnet.OutputDim()));
    //store the nnet
    nnet.Write(target_model_filename, binary);

    KALDI_LOG << "TRAINING FINISHED; ";
    time_now = time.Elapsed();
    stats.Print(opts.criterion, time_now);

#if HAVE_CUDA == 1
    CuDevice::Instantiate().PrintProfile();
#endif

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



