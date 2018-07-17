// nnetbin/nnet-forward-parallel.cc

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

#include <limits>

#include "nnet0/nnet-nnet.h"
#include "nnet0/nnet-loss.h"
#include "nnet0/nnet-pdf-prior.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"

#include "nnet0/nnet-compute-forward.h"

int main(int argc, char *argv[]) {
	  using namespace kaldi;
	  using namespace kaldi::nnet0;
	  typedef kaldi::int32 int32;

  try {
    const char *usage =
        "Perform forward pass through Neural Network in Parallel.\n"
        "\n"
        "Usage:  nnet-forward-parallel [options] <model-in> <feature-rspecifier> <sweep_frames_rspecifier>(optional) <feature-wspecifier>\n"
        "e.g.: \n"
        " nnet-forward-parallel --num-thread=2 nnet ark:features.ark scp:sweep.scp(optional) ark:mlpoutput.ark\n";

    ParseOptions po(usage);

    PdfPriorOptions prior_opts;
    prior_opts.Register(&po);

    NnetForwardOptions opts(&prior_opts);
    opts.Register(&po);

    std::string model_filename,
            feature_rspecifier,
            feature_wspecifier,
			sweep_frames_rspecifier = "";


    po.Read(argc, argv);

    if (po.NumArgs() == 3) {
    		model_filename = po.GetArg(1);
    		feature_rspecifier = po.GetArg(2);
    		feature_wspecifier = po.GetArg(3);
    }
    else if (po.NumArgs() == 4) {
		model_filename = po.GetArg(1);
		feature_rspecifier = po.GetArg(2);
		sweep_frames_rspecifier = po.GetArg(3);
		feature_wspecifier = po.GetArg(4);
	}else {
    		po.PrintUsage();
    		exit(1);
    }


        
    //Select the GPU
#if HAVE_CUDA==1
    if (opts.use_gpu == "yes")
        CuDevice::Instantiate().Initialize();
    //CuDevice::Instantiate().DisableCaching();
#endif


    NnetForwardStats stats;

    Timer time;
    double time_now = 0;

    KALDI_LOG << "Nnet Forward STARTED";

    NnetForwardParallel(&opts, model_filename,
    					feature_rspecifier, sweep_frames_rspecifier, feature_wspecifier, &stats);

    KALDI_LOG << "Nnet Forward FINISHED; ";

    time_now = time.Elapsed();

    stats.Print(time_now);

#if HAVE_CUDA==1
    if (kaldi::g_kaldi_verbose_level >= 1) {
      CuDevice::Instantiate().PrintProfile();
    }
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
