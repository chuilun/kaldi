// nnet0/nnet-compute-forward.h

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

#ifndef NNET_NNET_COMPUTE_FORWARD_H_
#define NNET_NNET_COMPUTE_FORWARD_H_

#include "nnet-trnopts.h"
#include "nnet0/nnet-pdf-prior.h"
#include "nnet0/nnet-nnet.h"
#include "base/timer.h"


namespace kaldi {
namespace nnet0 {

struct NnetForwardOptions {
    std::string feature_transform;
    bool no_softmax;
    bool apply_log;
    bool copy_posterior;
    std::string use_gpu;
    int32 num_threads;

    int32 time_shift;
    int32 batch_size;
    int32 num_stream;
    int32 dump_interval;
    int32 skip_frames;
    int32 sweep_time;
    std::string sweep_frames_str;
    bool  sweep_loop;
    bool  skip_inner;
    float blank_posterior_scale;

    const PdfPriorOptions *prior_opts;

    NnetForwardOptions(const PdfPriorOptions *prior_opts)
    	:feature_transform(""),no_softmax(false),apply_log(false),copy_posterior(true),use_gpu("no"),num_threads(1),
		 	 	 	 	 	 	 time_shift(0),batch_size(20),num_stream(0),dump_interval(0), 
                                 skip_frames(1), sweep_time(1), sweep_frames_str("0"), sweep_loop(false), skip_inner(false),
								 blank_posterior_scale(-1.0), prior_opts(prior_opts)
    {

    }

    void Register(OptionsItf *po)
    {
    	po->Register("feature-transform", &feature_transform, "Feature transform in front of main network (in nnet format)");
    	po->Register("no-softmax", &no_softmax, "No softmax on MLP output (or remove it if found), the pre-softmax activations will be used as log-likelihoods, log-priors will be subtracted");
    	po->Register("apply-log", &apply_log, "Transform MLP output to logscale");
    	po->Register("copy-posterior", &copy_posterior, "Copy posterior for skip frames output");
    	po->Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");


    	po->Register("num-threads", &num_threads, "Number of threads(GPUs) to use");


    	po->Register("time-shift", &time_shift, "LSTM : repeat last input frame N-times, discrad N initial output frames.");
        po->Register("batch-size", &batch_size, "---LSTM--- BPTT batch size");
        po->Register("num-stream", &num_stream, "---LSTM--- BPTT multi-stream training");
        po->Register("dump-interval", &dump_interval, "---LSTM--- num utts between model dumping [ 0 == disabled ]");
        po->Register("skip-frames", &skip_frames, "LSTM model skip frames for next input");
        po->Register("sweep-loop", &sweep_loop, "Sweep all frames indexes for each utterance in skip frames training if true, "
        	    		  "e.g. utt1:frame1, utt1:frame2, utt1:frame3 ...; otherwise sweep one frames index, e.g. utt1:frame1, utt2:frame2, utt3:frame3 ...");
        po->Register("skip-inner", &skip_inner, "Skip frame in neural network inner or input");
        po->Register("sweep-time", &sweep_time, "Sweep times for each utterance in skip frames training(Deprecated, use --sweep-frames instead)");
        po->Register("sweep-frames", &sweep_frames_str, "Sweep frames index for each utterance in skip frames decoding, e.g. 0");
        po->Register("blank-posterior-scale", &blank_posterior_scale, "For CTC decoding, scale blank label posterior by a constant value(e.g. 0.11), other label posteriors are directly used in decoding.");
    }

};

struct NnetForwardStats {

	int32 num_done;

	kaldi::int64 total_frames;

	NnetForwardStats() { std::memset(this, 0, sizeof(*this)); }

	void Print(double time_now)
	{
	    // final message
	    KALDI_LOG << "Done " << num_done << " files"
	              << " in " << time_now/60 << "min,"
	              << " (fps " << total_frames/time_now << ")";
	}
};

void NnetForwardParallel(const NnetForwardOptions *opts,
						std::string	model_filename,
						std::string feature_rspecifier,
						std::string sweep_frames_rspecifier,
						std::string feature_wspecifier,
						NnetForwardStats *stats);


} // namespace nnet0
} // namespace kaldi
#endif /* NNET_NNET_COMPUTE_FORWARD_H_ */
