// online0/online-nnet-forward.h

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

#ifndef ONLINE0_ONLINE_NNET_FORWARD_H_
#define ONLINE0_ONLINE_NNET_FORWARD_H_

#include "util/circular-queue.h"
#include "thread/kaldi-mutex.h"
#include "nnet0/nnet-nnet.h"
#include "nnet0/nnet-trnopts.h"
#include "nnet0/nnet-pdf-prior.h"

namespace kaldi {

struct OnlineNnetForwardOptions {
    typedef nnet0::PdfPriorOptions PdfPriorOptions;
    std::string feature_transform;
    std::string network_model;
    bool no_softmax;
    bool apply_log;
    std::string use_gpu;
    int32 gpuid;
    int32 num_threads;

    int32 batch_size;
    int32 num_stream;
    float blank_posterior_scale;

    PdfPriorOptions prior_opts;

    OnlineNnetForwardOptions()
    	:feature_transform(""),network_model(""),no_softmax(false),apply_log(false),
		 use_gpu("no"),gpuid(-1),num_threads(1),batch_size(6),num_stream(1),blank_posterior_scale(-1.0)
    {

    }

    void Register(OptionsItf *po)
    {
    	po->Register("feature-transform", &feature_transform, "Feature transform in front of main network (in nnet0 format)");
    	po->Register("network-model", &network_model, "Main neural network model (in nnet0 format)");
    	po->Register("no-softmax", &no_softmax, "No softmax on MLP output (or remove it if found), the pre-softmax activations will be used as log-likelihoods, log-priors will be subtracted");
    	po->Register("apply-log", &apply_log, "Transform MLP output to logscale");
    	po->Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");
        po->Register("gpuid", &gpuid, "gpuid < 0 for automatic select gpu, gpuid >= 0 for select specified gpu, only has effect if compiled with CUDA");
    	po->Register("num-threads", &num_threads, "Number of threads(GPUs) to use");
        po->Register("blank-posterior-scale", &blank_posterior_scale, "For CTC decoding, scale blank label posterior by a constant value(e.g. 0.11), other label posteriors are directly used in decoding.");



        //<jiayu>
        po->Register("batch-size", &batch_size, "---LSTM--- BPTT batch size");
        po->Register("num-stream", &num_stream, "---LSTM--- BPTT multi-stream training");
        //</jiayu>

        prior_opts.Register(po);
    }

};


class OnlineNnetForward {
public:
	OnlineNnetForward(const OnlineNnetForwardOptions &opts):
		opts_(opts), pdf_prior_(NULL)
	{
#if HAVE_CUDA==1
		if (opts_.use_gpu == "yes") {
			if (opts_.gpuid < 0)
				CuDevice::Instantiate().SelectGpu();
			else
				CuDevice::Instantiate().SelectPreferGpu(opts_.gpuid);
		}
#endif
        using namespace kaldi::nnet0;

		bool no_softmax = opts_.no_softmax;
		bool apply_log = opts_.apply_log;
		int32 num_stream = opts_.num_stream;
        float blank_posterior_scale = opts_.blank_posterior_scale;
		//int32 batch_size = opts_.batch_size;
		std::string feature_transform = opts_.feature_transform;
		std::string model_filename = opts_.network_model;

    		if (opts_.feature_transform != "")
    			nnet_transf_.Read(opts_.feature_transform);

    		nnet_.Read(opts_.network_model);

	    // optionally remove softmax,
	    Component::ComponentType last_type = nnet_.GetComponent(nnet_.NumComponents()-1).GetType();
	    if (no_softmax) {
	      if (last_type == Component::kSoftmax || last_type == Component::kBlockSoftmax) {
	        KALDI_LOG << "Removing " << Component::TypeToMarker(last_type) << " from the nnet " << model_filename;
	        nnet_.RemoveComponent(nnet_.NumComponents()-1);
	      } else {
	        KALDI_WARN << "Cannot remove softmax using --no-softmax=true, as the last component is " << Component::TypeToMarker(last_type);
	      }
	    }

	    // avoid some bad option combinations,
	    if (apply_log && no_softmax) {
	      KALDI_ERR << "Cannot use both --apply-log=true --no-softmax=true, use only one of the two!";
	    }

        if (blank_posterior_scale >= 0 && opts_.prior_opts.class_frame_counts != "") {
          KALDI_ERR << "Cannot use both --blank-posterior-scale --class-frame-counts, use only one of the two!";
        }

        // we will subtract log-priors later,
    		if (opts_.prior_opts.class_frame_counts != "")
	        pdf_prior_ = new PdfPrior(opts.prior_opts);

	    //int input_dim = feature_transform != "" ? nnet_transf_.InputDim() : nnet_.InputDim();
	    //int output_dim = nnet_.OutputDim();
	    //feat_.Resize(batch_size * num_stream, input_dim, kSetZero, kStrideEqualNumCols);
	    //feat_out_.Resize(batch_size * num_stream, output_dim, kSetZero, kStrideEqualNumCols);

	    new_utt_flags_.resize(num_stream, 1);
	}

    virtual ~OnlineNnetForward() {
        if (pdf_prior_ != NULL)
            delete pdf_prior_;
    }

	void Forward(const MatrixBase<BaseFloat> &in, Matrix<BaseFloat> *out) {
		CuMatrix<BaseFloat> feats_transf;
        feat_.Resize(in.NumRows(), in.NumCols(), kUndefined, kStrideEqualNumCols);
		feat_.CopyFromMat(in);
		nnet_transf_.Propagate(feat_, &feats_transf); // Feedforward
		// for streams with new utterance, history states need to be reset
		nnet_.ResetLstmStreams(new_utt_flags_);
		// forward pass
		nnet_.Propagate(feats_transf, &feat_out_);

        // ctc prior, only scale blank label posterior
        if (opts_.blank_posterior_scale >= 0) {
            feat_out_.ColRange(0, 1).Scale(opts_.blank_posterior_scale);
        }

		// convert posteriors to log-posteriors,
		if (opts_.apply_log) {
			feat_out_.Add(1e-20); // avoid log(0),
			feat_out_.ApplyLog();
		}

		// subtract log-priors from log-posteriors or pre-softmax,
		if (pdf_prior_ != NULL) {
			pdf_prior_->SubtractOnLogpost(&feat_out_);
		}

		out->Resize(feat_out_.NumRows(), feat_out_.NumCols(), kUndefined);
		out->CopyFromMat(feat_out_);
		new_utt_flags_[0] = 0;
	}

	int32 OutputDim() {
		return nnet_.OutputDim();
	}

	int32 InputDim() {
		return nnet_transf_.InputDim();
	}

	void ResetHistory() {
		new_utt_flags_[0] = 1;
	}

private:
	const OnlineNnetForwardOptions &opts_;
	kaldi::nnet0::PdfPrior *pdf_prior_;
	kaldi::nnet0::Nnet nnet_transf_;
	kaldi::nnet0::Nnet nnet_;
	CuMatrix<BaseFloat> feat_;
	CuMatrix<BaseFloat> feat_out_;
	std::vector<int> new_utt_flags_;
};

}

#endif /* ONLINE0_ONLINE_NNET_FORWARD_H_ */
