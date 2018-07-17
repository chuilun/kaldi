// nnetbin/nnet-forward-lfmmi.cc

// Copyright 2016-2017  AISpeech (Author: Wei Deng)

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


int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet0;
  try {
    const char *usage =
        "Perform forward pass through Neural Network in lattice free mmi.\n"
        "\n"
        "Usage:  nnet-forward-lfmmi [options] <model-in> <feature-rspecifier> <feature-wspecifier>\n"
        "e.g.: \n"
        " nnet-forward-lfmmi nnet ark:features.ark ark:mlpoutput.ark\n";

    ParseOptions po(usage);

    PdfPriorOptions prior_opts;
    prior_opts.Register(&po);

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in front of main network (in nnet format)");

    bool no_softmax = false;
    po.Register("no-softmax", &no_softmax, "No softmax on MLP output (or remove it if found), the pre-softmax activations will be used as log-likelihoods, log-priors will be subtracted");
    bool apply_log = false;
    po.Register("apply-log", &apply_log, "Transform MLP output to logscale");

    std::string use_gpu="no";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 

    using namespace kaldi;
    using namespace kaldi::nnet0;
    typedef kaldi::int32 int32;

    bool copy_posterior = false;
    int32 targets_delay = 0;
    int32 frames_per_chunk = 50;
    int32 extra_left_context = 0;
    int32 skip_frames = 1;
    int32 num_stream = 1;
    std::string sweep_frames_str("0");
    bool skip_inner = false;
    po.Register("targets_delay", &targets_delay, "LSTM : repeat last input frame N-times, discrad N initial output frames.");

    po.Register("extra-left-context", &extra_left_context,
                   "Number of frames of additional left-context to add on top "
                   "of the neural net's inherent left context (may be useful in "
                   "recurrent setups");

    po.Register("frames-per-chunk", &frames_per_chunk,
                   "Number of frames in each chunk that is separately evaluated "
                   "by the neural net.  Measured before any subsampling, if the "
                   "--frame-subsampling-factor options is used (i.e. counts "
                   "input frames");
    po.Register("skip-frames", &skip_frames, "LSTM model skip frames for next input");
    po.Register("num-stream", &num_stream, "---LSTM--- BPTT multi-stream training");
    po.Register("copy-posterior", &copy_posterior, "Copy posterior for skip frames output");
	po.Register("sweep-frames", &sweep_frames_str, "Sweep frames index for each utterance in skip frames decoding, e.g. 0:1");
    po.Register("skip-inner", &skip_inner, "Skip frame in neural network inner or input");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        feature_wspecifier = po.GetArg(3);
        
    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);
    // optionally remove softmax,
    Component::ComponentType last_type = nnet.GetComponent(nnet.NumComponents()-1).GetType();
    if (no_softmax) {
      if (last_type == Component::kSoftmax || last_type == Component::kBlockSoftmax) {
        KALDI_LOG << "Removing " << Component::TypeToMarker(last_type) << " from the nnet " << model_filename;
        nnet.RemoveComponent(nnet.NumComponents()-1);
      } else {
        KALDI_WARN << "Cannot remove softmax using --no-softmax=true, as the last component is " << Component::TypeToMarker(last_type);
      }
    }

    std::vector<int32> sweep_frames;
	if (!kaldi::SplitStringToIntegers(sweep_frames_str, ":", false, &sweep_frames))
		KALDI_ERR << "Invalid sweep-frames string " << sweep_frames_str;

	if (sweep_frames[0] > skip_frames || sweep_frames.size() > 1)
		KALDI_ERR << "invalid sweep frame index";

    // avoid some bad option combinations,
    if (apply_log && no_softmax) {
      KALDI_ERR << "Cannot use both --apply-log=true --no-softmax=true, use only one of the two!";
    }

    // we will subtract log-priors later,
    PdfPrior pdf_prior(prior_opts); 

    // disable dropout,
    nnet_transf.SetDropoutRetention(1.0);
    nnet.SetDropoutRetention(1.0);

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    BaseFloatMatrixWriter feature_writer(feature_wspecifier);

    int in_skip = skip_inner ? 1 : skip_frames,
        out_skip = skip_inner ? skip_frames : 1;
	int ctx_left = extra_left_context/skip_frames,
		chunk = frames_per_chunk/skip_frames,
		his_left = ctx_left + targets_delay;
    int out_frames = (ctx_left + chunk + targets_delay)*num_stream;
    int in_frames = out_frames*out_skip;

    CuMatrix<BaseFloat>  feats_transf, nnet_out;

    std::vector<std::string> keys(num_stream);
    std::vector<Matrix<BaseFloat> > feats(num_stream);
    std::vector<Posterior> targets(num_stream);
    std::vector<int> curt(num_stream, 0);
    std::vector<int> lent(num_stream, 0);
    std::vector<int> frame_num_utt(num_stream, 0);
    std::vector<int> new_utt_flags(num_stream, 1);

    std::vector<Matrix<BaseFloat> > utt_feats(num_stream);
    std::vector<int> utt_curt(num_stream, 0);
    std::vector<bool> utt_copied(num_stream, false);

    // bptt batch buffer
    int32 feat_dim = nnet.InputDim();
    int32 out_dim = nnet.OutputDim();
    Matrix<BaseFloat> feat, nnet_out_host;
    if (out_frames > 0) {
    	feat.Resize(in_frames, feat_dim, kSetZero);
    	nnet_out_host.Resize(out_frames, out_dim, kSetZero);
    }


    kaldi::int64 total_frames = 0;
    int32 num_done = 0, num_frames;


    Timer time;
    double time_now = 0;

    while (true)
	{
		 // loop over all streams, check if any stream reaches the end of its utterance,
		 // if any, feed the exhausted stream with a new utterance, update book-keeping infos
		for (int s = 0; s < num_stream; s++)
		{
			// this stream still has valid frames
			if (curt[s] < lent[s]) {
				//new_utt_flags[s] = 0;
				continue;
			}

			if (utt_curt[s] > 0 && !utt_copied[s])
			{
				feature_writer.Write(keys[s], utt_feats[s]);
				utt_copied[s] = true;
			}

			while (!feature_reader.Done())
			{
				const std::string key = feature_reader.Key();
				const Matrix<BaseFloat> &mat = feature_reader.Value();
				// forward the features through a feature-transform,
				nnet_transf.Feedforward(CuMatrix<BaseFloat>(mat), &feats_transf);

				num_done++;

				// checks ok, put the data in the buffers,
				keys[s] = key;
				feats[s].Resize(feats_transf.NumRows(), feats_transf.NumCols());
				feats_transf.CopyToMat(&feats[s]);
				//feats[s] = mat;
				curt[s] = sweep_frames[0];
				lent[s] = feats[s].NumRows();
				new_utt_flags[s] = 1;  // a new utterance feeded to this stream

				frame_num_utt[s] = lent[s]/skip_frames;
				frame_num_utt[s] += lent[s]%skip_frames > sweep_frames[0] ? 1 : 0;
				lent[s] = lent[s] > frame_num_utt[s]*skip_frames ? frame_num_utt[s]*skip_frames : lent[s];
				int32 utt_frames = copy_posterior ? lent[s] : frame_num_utt[s];
				utt_feats[s].Resize(utt_frames, out_dim, kUndefined);
				utt_copied[s] = false;
				utt_curt[s] = 0;

                feature_reader.Next();
				break;
			}
		}

		// we are done if all streams are exhausted
		int done = 1;
		for (int s = 0; s < num_stream; s++) {
			if (curt[s] < lent[s]) done = 0;  // this stream still contains valid data, not exhausted
		}

		if (done) break;

		int len = (ctx_left + chunk + targets_delay)*out_skip, utt_idx;
		for (int t = 0; t < len; t++) {
		   for (int s = 0; s < num_stream; s++) {
			   utt_idx = curt[s] + (t-ctx_left)*in_skip;
			   if (utt_idx < 0) utt_idx = 0;
			   if (utt_idx > lent[s]-1) utt_idx = lent[s]-1;
			   feat.Row(t*num_stream+s).CopyFromVec(feats[s].Row(utt_idx));
		   }
		}
		// increase each utterance current index
		for (int s = 0; s < num_stream; s++) {
			curt[s] += frames_per_chunk;
			if (curt[s] > lent[s]) curt[s] = lent[s];
		}

		num_frames = feat.NumRows();
		// report the speed
		if (num_done % 5000 == 0) {
		  time_now = time.Elapsed();
		  KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
						<< time_now/60 << " min; processed " << total_frames/time_now
						<< " frames per second.";
		}

		// apply optional feature transform
		//nnet_transf.Feedforward(CuMatrix<BaseFloat>(feat), &feats_transf);

		// for streams with new utterance, history states need to be reset
		nnet.ResetLstmStreams(new_utt_flags);

		// forward pass
		//nnet.Propagate(feats_transf, &nnet_out);
		nnet.Propagate(CuMatrix<BaseFloat>(feat), &nnet_out);

		// convert posteriors to log-posteriors,
		if (apply_log) {
		  nnet_out.Add(1e-20); // avoid log(0),
		  nnet_out.ApplyLog();
		}

		// subtract log-priors from log-posteriors or pre-softmax,
		if (prior_opts.class_frame_counts != "") {
		  pdf_prior.SubtractOnLogpost(&nnet_out);
		}

		nnet_out.CopyToMat(&nnet_out_host);

		for (int t = 0; t < chunk; t++) {
		   for (int s = 0; s < num_stream; s++) {
			   // feat shifting & padding
			   if (copy_posterior) {
				   for (int k = 0; k < skip_frames; k++) {
						if (utt_curt[s] < lent[s]) {
							utt_feats[s].Row(utt_curt[s]).CopyFromVec(nnet_out_host.Row((t+his_left)*num_stream+s));
							utt_curt[s]++;
						}
				   }
			   }
			   else {
				   if (utt_curt[s] < frame_num_utt[s]) {
					   utt_feats[s].Row(utt_curt[s]).CopyFromVec(nnet_out_host.Row((t+his_left)*num_stream+s));
					   utt_curt[s]++;
				   }
			   }
		   }
		}

       total_frames += num_frames;
	}
    
    // final message
    KALDI_LOG << "Done " << num_done << " files" 
              << " in " << time.Elapsed()/60 << "min," 
              << " (fps " << total_frames/time.Elapsed() << ")";

#if HAVE_CUDA==1
    if (kaldi::g_kaldi_verbose_level >= 1) {
      CuDevice::Instantiate().PrintProfile();
    }
#endif

    if (num_done == 0) return -1;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
