// nnetbin/nnet-lstm-sentence-ppl.cc

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
#include "nnet0/nnet-compute-lstm-lm-parallel.h"
#include "nnet0/nnet-activation.h"
#include "nnet0/nnet-class-affine-transform.h"
#include "nnet0/nnet-word-vector-transform.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet0;
  try {
    const char *usage =
        "Compute log target posterior pass through Neural Network.\n"
        "\n"
        "Usage:  nnet-lstm-sentence-ppl [options] <model-in> <feature-rspecifier> <feature-wspecifier>\n"
        "e.g.: \n"
        " nnet-lstm-sentence-ppl --num-stream=40 --batch-size=15 --class-boundary=data/lang/class_boundary.txt nnet ark:features.ark ark,t:mlpoutput.txt\n";

    ParseOptions po(usage);

    std::string classboundary_file = "";
    po.Register("class-boundary", &classboundary_file, "The fist index of each class(and final class class) in class based language model");

    std::string use_gpu="no";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 

    using namespace kaldi;
    using namespace kaldi::nnet0;
    typedef kaldi::int32 int32;

    int32 time_shift = 0;
    int32 batch_size = 15;
    int32 num_stream = 1;
    po.Register("time-shift", &time_shift, "LSTM : repeat last input frame N-times, discrad N initial output frames."); 
    po.Register("batch-size", &batch_size, "---LSTM--- BPTT batch size");
    po.Register("num-stream", &num_stream, "---LSTM--- BPTT multi-stream training");

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

    Nnet nnet;
    nnet.Read(model_filename);

    NnetLmUtil util;
    ClassAffineTransform *class_affine = NULL;
    //WordVectorTransform *word_transf = NULL;
    CBSoftmax *cb_softmax = NULL;
    for (int32 c = 0; c < nnet.NumComponents(); c++)
    {
    	if (nnet.GetComponent(c).GetType() == Component::kClassAffineTransform)
    		class_affine = &(dynamic_cast<ClassAffineTransform&>(nnet.GetComponent(c)));
    	/*else if (nnet.GetComponent(c).GetType() == Component::kWordVectorTransform)
    		word_transf = &(dynamic_cast<WordVectorTransform&>(nnet.GetComponent(c))); */
    	else if (nnet.GetComponent(c).GetType() == Component::kCBSoftmax)
    		cb_softmax = &(dynamic_cast<CBSoftmax&>(nnet.GetComponent(c)));
    }

    std::vector<int32> class_boundary, word2class;
    if (classboundary_file != "")
    {
	    Input in;
	    Vector<BaseFloat> classinfo;
	    in.OpenTextMode(classboundary_file);
	    classinfo.Read(in.Stream(), false);
	    in.Close();
	    util.SetClassBoundary(classinfo, class_boundary, word2class);
    }

    CBXent cbxent;
    Xent xent;

    if (NULL != class_affine)
    {
	    class_affine->SetClassBoundary(class_boundary);
        cb_softmax->SetClassBoundary(class_boundary);
        cbxent.SetClassBoundary(class_boundary);
    }

    // disable dropout,
    nnet.SetDropoutRetention(1.0);

    SequentialInt32VectorReader feature_reader(feature_rspecifier);
    BaseFloatVectorWriter feature_writer(feature_wspecifier);

    kaldi::int64 total_frames = 0;
    int32 num_done = 0, num_frames, num_pdf;
    num_pdf = nnet.OutputDim();
    num_frames = batch_size * num_stream;

    Matrix<BaseFloat> nnet_out_host;

    		std::vector<std::string> keys(num_stream);
    		std::vector<std::vector<int32> > feats(num_stream);
    	    std::vector<int> curt(num_stream, 0);
    	    std::vector<int> lent(num_stream, 0);
    	    std::vector<int> frame_num_utt(num_stream, 0);
    	    std::vector<int> new_utt_flags(num_stream, 0);

    	    std::vector<Vector<BaseFloat> > utt_feats(num_stream);
    	    std::vector<int> utt_curt(num_stream, 0);
    	    std::vector<bool> utt_copied(num_stream, 0);

    	    // bptt batch buffer
    	    Vector<BaseFloat> frame_mask(batch_size * num_stream, kSetZero);
    	    Vector<BaseFloat> feat(batch_size * num_stream, kSetZero);
            Matrix<BaseFloat> featmat(batch_size * num_stream, 1, kSetZero);
            CuMatrix<BaseFloat> words(batch_size * num_stream, 1, kSetZero);

    	    Vector<BaseFloat> sorted_frame_mask(batch_size * num_stream, kSetZero);
    	    std::vector<int32> target(batch_size * num_stream, kSetZero);
    	    std::vector<int32> sorted_target(batch_size * num_stream, kSetZero);
    	    std::vector<int32> sortedclass_target(batch_size * num_stream, kSetZero);
    	    std::vector<int32> sortedclass_target_index(batch_size * num_stream, kSetZero);
    	    std::vector<int32> sortedclass_target_reindex(batch_size * num_stream, kSetZero);

            Vector<BaseFloat> log_post_tgt_sorted(batch_size * num_stream, kSetZero);
            Vector<BaseFloat> log_post_tgt(batch_size * num_stream, kSetZero);
            CuMatrix<BaseFloat> nnet_out(num_frames, num_pdf, kSetZero), nnet_diff;


    	    Timer time;
    	    double time_now = 0;

    	    //num_stream=1 for lstm debug
    	    if (num_stream >= 1)
    	    while (1)
    	    {
    	    	 // loop over all streams, check if any stream reaches the end of its utterance,
    	    	 // if any, feed the exhausted stream with a new utterance, update book-keeping infos
    	    	for (int s = 0; s < num_stream; s++)
    	    	{
    	    		// this stream still has valid frames
    	    		if (curt[s] < lent[s]) {
    	    			new_utt_flags[s] = 0;
    	    		    continue;
    	    		}

    	    		if (utt_curt[s] > 0 && !utt_copied[s])
    	    		{
    	    			feature_writer.Write(keys[s], utt_feats[s]);
    	    			utt_copied[s] = true;
    	    		}

    	    		if(!feature_reader.Done())
    	    		{
    	    			keys[s] = feature_reader.Key();
    	    	    	feats[s] = feature_reader.Value();

    	    	    	num_done++;

    	                curt[s] = 0;
    	                lent[s] = feats[s].size() - 1;
    	                new_utt_flags[s] = 1;  // a new utterance feeded to this stream

    	                utt_feats[s].Resize(lent[s], kUndefined);
    	                utt_copied[s] = false;
    	                utt_curt[s] = 0;

                        feature_reader.Next();
    	    		}
    	    	}

    		        // we are done if all streams are exhausted
    		        int done = 1;
    		        for (int s = 0; s < num_stream; s++) {
    		            if (curt[s] < lent[s]) done = 0;  // this stream still contains valid data, not exhausted
    		        }

    		        if (done) break;

    		        // fill a multi-stream bptt batch
    		        // * frame_mask: 0 indicates padded frames, 1 indicates valid frames
    		        // * target: padded to batch_size
    		        // * feat: first shifted to achieve targets delay; then padded to batch_size
    		        for (int t = 0; t < batch_size; t++) {
    		            for (int s = 0; s < num_stream; s++) {
    		                // frame_mask & targets padding
    		                if (curt[s] < time_shift) {
    		                	frame_mask(t * num_stream + s) = 0;
    		                	target[t * num_stream + s] = feats[s][0];
    		                }
    		                else if (curt[s] < lent[s] + time_shift) {
    		                    frame_mask(t * num_stream + s) = 1;
    		                    target[t * num_stream + s] = feats[s][curt[s]-time_shift+1];
    		                } else {
    		                    frame_mask(t * num_stream + s) = 0;
    		                    target[t * num_stream + s] = 0; //feats[s][lent[s]-1];
    		                }
    		                // feat shifting & padding
    		                if (curt[s] < lent[s]) {
    		                    feat(t * num_stream + s) = feats[s][curt[s]];
    		                } else {
    		                    feat(t * num_stream + s) = 0; //feats[s][lent[s]-1];

    		                }

    		                curt[s]++;
    		            }
    		        }

    		        num_frames = feat.Dim();
    			    // report the speed
    			    if (num_done % 5000 == 0) {
    			      time_now = time.Elapsed();
    			      KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
    			                    << time_now/60 << " min; processed " << total_frames/time_now
    								<< " frames per second.";
    			    }

    			   // for streams with new utterance, history states need to be reset
    			   nnet.ResetLstmStreams(new_utt_flags);

    		        if (NULL != class_affine)
    		        {
    			        // sort output class id
    		        	util.SortUpdateClass(target, sorted_target, sortedclass_target,
    			        		sortedclass_target_index, sortedclass_target_reindex, frame_mask, sorted_frame_mask, word2class);
    			        class_affine->SetUpdateClassId(sortedclass_target, sortedclass_target_index, sortedclass_target_reindex);
    			        cb_softmax->SetUpdateClassId(sortedclass_target);
    		        }


    		        // forward pass
    		        featmat.CopyColFromVec(feat, 0);
    	            words.CopyFromMat(featmat);

    		        nnet.Propagate(words, &nnet_out);

    		        // evaluate objective function we've chosen
    		        if (NULL != class_affine) {
    		        	cbxent.Eval(sorted_frame_mask, nnet_out, sorted_target, &nnet_diff);
    		        	cbxent.GetTargetWordPosterior(log_post_tgt_sorted);
    	   		    	for (int i = 0; i < num_frames; i++)
    	   		    		log_post_tgt(i) = log_post_tgt_sorted(sortedclass_target_reindex[i]);
    		        } else {
    		        	xent.Eval(frame_mask, nnet_out, target, &nnet_diff);
    		        	xent.GetTargetWordPosterior(log_post_tgt);
    		        }

    		       for (int t = 0; t < batch_size; t++) {
    		           for (int s = 0; s < num_stream; s++) {
						   if (utt_curt[s] < lent[s]) {
							   utt_feats[s](utt_curt[s]) = log_post_tgt(t * num_stream + s);
							   utt_curt[s]++;
						   }
    		           }
    		       }

    		       total_frames += num_frames;
    	    }

    // final message
    KALDI_LOG << "Done " << num_done << " files"
              << " in " << time.Elapsed()/60 << "min,"
              << " (fps " << total_frames/time.Elapsed() << ")";

    if (NULL != class_affine)
    	KALDI_LOG << cbxent.Report();
    else
    	KALDI_LOG << xent.Report();

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
