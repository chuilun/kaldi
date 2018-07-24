// nnetbin/nnet-kws-confidence.cc

// Copyright 2016-2017   Shanghai Jiao Tong University (author: Wei Deng)

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
        "Read neural network output to get key words confidence score.\n"
        "\n"
        "Usage:  nnet-kws-confidence [options] <feature_rspecifier> <smooth_wspecifier> <confidence_wspecifier>\n"
        "e.g.: \n"
        " nnet-kws-confidence --keywords-id=\"218:115:348:369\" --smooth-window=7 --sliding-window=35 ark:mlpoutput.ark ark:smooth.ark ark:confidence.ark\n";

    ParseOptions po(usage);

    using namespace kaldi;
    using namespace kaldi::nnet0;
    typedef kaldi::int32 int32;

    int32 w_smooth = 7;
    po.Register("smooth-window", &w_smooth, "Smooth the posteriors over a fixed time window of size smooth-window.");
    int32 w_max = 30;
    po.Register("sliding-window", &w_max, "The confidence score is computed within a sliding window of size sliding-window.");


    std::string keywords_str;
    po.Register("keywords-id", &keywords_str, "keywords index in network output(e.g. 348|363:369|328|355:349.");

    int32 word_interval = 30;
	po.Register("word-interval", &word_interval, "Word interval between each keyword");

	BaseFloat wakeup_threshold = 0.5;
	po.Register("wakeup-threshold", &wakeup_threshold, "Greater or equal this threshold will be wakeup.");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string output_rspecifier = po.GetArg(1),
    	smooth_wspecifier = po.GetArg(2),
		confidence_wspecifier = po.GetArg(3);

    //keywords id list
    std::vector<std::string> kws_str;
    std::vector<std::vector<int32> > keywords;
    kaldi::SplitStringToVector(keywords_str, "|", false, &kws_str);
    keywords.resize(kws_str.size());
    for(int i = 0; i < kws_str.size(); i++) {
    		if (!kaldi::SplitStringToIntegers(kws_str[i], ":", false, &keywords[i]))
    	    		KALDI_ERR << "Invalid keywords id string " << kws_str[i];
    }

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader output_reader(output_rspecifier);
    BaseFloatMatrixWriter smooth_writer(smooth_wspecifier);
    BaseFloatMatrixWriter confidence_writer(confidence_wspecifier);
    //BaseFloatVectorWriter confidence_writer(confidence_wspecifier);

    Matrix<BaseFloat> post_smooth, confidence, buffer;

    Timer time;
    double time_now = 0;
    int32 num_done = 0;
    float score;
    int wakeup_frame, iswakeup;

    // iterate over all feature files
    for (; !output_reader.Done(); output_reader.Next()) {
		// read
		const Matrix<BaseFloat> &posterior = output_reader.Value();
		std::string utt = output_reader.Key();
		KALDI_VLOG(2) << "Processing utterance " << num_done+1
					<< ", " << utt
					<< ", " << posterior.NumRows() << "frm";

		// kws confidence
		int rows = posterior.NumRows();
		//int cols = mat.NumCols();
		int cols = kws_str.size()+1;
		post_smooth.Resize(rows, cols);
		confidence.Resize(rows, 2*cols);
		buffer.Resize(w_max+1, keywords.size()*2+1);

		int hs, hm, pre_t, cur_t;
		float sum, mul, sscore, pre_score;

		// posterior smoothing
		for (int j = 0; j < rows; j++) {
		  for (int i = 1; i < cols; i++) {
			  hs = j-w_smooth+1 > 0 ? j-w_smooth+1 : 0;
			  sum = 0;
			  for (int k = hs; k <= j; k++) {
				  for (int m = 0; m < keywords[i-1].size(); m++)
					  sum += posterior(k, keywords[i-1][m]);
			  }
			  post_smooth(j, i) = sum/(j-hs+1);
		  }
		}

		// compute confidence score
		// confidence.Set(1.0);
		for (int j = 0; j < rows; j++) {
			mul = 1.0;
			hm = j-w_max+1 > 0 ? j-w_max+1 : 0;

			// the first keyword
			for (int k = 1; k <= j-hm+1; k++) {
				buffer(k, 1) = post_smooth(k+hm-1,1);
				buffer(k, 2) = k; // time stamp
				if (buffer(k, 1) < buffer(k-1, 1)) {
					buffer(k, 1) = buffer(k-1, 1);
					buffer(k, 2) = buffer(k-1, 2);
				}
			}

			// 2,...,n keywords
			for (int i = 2; i < cols; i++) {
				for (int k = i; k <= j-hm+1; k++) {
					buffer(k, 2*i-1) = buffer(k-1, 2*i-1);
					buffer(k, 2*i) = buffer(k-1, 2*i);
					sscore = buffer(k-1, 2*i-3) * post_smooth(k+hm-1, i);
					if (buffer(k, 2*i-1) < sscore) {
						buffer(k, 2*i-1) = sscore;
						buffer(k, 2*i) = k-1; //(k-1)+(hm-1);
					}
				}
			}

	        // final score
			mul = buffer(j-hm+1, 2*(cols-1)-1);
			confidence(j,0) = pow(mul, 1.0/(cols-1));
			confidence(j,1) = j; // time stamp

			// back tracking
			cur_t = j-hm+1;
			for (int i = cols-1; i > 1; i--) {
				pre_t = buffer(cur_t, 2*i);
				pre_score = buffer(pre_t, 2*(i-1)-1);

				// the nth keyword score
				confidence(j,2*i) = mul/pre_score;

				// the nth keyword time stamp
				confidence(j,2*i+1) = (pre_t+1)+(hm-1);

				mul = pre_score;
				cur_t = pre_t;
			}

			confidence(j,2) = buffer(cur_t, 1);
			confidence(j,3) = buffer(cur_t, 2)+(hm-1);
		}

		smooth_writer.Write(output_reader.Key(), post_smooth);
		confidence_writer.Write(output_reader.Key(), confidence);

		// is wakeup?
		bool flag;
		int interval;
		score = wakeup_frame = iswakeup = 0;
		for (int j = 0; j < rows; j++) {
            flag = true;
			for (int i = 2; i < cols; i++) {
				interval = confidence(j,2*i+1)-confidence(j,2*(i-1)+1);
				if (interval >= word_interval || interval <= 0) {
					flag = false;
					break;
				}
			}

			if (score < confidence(j,0)) {
				score = confidence(j,0);
				wakeup_frame = j;
			}

			if (confidence(j,0) >= wakeup_threshold && flag) {
				iswakeup = 1;
			}
		}

		// report results
		if (kaldi::g_kaldi_verbose_level >= 1) {
			std::ostringstream os;
			os << wakeup_frame << " ";
			for (int i = 0; i < cols; i++) {
				os << confidence(wakeup_frame,2*i) << " ";
			}
			os << confidence(wakeup_frame, 3) << " ";
			for (int i = 2; i < cols; i++) {
				os << confidence(wakeup_frame,2*i+1)-confidence(wakeup_frame,2*(i-1)+1) << "\t";
			}
			os << std::endl;
			KALDI_VLOG(1) << os.str();
		}

		KALDI_LOG << iswakeup << " " << std::string(utt);


		// progress log
		if (num_done % 100 == 0) {
			time_now = time.Elapsed();
			KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
						  << time_now/60 << " min; processed " << tot_t/time_now
						  << " frames per second.";
		}
		num_done++;
		tot_t += posterior.NumRows();
    }
    
    // final message
    KALDI_LOG << "Done " << num_done << " files" 
              << " in " << time.Elapsed()/60 << "min," 
              << " (fps " << tot_t/time.Elapsed() << ")"; 

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
