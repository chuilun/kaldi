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

    int32 w_smooth = 10;
    po.Register("smooth-window", &w_smooth, "Smooth the posteriors over a fixed time window of size smooth-window.");
    int32 w_max = 40;
    po.Register("sliding-window", &w_max, "The confidence score is computed within a sliding window of size sliding-window.");


    std::string keywords_str;
    po.Register("keywords-id", &keywords_str, "keywords index in network output(e.g. 348|363:369|328|355:349.");

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

    Matrix<BaseFloat> post_smooth, confidence;

    Timer time;
    double time_now = 0;
    int32 num_done = 0;

    // iterate over all feature files
    for (; !output_reader.Done(); output_reader.Next()) {
      // read
      const Matrix<BaseFloat> &mat = output_reader.Value();
      std::string utt = output_reader.Key();
      KALDI_VLOG(2) << "Processing utterance " << num_done+1 
                    << ", " << utt
                    << ", " << mat.NumRows() << "frm";

      // kws confidence
      int rows = mat.NumRows();
      //int cols = mat.NumCols();
      int cols = kws_str.size()+1;
      post_smooth.Resize(rows, cols);
      confidence.Resize(rows, 2*cols);
      int hs, hm;
      float sum, max, maxid, mul;

      // posterior smoothing
      for (int j = 0; j < rows; j++) {
		  for (int i = 1; i < cols; i++) {
			  hs = j-w_smooth+1 > 0 ? j-w_smooth+1 : 0;
			  sum = 0;
			  for (int k = hs; k <= j; k++) {
				  for (int m = 0; m < keywords[i-1].size(); m++)
					  sum += mat(k, keywords[i-1][m]);
			  }
			  post_smooth(j, i) = sum/(j-hs+1);
		  }
      }

      // compute confidence score
      // confidence.Set(1.0);
      for (int j = 0; j < rows; j++) {
          mul = 1.0;
		  for (int i = 1; i < cols; i++) { // 1,2,...,n-1 keywords
			  hm = j-w_max+1 > 0 ? j-w_max+1 : 0;
			  max = 0;
			  maxid = hm;
			  for (int k = hm; k <= j; k++) {
				  if (max < post_smooth(k, i)) {
					   max = post_smooth(k, i);
					   maxid = k;
				  }
			  }
			  confidence(j,2*i) = max;
			  confidence(j,2*i+1) = maxid;
			  mul *= max;
		  }
    	  	  confidence(j,0) = pow(mul, 1.0/(cols-1));
    	  	  confidence(j,1) = j;
      }

      smooth_writer.Write(output_reader.Key(), post_smooth);
      confidence_writer.Write(output_reader.Key(), confidence);

      // progress log
      if (num_done % 100 == 0) {
        time_now = time.Elapsed();
        KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                      << time_now/60 << " min; processed " << tot_t/time_now
                      << " frames per second.";
      }
      num_done++;
      tot_t += mat.NumRows();
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
