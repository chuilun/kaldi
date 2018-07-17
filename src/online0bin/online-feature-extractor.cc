// online0bin/online-feature-extractor.cc

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


#include "base/timer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/wave-reader.h"

#include "online/onlinebin-util.h"
#include "online0/online-nnet-feature-pipeline.h"

int main(int argc, char *argv[])
{
	try {

	    using namespace kaldi;
	    using namespace fst;

	    typedef kaldi::int32 int32;

	    const char *usage =
	        "Reads in wav file(s) and simulate extract online feature "
	        "(nnet0 or nnet1 setup), Note: some configuration values and inputs are\n"
	    	"set via config files whose filenames are passed as options\n"
	    	"\n"
	        "Usage: online-feature-extractor [config option] <wav-rspecifier> <feats-wspecifier>\n"
	    	"e.g.: \n"
	        "	online-feature-extractor --config=conf/online_feature.conf scp:wav.scp ark,scp:feat.ark,feat.scp \n";

	    ParseOptions po(usage);

	    /// feature pipeline config
	    OnlineNnetFeaturePipelineConfig feature_cfg;
	    feature_cfg.Register(&po);

		po.Read(argc, argv);

	    if (po.NumArgs() != 2) {
	      po.PrintUsage();
	      exit(1);
	    }

	    std::string wav_rspecifier = po.GetArg(1);
	    std::string output_wspecifier = po.GetArg(2);

		OnlineNnetFeaturePipelineOptions feature_opts(feature_cfg);
		OnlineNnetFeaturePipeline feature_pipeline(feature_opts);

		SequentialTableReader<WaveHolder> wav_reader(wav_rspecifier);
		BaseFloatMatrixWriter feat_writer(output_wspecifier);

		BaseFloat samp_freq;
		int frame_ready;

		Matrix<BaseFloat> feat;

		kaldi::int64 frame_count = 0;

		Timer timer;
		for (; !wav_reader.Done(); wav_reader.Next()) {
			std::string utt = wav_reader.Key();
			const WaveData &wave_data = wav_reader.Value();
			// get the data for channel zero (if the signal is not mono, we only
			// take the first channel).
			SubVector<BaseFloat> data(wave_data.Data(), 0);

			samp_freq = wave_data.SampFreq();

			feature_pipeline.Reset();
			feature_pipeline.AcceptWaveform(samp_freq, data);
			feature_pipeline.InputFinished();

			frame_ready = feature_pipeline.NumFramesReady();
			feat.Resize(frame_ready, feature_pipeline.Dim(), kUndefined);

			for (int i = 0; i < frame_ready; i++) {
				feature_pipeline.GetFrame(i, &feat.Row(i));
			}

			feat_writer.Write(utt, feat);
			frame_count += frame_ready;
		}

		double elapsed = timer.Elapsed();
		KALDI_LOG << "Time taken [excluding initialization] "<< elapsed
			  << "s: real-time factor assuming 100 frames/sec is "
			  << (elapsed*100.0/frame_count);

	    return 0;
	} catch(const std::exception& e) {
	    std::cerr << e.what();
	    return -1;
	  }
} // main()

