// online0bin/online-kws-test.cc

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
#include "feat/wave-reader.h"
#include "online0/online-keyword-spotting.h"

int main(int argc, char *argv[])
{
	try {

	    using namespace kaldi;

	    typedef kaldi::int32 int32;

	    const char *usage =
	    		"Reads in wav file(s) and simulates online keyword spotting with neural nets "
	    		"(nnet0 or nnet1 setup), Note: some configuration values and inputs are\n"
	    	"set via config files whose filenames are passed as options\n"
	    	"\n"
	        "Usage: online-kws-test [config option]\n"
	    	"e.g.: \n"
	        "	online-kws-test --cfg=conf/kws.conf wav.scp\n";

	    ParseOptions po(usage);

	    std::string cfg;
	    po.Register("cfg", &cfg, "keyword spotting config file");

        std::string audio_format = "wav";
        po.Register("audio-format", &audio_format, "input audio format(e.g. wav, pcm)");

        po.Read(argc, argv);

        if (po.NumArgs() < 1) {
        		po.PrintUsage();
        		exit(1);
        }

        std::string wavlist_rspecifier = po.GetArg(1);

        OnlineKeywordSpotting kws(cfg);
	    kws.InitKws();

	    std::ifstream wavlist_reader(wavlist_rspecifier);

	    BaseFloat  total_frames = 0, num_frames;
        size_t size;
        Matrix<BaseFloat> audio_data;
	    FeatState state;
        char fn[1024];

        Timer timer;
	    while (wavlist_reader.getline(fn, 1024)) {
            if (audio_format == "wav") {
			    bool binary;
			    Input ki(fn, &binary);
			    WaveHolder holder;
			    holder.Read(ki.Stream());

			    const WaveData &wave_data = holder.Value();
                audio_data = wave_data.Data();
            }
            else if (audio_format == "pcm") {
                std::ifstream pcm_reader(fn, std::ios::binary);
                // get length of file:
                pcm_reader.seekg(0, std::ios::end);
                int length = pcm_reader.tellg();
                pcm_reader.seekg(0, std::ios::beg);
                size = length/sizeof(short);
                std::vector<short> buffer(size);
                // read data as a block:
                pcm_reader.read((char*)&buffer.front(), length);
                audio_data.Resize(1, size);
                for (int i = 0; i < size; i++)
                    audio_data(0, i) = buffer[i];
                pcm_reader.close();
            }
            else
                KALDI_ERR << "Unsupported input audio format, now only support wav or pcm.";

			// get the data for channel zero (if the signal is not mono, we only
			// take the first channel).
			SubVector<BaseFloat> data(audio_data, 0);

			// one utterance
			kws.Reset();

			state = FEAT_END;
			num_frames = kws.FeedData((void*)data.Data(), data.Dim()*sizeof(float), state);
			int iswakeup = kws.isWakeUp();
			KALDI_LOG << iswakeup << " " << std::string(fn);

			total_frames += num_frames;
	    }

	    wavlist_reader.close();

	    double elapsed = timer.Elapsed();
		KALDI_LOG << "Time taken [excluding initialization] "<< elapsed
			  << "s: real-time factor assuming 100 frames/sec is "
			  << (elapsed*100.0/total_frames);

	    return 0;
	} catch(const std::exception& e) {
		std::cerr << e.what();
		return -1;
	}
} // main()


