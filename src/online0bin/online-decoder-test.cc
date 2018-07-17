// online0bin/online-decoder-test.cc

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
#include "online0/online-fst-decoder.h"

int main(int argc, char *argv[])
{
	try {

	    using namespace kaldi;
	    using namespace fst;

	    typedef kaldi::int32 int32;

	    const char *usage =
	    		"Reads in wav file(s) and simulates online decoding with neural nets "
	    		"(nnet0 or nnet1 setup), Note: some configuration values and inputs are\n"
	    	"set via config files whose filenames are passed as options\n"
	    	"\n"
	        "Usage: online-decoder-test [config option]\n"
	    	"e.g.: \n"
	        "	online-decoder-test --cfg=conf/decode.conf wav.scp\n";

	    ParseOptions po(usage);

	    std::string cfg;
	    po.Register("cfg", &cfg, "decoder config file");

        std::string audio_format = "wav";
        po.Register("audio-format", &audio_format, "input audio format(e.g. wav, pcm)");

        po.Read(argc, argv);

        if (po.NumArgs() < 1) {
        		po.PrintUsage();
        		exit(1);
        }

        std::string wavlist_rspecifier = po.GetArg(1);

	    OnlineFstDecoder decoder(cfg);
	    decoder.InitDecoder();

	    std::ifstream wavlist_reader(wavlist_rspecifier);

	    BaseFloat chunk_length_secs = 0.2, total_frames = 0, samp_freq = 16000;
        size_t size;
        Matrix<BaseFloat> audio_data;
	    Result *result;
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
                samp_freq = wave_data.SampFreq();
            }
            else if (audio_format == "pcm") {
                std::ifstream pcm_reader(fn, ios::binary);
                // get length of file:  
                pcm_reader.seekg(0, ios::end);
                int length = pcm_reader.tellg();
                pcm_reader.seekg(0, ios::beg);
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

            int32 chunk_length;
			if (chunk_length_secs > 0) {
				chunk_length = int32(samp_freq * chunk_length_secs);
				if (chunk_length == 0) chunk_length = 1;
			} else {
				chunk_length = std::numeric_limits<int32>::max();
			}

			// one utterance
			decoder.Reset();

			int32 samp_offset = 0;
			while (samp_offset < data.Dim()) {
				int32 samp_remaining = data.Dim() - samp_offset;
				int32 num_samp = chunk_length < samp_remaining ? chunk_length : samp_remaining;
                num_samp = data.Dim();

				SubVector<BaseFloat> wave_part(data, samp_offset, num_samp);
				samp_offset += num_samp;
				state = samp_offset >= data.Dim() ? FEAT_END : FEAT_APPEND;

                /*
				// feed part data
				if (state == FEAT_END) {
				    decoder.FeedData((void*)wave_part.Data(), wave_part.Dim()*sizeof(float), FEAT_APPEND);
				    decoder.FeedData((void*)wave_part.Data(), 0, FEAT_END);
                }
                else 
                */
				decoder.FeedData((void*)wave_part.Data(), wave_part.Dim()*sizeof(float), state);
				// get part result
				result = decoder.GetResult(state);
                result->utt = std::string(fn);
                if (state == FEAT_END)
            	    		KALDI_LOG << "Finish decode utterance: " << result->utt
							<< ", Log-like per frame is " << result->score_ << " over "
			                  << result->num_frames << " frames.";
                //usleep(chunk_length_secs/2*1e6);
			}
			total_frames += result->num_frames;
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


