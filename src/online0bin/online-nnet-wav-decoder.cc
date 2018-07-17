// online0bin/online-nnet-wav-decoder.cc

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

#include "online0/online-util.h"
#include "online0/online-nnet-decoding.h"

void PrintResult(int &curt, fst::SymbolTable *word_syms, kaldi::Result &result, bool state) {
	std::vector<int32> word_ids;
	int size = result.word_ids_.size();
	for (int i = curt; i < size; i++)
		word_ids.push_back(result.word_ids_[i]);

	kaldi::PrintPartialResult(word_ids, word_syms, state == true );
	curt = size;
}

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
	        "Usage: online-nnet-decoder [config option]\n"
	    	"e.g.: \n"
	        "	online-nnet-decoder --config=conf/online_decoder.conf --wavscp=wav.scp\n";

	    ParseOptions po(usage);

	    OnlineNnetDecodingOptions decoding_opts;
	    decoding_opts.Register(&po);

	    std::string wav_rspecifier;
		po.Register("wavscp", &wav_rspecifier, "wav list for decode");

	    po.Read(argc, argv);

	    if (argc < 2) {
	        po.PrintUsage();
	        exit(1);
	    }


	    OnlineNnetFasterDecoderOptions decoder_opts;
	    OnlineNnetForwardOptions forward_opts;
	    OnlineNnetFeaturePipelineOptions feature_opts(decoding_opts.feature_cfg);
		ReadConfigFromFile(decoding_opts.decoder_cfg, &decoder_opts);
		ReadConfigFromFile(decoding_opts.forward_cfg, &forward_opts);

#if HAVE_CUDA==1
    if (forward_opts.use_gpu == "yes")
        CuDevice::Instantiate().Initialize();
#endif

	    Int32VectorWriter words_writer(decoding_opts.words_wspecifier);
	    Int32VectorWriter alignment_writer(decoding_opts.alignment_wspecifier);
        Result result;

	    //SequentialTableReader<WaveHolder> wav_reader(wav_rspecifier);
	    std::ifstream wav_reader(wav_rspecifier);

	    TransitionModel trans_model;
		bool binary;
		Input ki(decoding_opts.model_rspecifier, &binary);
		trans_model.Read(ki.Stream(), binary);

	    OnlineDecodableMatrixMapped decodable(trans_model, decoding_opts.acoustic_scale);
	    DecoderSync decoder_sync;

	    fst::Fst<fst::StdArc> *decode_fst = ReadFstKaldi(decoding_opts.fst_rspecifier);
	    fst::SymbolTable *word_syms = NULL;
	    if (!(word_syms = fst::SymbolTable::ReadText(decoding_opts.word_syms_filename)))
	        KALDI_ERR << "Could not read symbol table from file " << decoding_opts.word_syms_filename;

	    OnlineNnetFasterDecoder decoder(*decode_fst, decoder_opts);
	    OnlineNnetDecodingClass decoding(decoding_opts,
	    								&decoder, &decodable, &decoder_sync, &result);
		// The initialization of the following class spawns the threads that
	    // process the examples.  They get re-joined in its destructor.
	    MultiThreader<OnlineNnetDecodingClass> m(1, decoding);

        // client feature
        Timer timer;

        OnlineNnetFeaturePipeline feature_pipeline(feature_opts);
        OnlineNnetForward forward(forward_opts);
        BaseFloat chunk_length_secs = decoding_opts.chunk_length_secs;
        int feat_dim = feature_pipeline.Dim();
        int skip_frames = decoding_opts.skip_frames;
        bool skip_inner = decoding_opts.skip_inner;
        bool copy_posterior = decoding_opts.copy_posterior;
        int batch_size = forward_opts.batch_size * skip_frames;
        int in_skip = skip_inner ? 1:skip_frames,
        out_skip = skip_inner ? skip_frames : 1;
        Matrix<BaseFloat> feat(out_skip*forward_opts.batch_size, feat_dim);
        Matrix<BaseFloat> feat_out, feat_out_ready;
        char fn[1024];

        kaldi::int64 frame_count = 0;

        while (wav_reader.getline(fn, 1024)) {
        	WaveHolder holder;
        	bool binary;
        	Input ki(fn, &binary);
        	holder.Read(ki.Stream());

        	const WaveData &wave_data = holder.Value();
            // get the data for channel zero (if the signal is not mono, we only
            // take the first channel).
            SubVector<BaseFloat> data(wave_data.Data(), 0);

            BaseFloat samp_freq = wave_data.SampFreq();
            int32 chunk_length;
			if (chunk_length_secs > 0) {
				chunk_length = int32(samp_freq * chunk_length_secs);
				if (chunk_length == 0) chunk_length = 1;
			} else {
				chunk_length = std::numeric_limits<int32>::max();
			}

			feature_pipeline.Reset();
			decodable.Reset();
			forward.ResetHistory();
			int32 samp_offset = 0, frame_offset = 0, frame_ready, curt = 0;
			while (samp_offset < data.Dim()) {
				int32 samp_remaining = data.Dim() - samp_offset;
				int32 num_samp = chunk_length < samp_remaining ? chunk_length : samp_remaining;

				SubVector<BaseFloat> wave_part(data, samp_offset, num_samp);
				feature_pipeline.AcceptWaveform(samp_freq, wave_part);
				samp_offset += num_samp;
				if (samp_offset >= data.Dim())
					feature_pipeline.InputFinished();

				while (true) {
					frame_ready = feature_pipeline.NumFramesReady();
					if (!feature_pipeline.IsLastFrame(frame_ready-1) && frame_ready < frame_offset+batch_size)
						break;
                    else if (feature_pipeline.IsLastFrame(frame_ready-1) && frame_ready == frame_offset)
                        break;
					else if (feature_pipeline.IsLastFrame(frame_ready-1) && frame_ready < frame_offset+batch_size) {
						frame_ready -= frame_offset;
						feat.SetZero();
					}
					else
						frame_ready = batch_size;

					for (int i = 0; i < frame_ready; i += in_skip) {
						feature_pipeline.GetFrame(frame_offset+i, &feat.Row(i/in_skip));
					}
					frame_offset += frame_ready;

					// feed forward to neural network
					forward.Forward(feat, &feat_out);

					// copy posterior
					if (copy_posterior) {
						feat_out_ready.Resize(frame_ready, feat_out.NumCols(), kUndefined);
						for (int i = 0; i < frame_ready; i++)
							feat_out_ready.Row(i).CopyFromVec(feat_out.Row(i/skip_frames));

						decodable.AcceptLoglikes(&feat_out_ready);
					}
					else
						decodable.AcceptLoglikes(&feat_out);

					decoder_sync.DecoderSignal();
				} // part wav data

				PrintResult(curt, word_syms, result, false);
			} // finish a wav 

			frame_count += frame_offset;
			decodable.InputIsFinished();
			decoder_sync.DecoderSignal();
			// waiting a utterance finished
			decoder_sync.UtteranceWait();

			PrintResult(curt, word_syms, result, true);
            result.clear();
			KALDI_LOG << "Finish decode utterance: " << fn;
        }

        decoder_sync.Abort();
		wav_reader.close();

        double elapsed = timer.Elapsed();
        KALDI_LOG << "Time taken [excluding initialization] "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed*100.0/frame_count);

	    //delete decode_fst;
	    //delete word_syms;
	    usleep(0.1*1e6);
		decoder_sync.DecoderSignal();
	    return 0;
	} catch(const std::exception& e) {
	    std::cerr << e.what();
	    return -1;
	}
} // main()
