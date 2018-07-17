// online0bin/online-faster-decoder.cc

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
#include "online0/online-fst-decoder.h"

int main(int argc, char *argv[])
{
	try {

	    using namespace kaldi;
	    using namespace fst;

	    typedef kaldi::int32 int32;

	    const char *usage =
	        "Reads log-likelihoods as matrices and simulates online decoding "
	        "(model is needed only for the integer mappings in its transition-model), "
	        "Note: some configuration values and inputs are\n"
	    	"set via config files whose filenames are passed as options\n"
	    	"\n"
	        "Usage: online-faster-decoder [config option] <loglikes-rspecifier> \n"
	    	"e.g.: \n"
	        "	online-faster-decoder --config=conf/online_decoder.conf <loglikes-rspecifier> \n";

	    ParseOptions po(usage);
	    OnlineNnetDecodingOptions decoding_opts;
	    decoding_opts.Register(&po);

	    po.Read(argc, argv);

	    if (po.NumArgs() != 1) {
	      po.PrintUsage();
	      exit(1);
	    }

	    std::string loglikes_rspecifier = po.GetArg(1);

	    OnlineNnetFasterDecoderOptions decoder_opts;
	    ReadConfigFromFile(decoding_opts.decoder_cfg, &decoder_opts);

	    Int32VectorWriter words_writer(decoding_opts.words_wspecifier);
	    Int32VectorWriter alignment_writer(decoding_opts.alignment_wspecifier);

	    TransitionModel trans_model;
	    ReadKaldiObject(decoding_opts.model_rspecifier, &trans_model);

	    fst::Fst<fst::StdArc> *decode_fst = ReadFstKaldi(decoding_opts.fst_rspecifier);
	    fst::SymbolTable *word_syms = NULL;
	    if (!(word_syms = fst::SymbolTable::ReadText(decoding_opts.word_syms_filename)))
	        KALDI_ERR << "Could not read symbol table from file " << decoding_opts.word_syms_filename;

	    DecoderSync decoder_sync;
        Result result;

	    OnlineNnetFasterDecoder decoder(*decode_fst, decoder_opts);
	    OnlineDecodableMatrixMapped decodable(trans_model, decoding_opts.acoustic_scale);

	    OnlineNnetDecodingClass decoding(decoding_opts,
	    								&decoder, &decodable, &decoder_sync,
										&result);
		// The initialization of the following class spawns the threads that
	    // process the examples.  They get re-joined in its destructor.
	    MultiThreader<OnlineNnetDecodingClass> m(1, decoding);

	    SequentialBaseFloatMatrixReader loglikes_reader(loglikes_rspecifier);

        Timer timer;

        kaldi::int64 frame_count = 0;

        for (; !loglikes_reader.Done(); loglikes_reader.Next()) {
            std::string utt_key = loglikes_reader.Key();
        	const Matrix<BaseFloat> &loglikes = loglikes_reader.Value();

            decodable.Reset();
            decodable.AcceptLoglikes(&loglikes);
			decodable.InputIsFinished();
            decoder_sync.DecoderSignal();

			frame_count += loglikes.NumRows();
			// waiting a utterance finished
            decoder_sync.DecoderSignal();
			decoder_sync.UtteranceWait();
            
            //decoder.GetResult(FEAT_END);
            PrintPartialResult(result.word_ids_, word_syms, true);
			KALDI_LOG << "Finish decode utterance: " << utt_key;
            result.clear();
        }

        decoder_sync.Abort();

        double elapsed = timer.Elapsed();
        KALDI_LOG << "Time taken [excluding initialization] "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed*100.0/frame_count);

	    delete decode_fst;
	    delete word_syms;
	    return 0;
	} catch(const std::exception& e) {
	    std::cerr << e.what();
	    return -1;
	  }
} // main()
