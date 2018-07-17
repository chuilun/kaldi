// online0/online-nnet-faster-decoder.h

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

#ifndef ONLINE0_ONLINE_NNET_FASTER_DECODER_H_
#define ONLINE0_ONLINE_NNET_FASTER_DECODER_H_

#include "util/stl-utils.h"
#include "decoder/faster-decoder.h"
#include "hmm/transition-model.h"
#include "online0/online-util.h"

namespace kaldi {

// Extends the definition of FasterDecoder's options to include additional
// parameters. The meaning of the "beam" option is also redefined as
// the _maximum_ beam value allowed.
struct OnlineNnetFasterDecoderOptions : public FasterDecoderOptions {
	  BaseFloat rt_min; // minimum decoding runtime factor
	  BaseFloat rt_max; // maximum decoding runtime factor
	  int32 batch_size; // number of features decoded in one go
	  int32 inter_utt_sil; // minimum silence (#frames) to trigger end of utterance
	  int32 max_utt_len_; // if utt. is longer, we accept shorter silence as utt. separators
	  int32 update_interval; // beam update period in # of frames
	  BaseFloat beam_update; // rate of adjustment of the beam
	  BaseFloat max_beam_update; // maximum rate of beam adjustment
	  std::string cutoff;

	  OnlineNnetFasterDecoderOptions() :
	    rt_min(0.7), rt_max(0.75), batch_size(18),
	    inter_utt_sil(50), max_utt_len_(1500),
	    update_interval(3), beam_update(0.01),
	    max_beam_update(0.05), cutoff("hybrid") {}

	  void Register(OptionsItf *opts, bool full = true) {
	    FasterDecoderOptions::Register(opts, full);
	    opts->Register("rt-min", &rt_min,
	                   "Approximate minimum decoding run time factor");
	    opts->Register("rt-max", &rt_max,
	                   "Approximate maximum decoding run time factor");
	    opts->Register("batch-size", &batch_size,
	                   "number of features decoded in one go");
	    opts->Register("update-interval", &update_interval,
	                   "Beam update interval in frames");
	    opts->Register("beam-update", &beam_update, "Beam update rate");
	    opts->Register("max-beam-update", &max_beam_update, "Max beam update rate");
	    opts->Register("inter-utt-sil", &inter_utt_sil,
	                   "Maximum # of silence frames to trigger new utterance");
	    opts->Register("max-utt-length", &max_utt_len_,
	                   "If the utterance becomes longer than this number of frames, "
	                   "shorter silence is acceptable as an utterance separator");
	    opts->Register("cutoff", &cutoff,
	    					"token cutoff algorithm, e.g. ctc or hmm-dnn hybrid");
	  }
};

class OnlineNnetFasterDecoder : public FasterDecoder {


public:
	// Codes returned by Decode() to show the current state of the decoder
	enum DecodeState {
		kStartFeats = 1, // Start from the Decodable
		kEndFeats = 2, // No more scores are available from the Decodable
		kEndBatch = 3, // End of batch - end of utterance not reached yet
	};

	OnlineNnetFasterDecoder(const fst::Fst<fst::StdArc> &fst,
							const OnlineNnetFasterDecoderOptions &opts):
								FasterDecoder(fst, opts), opts_(opts),
								max_beam_(opts.beam), effective_beam_(FasterDecoder::config_.beam),
								state_(kStartFeats), frame_(0), utt_frames_(0),
								immortal_tok_(NULL), prev_immortal_tok_(NULL)
							{

							}

	DecodeState Decode(DecodableInterface *decodable);

	// Makes a linear graph, by tracing back from the last "immortal" token
	// to the previous one
	bool PartialTraceback(fst::MutableFst<LatticeArc> *out_fst);

	// Makes a linear graph, by tracing back from the best currently active token
	// to the last immortal token. This method is meant to be invoked at the end
	// of an utterance in order to get the last chunk of the hypothesis
	void FinishTraceBack(fst::MutableFst<LatticeArc> *fst_out);

	int32 frame() { return frame_; }

	void ResetDecoder(bool full);

private:

	// Returns a linear fst by tracing back the last N frames, beginning
	// from the best current token
	void TracebackNFrames(int32 nframes, fst::MutableFst<LatticeArc> *out_fst);

	// Makes a linear "lattice", by tracing back a path delimited by two tokens
	void MakeLattice(const Token *start,
				   const Token *end,
				   fst::MutableFst<LatticeArc> *out_fst) const;

	// Searches for the last token, ancestor of all currently active tokens
	void UpdateImmortalToken();


	const OnlineNnetFasterDecoderOptions &opts_;
	const BaseFloat max_beam_; // the maximum allowed beam
	BaseFloat &effective_beam_; // the currently used beam
	DecodeState state_; // the current state of the decoder
	int32 frame_; // the next frame to be processed
	int32 utt_frames_; // # frames processed from the current utterance
	Token *immortal_tok_;      // "immortal" token means it's an ancestor of ...
	Token *prev_immortal_tok_; // ... all currently active tokens
	KALDI_DISALLOW_COPY_AND_ASSIGN(OnlineNnetFasterDecoder);
};

}	 // namespace kaldi



#endif /* ONLINE0_ONLINE_NNET_FASTER_DECODER_H_ */
