// online0/online-nnet-feature-pipeline.h

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


#ifndef KALDI_ONLINE0_ONLINE_NNET_FEATURE_PIPELINE_H_
#define KALDI_ONLINE0_ONLINE_NNET_FEATURE_PIPELINE_H_

#include "online0/online-feature.h"

namespace kaldi {

struct OnlineNnetFeaturePipelineConfig {
    BaseFloat samp_freq;
	std::string feature_type;
	std::string mfcc_config;
	std::string plp_config;
	std::string fbank_config;

	bool add_pitch;
	// the following contains the type of options that you could give to
	// compute-and-process-kaldi-pitch-feats.
	std::string online_pitch_config;

	bool add_cmvn;
	std::string cmvn_config;
	bool add_deltas;
	std::string delta_config;
	bool splice_feats;
	std::string splice_config;

	OnlineNnetFeaturePipelineConfig():
		samp_freq(16000), feature_type("fbank"), add_pitch(false),
		add_cmvn(false), add_deltas(false), splice_feats(false) { }

	void Register(OptionsItf *opts) {
	    opts->Register("sample-frequency", &samp_freq,
	                   "Waveform data sample frequency (must match the waveform file, if specified there)");
	    opts->Register("feature-type", &feature_type,
	                   "Base feature type [mfcc, plp, fbank]");
	    opts->Register("mfcc-config", &mfcc_config, "Configuration file for "
	                   "MFCC features (e.g. conf/mfcc.conf)");
	    opts->Register("plp-config", &plp_config, "Configuration file for "
	                   "PLP features (e.g. conf/plp.conf)");
	    opts->Register("fbank-config", &fbank_config, "Configuration file for "
	                   "filterbank features (e.g. conf/fbank.conf)");
	    opts->Register("add-pitch", &add_pitch, "Append pitch features to raw "
	                   "MFCC/PLP/filterbank features [but not for iVector extraction]");
	    opts->Register("online-pitch-config", &online_pitch_config, "Configuration "
	                   "file for online pitch features, if --add-pitch=true (e.g. "
	                   "conf/online_pitch.conf)");
        opts->Register("add-cmvn", &add_cmvn,
                       "Apply cmvn features.");
	    opts->Register("cmvn-config", &cmvn_config, "Configuration class "
	                   "file for online CMVN features (e.g. conf/online_cmvn.conf)");
	    opts->Register("add-deltas", &add_deltas,
	                   "Append delta features.");
	    opts->Register("delta-config", &delta_config, "Configuration file for "
	                   "delta feature computation (if not supplied, will not apply "
	                   "delta features; supply empty config to use defaults.)");
	    opts->Register("splice-feats", &splice_feats, "Splice features with left and "
	                   "right context.");
	    opts->Register("splice-config", &splice_config, "Configuration file "
	                   "for frame splicing, if done (e.g. prior to LDA)");
	  }

};


struct OnlineNnetFeaturePipelineOptions {
	OnlineNnetFeaturePipelineOptions():
		feature_type("fbank"), add_pitch(false),
		add_cmvn(false), add_deltas(false), splice_feats(false), samp_freq(16000) { }

	OnlineNnetFeaturePipelineOptions(const OnlineNnetFeaturePipelineConfig &config);

	std::string feature_type;  // "mfcc" or "plp" or "fbank"

	MfccOptions mfcc_opts;    	// options for MFCC computation, if feature_type == "mfcc"
	PlpOptions plp_opts;  		// options for PLP computation, if feature_type == "plp"
	FbankOptions fbank_opts;  	// options for filterbank computation, if feature_type == "fbank"

	bool add_pitch;
	//PitchExtractionOptions pitch_opts;  // Options for pitch extraction, if done.
	//ProcessPitchOptions pitch_process_opts;  // Options for pitch post-processing

	bool add_cmvn;
	OnlineStreamCmvnOptions cmvn_opts;  // Options for online CMN/CMVN computation.

	bool add_deltas;
	DeltaFeaturesOptions delta_opts;  // Options for delta computation, if done.

	bool splice_feats;
	OnlineStreamSpliceOptions splice_opts;  // Options for frame splicing, if done.

	BaseFloat samp_freq;

private:
	KALDI_DISALLOW_COPY_AND_ASSIGN(OnlineNnetFeaturePipelineOptions);
};


class OnlineNnetFeaturePipeline: public OnlineStreamBaseFeature {
public:
	explicit OnlineNnetFeaturePipeline(const OnlineNnetFeaturePipelineOptions &opts);
	virtual ~OnlineNnetFeaturePipeline();

	/// Member functions from OnlineStreamFeatureInterface:
	virtual int32 Dim() const;
	virtual bool IsLastFrame(int32 frame) const;
	virtual int32 NumFramesReady() const;
	virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);
	virtual BaseFloat FrameShiftInSeconds() const;
	virtual void Reset();

	/// Accept more data to process.  It won't actually process it until you call
	/// GetFrame() [probably indirectly via (decoder).AdvanceDecoding()], when you
	/// call this function it will just copy it).  sampling_rate is necessary just
	/// to assert it equals what's in the config.
	void AcceptWaveform(BaseFloat sampling_rate, const VectorBase<BaseFloat> &waveform);

	// InputFinished() tells the class you won't be providing any
	// more waveform.  This will help flush out the last few frames
	// of delta or LDA features, and finalize the pitch features
	// (making them more accurate).
	void InputFinished();

private:

	const OnlineNnetFeaturePipelineOptions &opts_;

	OnlineStreamBaseFeature *base_feature_;		// MFCC/PLP/filterbank
	//OnlinePitchFeature *pitch_;              // Raw pitch, if used
	//OnlineProcessPitch *pitch_feature_;  // Processed pitch, if pitch used
	//OnlineStreamFeatureInterface *feature_;        // CMVN (+ processed pitch)

	OnlineStreamCmvnFeature *cmvn_feature_;
	OnlineStreamDeltaFeature *delta_feature_;
	OnlineStreamSpliceFeature *splice_feature_;  // This may be NULL if we're not
	                                             // doing splicing or deltas.
	OnlineStreamFeatureInterface *final_feature_;
};

} // kaldi namespace

#endif /* KALDI_ONLINE0_ONLINE_NNET_FEATURE_PIPELINE_H_ */
