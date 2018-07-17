// online0/online-feature.h

// Copyright 2013   Johns Hopkins University (author: Daniel Povey)
//           2014   Yanqing Sun, Junjie Wang,
//                  Daniel Povey, Korbinian Riedhammer

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


#ifndef KALDI_ONLINE0_ONLINE_FEATURE_H_
#define KALDI_ONLINE0_ONLINE_FEATURE_H_

#include <string>
#include <vector>
#include <deque>

#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "base/kaldi-error.h"
#include "feat/feature-functions.h"
#include "feat/feature-mfcc.h"
#include "feat/feature-plp.h"
#include "feat/feature-fbank.h"
//#include "feat/pitch-functions.h"
#include "online0/online-feature-interface.h"

namespace kaldi {
/// @addtogroup  onlinefeat OnlineFeatureExtraction
/// @{


/// This is a templated class for online feature extraction;
/// it's templated on a class like MfccComputer or PlpComputer
/// that does the basic feature extraction.
template<class C>
class OnlineStreamGenericBaseFeature: public OnlineStreamBaseFeature {
 public:
  //
  // First, functions that are present in the interface:
  //
  virtual int32 Dim() const { return computer_.Dim(); }

  // Note: IsLastFrame() will only ever return true if you have called
  // InputFinished() (and this frame is the last frame).
  virtual bool IsLastFrame(int32 frame) const {
    return input_finished_ && frame == NumFramesReady() - 1;
  }

  virtual BaseFloat FrameShiftInSeconds() const {
    return computer_.GetFrameOptions().frame_shift_ms / 1000.0f;
  }

  virtual int32 NumFramesReady() const { return features_.size(); }

  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);

  virtual void Reset() {
	DeletePointers(&features_);
	features_.resize(0);
	input_finished_ = false;
	waveform_offset_ = 0;
    waveform_remainder_.Resize(0);
  }

  // Next, functions that are not in the interface.


  // Constructor from options class
  explicit OnlineStreamGenericBaseFeature(const typename C::Options &opts);

  // This would be called from the application, when you get
  // more wave data.  Note: the sampling_rate is only provided so
  // the code can assert that it matches the sampling rate
  // expected in the options.
  virtual void AcceptWaveform(BaseFloat sampling_rate,
                              const VectorBase<BaseFloat> &waveform);


  // InputFinished() tells the class you won't be providing any
  // more waveform.  This will help flush out the last frame or two
  // of features, in the case where snip-edges == false; it also
  // affects the return value of IsLastFrame().
  virtual void InputFinished() {
    input_finished_ = true;
    ComputeFeatures();
  }

  virtual ~OnlineStreamGenericBaseFeature() {
    DeletePointers(&features_);
  }

 private:
  // This function computes any additional feature frames that it is possible to
  // compute from 'waveform_remainder_', which at this point may contain more
  // than just a remainder-sized quantity (because AcceptWaveform() appends to
  // waveform_remainder_ before calling this function).  It adds these feature
  // frames to features_, and shifts off any now-unneeded samples of input from
  // waveform_remainder_ while incrementing waveform_offset_ by the same amount.
  void ComputeFeatures();

  C computer_;  // class that does the MFCC or PLP or filterbank computation

  FeatureWindowFunction window_function_;

  // features_ is the Mfcc or Plp or Fbank features that we have already computed.

  std::vector<Vector<BaseFloat>*> features_;

  // True if the user has called "InputFinished()"
  bool input_finished_;

  // The sampling frequency, extracted from the config.  Should
  // be identical to the waveform supplied.
  BaseFloat sampling_frequency_;

  // waveform_offset_ is the number of samples of waveform that we have
  // already discarded, i.e. thatn were prior to 'waveform_remainder_'.
  int64 waveform_offset_;

  // waveform_remainder_ is a short piece of waveform that we may need to keep
  // after extracting all the whole frames we can (whatever length of feature
  // will be required for the next phase of computation).
  Vector<BaseFloat> waveform_remainder_;
};

typedef OnlineStreamGenericBaseFeature<MfccComputer> OnlineStreamMfcc;
typedef OnlineStreamGenericBaseFeature<PlpComputer> OnlineStreamPlp;
typedef OnlineStreamGenericBaseFeature<FbankComputer> OnlineStreamFbank;


class OnlineStreamDeltaFeature: public OnlineStreamFeatureInterface {
 public:
  //
  // First, functions that are present in the interface:
  //
  virtual int32 Dim() const;

  virtual bool IsLastFrame(int32 frame) const {
    return src_->IsLastFrame(frame);
  }

  virtual BaseFloat FrameShiftInSeconds() const {
    return src_->FrameShiftInSeconds();
  }

  virtual int32 NumFramesReady() const;

  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);

  virtual void Reset() {
    src_->Reset();  
  }

  //
  // Next, functions that are not in the interface.
  //
  OnlineStreamDeltaFeature(const DeltaFeaturesOptions &opts,
                     OnlineStreamFeatureInterface *src);

 private:
  OnlineStreamFeatureInterface *src_;  // Not owned here
  DeltaFeaturesOptions opts_;
  DeltaFeatures delta_features_;  // This class contains just a few
                                  // coefficients.
};


struct OnlineStreamCmvnOptions {
  int32 min_window;
  int32 cmn_window;
  bool normalize_mean;
  bool normalize_variance;

  OnlineStreamCmvnOptions():
	  min_window(100),
      cmn_window(600),
	  normalize_mean(true),
	  normalize_variance(false)
	  { }

  void Check() {
		KALDI_ASSERT(min_window >= 0);
		if (cmn_window >= 0)
			KALDI_ASSERT(cmn_window >= min_window);
  }

  void Register(OptionsItf *opts) {
	    opts->Register("min-cmn-window", &min_window, "Minimum CMN window "
	                   "used at start of decoding (adds latency only at start). "
	                   "Only applicable if center == false, ignored if center==true");
	    opts->Register("cmn-window", &cmn_window, "Window in frames for running "
	                   "average CMN computation");
	    opts->Register("norm-mean", &normalize_mean, "If true, do mean normalization "
	                     "(note: you cannot normalize the variance but not the mean)");
	    opts->Register("norm-vars", &normalize_variance, "If true, normalize "
	                   "variance to one."); // naming this as in apply-cmvn.cc
  }
};

/**
   This class does an online version of the cepstral mean and [optionally]
   variance, but note that this is not equivalent to the offline version.  This
   is necessarily so, as the offline computation involves looking into the
   future.  If you plan to use features normalized with this type of CMVN then
   you need to train in a `matched' way, i.e. with the same type of features.
   We normally only do so in the "online" nnet-based decoding, e.g.nnet0
*/

class OnlineStreamCmvnFeature: public OnlineStreamFeatureInterface {
 public:
  //
  // First, functions that are present in the interface:
  //
  virtual int32 Dim() const {
    return src_->Dim();
  }

  virtual bool IsLastFrame(int32 frame) const {
    return src_->IsLastFrame(frame);
  }

  virtual BaseFloat FrameShiftInSeconds() const {
    return src_->FrameShiftInSeconds();
  }

  virtual int32 NumFramesReady() const;

  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);

  virtual void Reset() {
	DeletePointers(&features_);
	sum_.Resize(src_->Dim());
	sumsq_.Resize(src_->Dim());
	features_.resize(0);
    src_->Reset();
  }

  //
  // Next, functions that are not in the interface.
  //
  OnlineStreamCmvnFeature(const OnlineStreamCmvnOptions &opts,
		  OnlineStreamFeatureInterface *src);

  virtual ~OnlineStreamCmvnFeature() {
    DeletePointers(&features_);
  }

 private:
  void ComputeCmvnInternal();

  const OnlineStreamCmvnOptions &opts_;
  OnlineStreamFeatureInterface *src_;  // Not owned here

  Vector<double> sum_;
  Vector<double> sumsq_;

  std::vector<Vector<BaseFloat>*> features_;


  KALDI_DISALLOW_COPY_AND_ASSIGN(OnlineStreamCmvnFeature);
};


/**
 * splice feature
 */
struct OnlineStreamSpliceOptions {
  int32 left_context;
  int32 right_context;
  int32 context;
  bool  custom_splice;
  OnlineStreamSpliceOptions(): left_context(0), right_context(0), context(0), custom_splice(false) { }
  void Register(OptionsItf *opts) {
	  opts->Register("left-context", &left_context, "Left-context for frame ");
	  opts->Register("right-context", &right_context, "Right-context for frame ");
	  opts->Register("context", &context, "Right-context and Left-context for frame ");
	  opts->Register("custom-splice", &custom_splice, "Use custom Right-context and Left-context for frame ");
  }
};

class OnlineStreamSpliceFeature: public OnlineStreamFeatureInterface {
 public:
  //
  // First, functions that are present in the interface:
  //
  virtual int32 Dim() const {
    return src_->Dim() * (1 + left_context_ + right_context_);
  }

  virtual bool IsLastFrame(int32 frame) const {
    return src_->IsLastFrame(frame);
  }

  virtual BaseFloat FrameShiftInSeconds() const {
    return src_->FrameShiftInSeconds();
  }

  virtual int32 NumFramesReady() const;

  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);

  virtual void Reset() {
    src_->Reset(); 
  }

  //
  // Next, functions that are not in the interface.
  //
  OnlineStreamSpliceFeature(const OnlineStreamSpliceOptions &opts,
		  OnlineStreamFeatureInterface *src);

 private:
  int32 left_context_;
  int32 right_context_;
  OnlineStreamFeatureInterface *src_;  // Not owned here
};


/// @} End of "addtogroup onlinefeat"
}  // namespace kaldi

#endif  // KALDI_ONLINE0_ONLINE_FEATURE_H_
