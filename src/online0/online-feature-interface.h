// online0/online-feature-interface.h

// Copyright    2013  Johns Hopkins University (author: Daniel Povey)
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

#ifndef KALDI_ONLINE0_ONLINE_FEATURE_INTERFACE_H_
#define KALDI_ONLINE0_ONLINE_FEATURE_INTERFACE_H_ 1
#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "itf/online-feature-itf.h"

namespace kaldi {
/// @ingroup Interfaces
/// @{

/**
   OnlineFeatureInterface is an interface for online feature processing (it is
   also usable in the offline setting, but currently we're not using it for
   that).  This is for use in the online2/ directory, and it supersedes the
   interface in ../online/online-feat-input.h.  We have a slighty different
   model that puts more control in the hands of the calling thread, and won't
   involve waiting on semaphores in the decoding thread.

   This interface only specifies how the object *outputs* the features.
   How it obtains the features, e.g. from a previous object or objects of type
   OnlineFeatureInterface, is not specified in the interface and you will
   likely define new constructors or methods in the derived type to do that.

   You should appreciate that this interface is designed to allow random
   access to features, as long as they are ready.  That is, the user
   can call GetFrame for any frame less than NumFramesReady(), and when
   implementing a child class you must not make assumptions about the
   order in which the user makes these calls.
*/
   
class OnlineStreamFeatureInterface : public OnlineFeatureInterface {
 public:
  // Reset feature extraction status
  virtual void Reset() = 0;

  /// Virtual destructor.  Note: constructors that take another member of
  /// type OnlineFeatureInterface are not expected to take ownership of
  /// that pointer; the caller needs to keep track of that manually.
  virtual ~OnlineStreamFeatureInterface() { }
  
};


/// Add a virtual class for "source" features such as MFCC or PLP or pitch
/// features.
class OnlineStreamBaseFeature: public OnlineStreamFeatureInterface {
 public:
  /// This would be called from the application, when you get more wave data.
  /// Note: the sampling_rate is typically only provided so the code can assert
  /// that it matches the sampling rate expected in the options.
  virtual void AcceptWaveform(BaseFloat sampling_rate,
                              const VectorBase<BaseFloat> &waveform) = 0;

  /// InputFinished() tells the class you won't be providing any
  /// more waveform.  This will help flush out the last few frames
  /// of delta or LDA features (it will typically affect the return value
  /// of IsLastFrame.
  virtual void InputFinished() = 0;
};


/// @}
}  // namespace Kaldi

#endif  // KALDI_ONLINE0_ONLINE_FEATURE_INTERFACE_H_
