// decoder/decoder-ctc-wrappers.h

// Copyright   2014  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_DECODER_DECODER_CTC_WRAPPERS_H_
#define KALDI_DECODER_DECODER_CTC_WRAPPERS_H_

#include "itf/options-itf.h"
#include "decoder/lattice-faster-decoder.h"

// This header contains declarations from various convenience functions that are called
// from binary-level programs such as gmm-decode-faster.cc, gmm-align-compiled.cc, and
// so on.

namespace kaldi {


/// This function DecodeUtteranceLatticeFaster is used in several decoders, and
/// we have moved it here.  Note: this is really "binary-level" code as it
/// involves table readers and writers; we've just put it here as there is no
/// other obvious place to put it.  If determinize == false, it writes to
/// lattice_writer, else to compact_lattice_writer.  The writers for
/// alignments and words will only be written to if they are open.
bool DecodeUtteranceLatticeFasterCtc(
    LatticeFasterDecoder &decoder, // not const but is really an input.
    DecodableInterface &decodable, // not const but is really an input.
    const fst::SymbolTable *word_syms,
    std::string utt,
    double acoustic_scale,
    bool determinize,
    bool allow_partial,
    Int32VectorWriter *alignments_writer,
    Int32VectorWriter *words_writer,
    CompactLatticeWriter *compact_lattice_writer,
    LatticeWriter *lattice_writer,
    double *like_ptr);  // puts utterance's likelihood in like_ptr on success.

/// This class basically does the same job as the function
/// DecodeUtteranceLatticeFaster, but in a way that allows us
/// to build a multi-threaded command line program more easily,
/// using code in ../thread/kaldi-task-sequence.h.  The main
/// computation takes place in operator (), and the output happens
/// in the destructor.
class DecodeUtteranceLatticeFasterCtcClass {
 public:
  // Initializer sets various variables.
  // NOTE: we "take ownership" of "decoder" and "decodable".  These
  // are deleted by the destructor.  On error, "num_err" is incremented.
  DecodeUtteranceLatticeFasterCtcClass(
      LatticeFasterDecoder *decoder,
      DecodableInterface *decodable,
      const fst::SymbolTable *word_syms,
      std::string utt,
      BaseFloat acoustic_scale,
      bool determinize,
      bool allow_partial,
      Int32VectorWriter *alignments_writer,
      Int32VectorWriter *words_writer,
      CompactLatticeWriter *compact_lattice_writer,
      LatticeWriter *lattice_writer,
      double *like_sum, // on success, adds likelihood to this.
      int64 *frame_sum, // on success, adds #frames to this.
      int32 *num_done, // on success (including partial decode), increments this.
      int32 *num_err,  // on failure, increments this.
      int32 *num_partial);  // If partial decode (final-state not reached), increments this.
  void operator () (); // The decoding happens here.
  ~DecodeUtteranceLatticeFasterCtcClass(); // Output happens here.
 private:
  // The following variables correspond to inputs:
  LatticeFasterDecoder *decoder_;
  DecodableInterface *decodable_;
  const fst::SymbolTable *word_syms_;
  std::string utt_;
  BaseFloat acoustic_scale_;
  bool determinize_;
  bool allow_partial_;
  Int32VectorWriter *alignments_writer_;
  Int32VectorWriter *words_writer_;
  CompactLatticeWriter *compact_lattice_writer_;
  LatticeWriter *lattice_writer_;
  double *like_sum_;
  int64 *frame_sum_;
  int32 *num_done_;
  int32 *num_err_;
  int32 *num_partial_;

  // The following variables are stored by the computation.
  bool computed_; // operator ()  was called.
  bool success_; // decoding succeeded (possibly partial)
  bool partial_; // decoding was partial.
  CompactLattice *clat_; // Stored output, if determinize_ == true.
  Lattice *lat_; // Stored output, if determinize_ == false.
};


} // end namespace kaldi.


#endif
