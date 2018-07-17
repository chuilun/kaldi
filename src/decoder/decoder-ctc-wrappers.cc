// decoder/decoder-wrappers.cc

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)

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

#include "decoder/decoder-ctc-wrappers.h"
#include "decoder/faster-decoder.h"

namespace kaldi {




// Takes care of output.  Returns true on success.
bool DecodeUtteranceLatticeFasterCtc(
    LatticeFasterDecoder &decoder, // not const but is really an input.
    DecodableInterface &decodable, // not const but is really an input.
    const fst::SymbolTable *word_syms,
    std::string utt,
    double acoustic_scale,
    bool determinize,
    bool allow_partial,
    Int32VectorWriter *alignment_writer,
    Int32VectorWriter *words_writer,
    CompactLatticeWriter *compact_lattice_writer,
    LatticeWriter *lattice_writer,
    double *like_ptr) { // puts utterance's like in like_ptr on success.
  using fst::VectorFst;

  if (!decoder.DecodeCtc(&decodable)) {
    KALDI_WARN << "Failed to decode file " << utt;
    return false;
  }
  if (!decoder.ReachedFinal()) {
    if (allow_partial) {
      KALDI_WARN << "Outputting partial output for utterance " << utt
                 << " since no final-state reached\n";
    } else {
      KALDI_WARN << "Not producing output for utterance " << utt
                 << " since no final-state reached and "
                 << "--allow-partial=false.\n";
      return false;
    }
  }

  double likelihood;
  LatticeWeight weight;
  int32 num_frames;
  { // First do some stuff with word-level traceback...
    VectorFst<LatticeArc> decoded;
    if (!decoder.GetBestPath(&decoded))
      // Shouldn't really reach this point as already checked success.
      KALDI_ERR << "Failed to get traceback for utterance " << utt;

    std::vector<int32> alignment;
    std::vector<int32> words;
    GetLinearSymbolSequence(decoded, &alignment, &words, &weight);
    num_frames = alignment.size();
    if (words_writer->IsOpen())
      words_writer->Write(utt, words);
    if (alignment_writer->IsOpen())
      alignment_writer->Write(utt, alignment);
    if (word_syms != NULL) {
      std::cerr << utt << ' ';
      for (size_t i = 0; i < words.size(); i++) {
        std::string s = word_syms->Find(words[i]);
        if (s == "")
          KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
        std::cerr << s << ' ';
      }
      std::cerr << '\n';
    }
    likelihood = -(weight.Value1() + weight.Value2());
  }

  // Get lattice, and do determinization if requested.
  Lattice lat;
  decoder.GetRawLattice(&lat);
  if (lat.NumStates() == 0)
    KALDI_ERR << "Unexpected problem getting lattice for utterance " << utt;
  fst::Connect(&lat);
  if (determinize) {
    CompactLattice clat;
    if (!DeterminizeLatticePhonePrunedCtcWrapper(
            &lat,
            decoder.GetOptions().lattice_beam,
            &clat,
            decoder.GetOptions().det_opts))
      KALDI_WARN << "Determinization finished earlier than the beam for "
                 << "utterance " << utt;
    // We'll write the lattice without acoustic scaling.
    if (acoustic_scale != 0.0)
      fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale), &clat);
    compact_lattice_writer->Write(utt, clat);
  } else {
    // We'll write the lattice without acoustic scaling.
    if (acoustic_scale != 0.0)
      fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale), &lat);
    lattice_writer->Write(utt, lat);
  }
  KALDI_LOG << "Log-like per frame for utterance " << utt << " is "
            << (likelihood / num_frames) << " over "
            << num_frames << " frames.";
  KALDI_VLOG(2) << "Cost for utterance " << utt << " is "
                << weight.Value1() << " + " << weight.Value2();
  *like_ptr = likelihood;
  return true;
}



DecodeUtteranceLatticeFasterCtcClass::DecodeUtteranceLatticeFasterCtcClass(
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
    int32 *num_partial):  // If partial decode (final-state not reached), increments this.
    decoder_(decoder), decodable_(decodable),
    word_syms_(word_syms), utt_(utt), acoustic_scale_(acoustic_scale),
    determinize_(determinize), allow_partial_(allow_partial),
    alignments_writer_(alignments_writer),
    words_writer_(words_writer),
    compact_lattice_writer_(compact_lattice_writer),
    lattice_writer_(lattice_writer),
    like_sum_(like_sum), frame_sum_(frame_sum),
    num_done_(num_done), num_err_(num_err),
    num_partial_(num_partial),
    computed_(false), success_(false), partial_(false),
    clat_(NULL), lat_(NULL) { }


void DecodeUtteranceLatticeFasterCtcClass::operator () () {
  // Decoding and lattice determinization happens here.
  computed_ = true; // Just means this function was called-- a check on the
  // calling code.
  success_ = true;
  using fst::VectorFst;
  if (!decoder_->DecodeCtc(decodable_)) {
    KALDI_WARN << "Failed to decode file " << utt_;
    success_ = false;
  }
  if (!decoder_->ReachedFinal()) {
    if (allow_partial_) {
      KALDI_WARN << "Outputting partial output for utterance " << utt_
                 << " since no final-state reached\n";
      partial_ = true;
    } else {
      KALDI_WARN << "Not producing output for utterance " << utt_
                 << " since no final-state reached and "
                 << "--allow-partial=false.\n";
      success_ = false;
    }
  }
  if (!success_) return;

  // Get lattice, and do determinization if requested.
  lat_ = new Lattice;
  decoder_->GetRawLattice(lat_);
  if (lat_->NumStates() == 0)
    KALDI_ERR << "Unexpected problem getting lattice for utterance " << utt_;
  fst::Connect(lat_);
  if (determinize_) {
    clat_ = new CompactLattice;
    if (!DeterminizeLatticePhonePrunedCtcWrapper(
            lat_,
            decoder_->GetOptions().lattice_beam,
            clat_,
            decoder_->GetOptions().det_opts))
      KALDI_WARN << "Determinization finished earlier than the beam for "
                 << "utterance " << utt_;
    delete lat_;
    lat_ = NULL;
    // We'll write the lattice without acoustic scaling.
    if (acoustic_scale_ != 0.0)
      fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale_), clat_);
  } else {
    // We'll write the lattice without acoustic scaling.
    if (acoustic_scale_ != 0.0)
      fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale_), lat_);
  }
}

DecodeUtteranceLatticeFasterCtcClass::~DecodeUtteranceLatticeFasterCtcClass() {
  if (!computed_)
    KALDI_ERR << "Destructor called without operator (), error in calling code.";

  if (!success_) {
    if (num_err_ != NULL) (*num_err_)++;
  } else { // successful decode.
    // Getting the one-best output is lightweight enough that we can do it in
    // the destructor (easier than adding more variables to the class, and
    // will rarely slow down the main thread.)
    double likelihood;
    LatticeWeight weight;
    int32 num_frames;
    { // First do some stuff with word-level traceback...
      // This is basically for diagnostics.
      fst::VectorFst<LatticeArc> decoded;
      decoder_->GetBestPath(&decoded);
      if (decoded.NumStates() == 0) {
        // Shouldn't really reach this point as already checked success.
        KALDI_ERR << "Failed to get traceback for utterance " << utt_;
      }
      std::vector<int32> alignment;
      std::vector<int32> words;
      GetLinearSymbolSequence(decoded, &alignment, &words, &weight);
      num_frames = alignment.size();
      if (words_writer_->IsOpen())
        words_writer_->Write(utt_, words);
      if (alignments_writer_->IsOpen())
        alignments_writer_->Write(utt_, alignment);
      if (word_syms_ != NULL) {
        std::cerr << utt_ << ' ';
        for (size_t i = 0; i < words.size(); i++) {
          std::string s = word_syms_->Find(words[i]);
          if (s == "")
            KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
          std::cerr << s << ' ';
        }
        std::cerr << '\n';
      }
      likelihood = -(weight.Value1() + weight.Value2());
    }

    // Ouptut the lattices.
    if (determinize_) { // CompactLattice output.
      KALDI_ASSERT(compact_lattice_writer_ != NULL && clat_ != NULL);
      if (clat_->NumStates() == 0) {
        KALDI_WARN << "Empty lattice for utterance " << utt_;
      } else {
        compact_lattice_writer_->Write(utt_, *clat_);
      }
      delete clat_;
      clat_ = NULL;
    } else {
      KALDI_ASSERT(lattice_writer_ != NULL && lat_ != NULL);
      if (lat_->NumStates() == 0) {
        KALDI_WARN << "Empty lattice for utterance " << utt_;
      } else {
        lattice_writer_->Write(utt_, *lat_);
      }
      delete lat_;
      lat_ = NULL;
    }

    // Print out logging information.
    KALDI_LOG << "Log-like per frame for utterance " << utt_ << " is "
              << (likelihood / num_frames) << " over "
              << num_frames << " frames.";
    KALDI_VLOG(2) << "Cost for utterance " << utt_ << " is "
                  << weight.Value1() << " + " << weight.Value2();

    // Now output the various diagnostic variables.
    if (like_sum_ != NULL) *like_sum_ += likelihood;
    if (frame_sum_ != NULL) *frame_sum_ += num_frames;
    if (num_done_ != NULL) (*num_done_)++;
    if (partial_ && num_partial_ != NULL) (*num_partial_)++;
  }
  // We were given ownership of these two objects that were passed in in
  // the initializer.
  delete decoder_;
  delete decodable_;
}



} // end namespace kaldi.
