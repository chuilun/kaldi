// online0/online-nnet-faster-decoder.cc

// Copyright 2012 Cisco Systems (author: Matthias Paulik)
// Copyright 2015-2016   Shanghai Jiao Tong University (author: Wei Deng)

//   Modifications to the original contribution by Cisco Systems made by:
//   Vassil Panayotov

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
#include "online0/online-nnet-faster-decoder.h"
#include "fstext/fstext-utils.h"
#include "hmm/hmm-utils.h"

namespace kaldi {

void OnlineNnetFasterDecoder::ResetDecoder(bool full) {
  ClearToks(toks_.Clear());
  StateId start_state = fst_.Start();
  KALDI_ASSERT(start_state != fst::kNoStateId);
  Arc dummy_arc(0, 0, Weight::One(), start_state);
  Token *dummy_token = new Token(dummy_arc, NULL);
  toks_.Insert(start_state, dummy_token);
  prev_immortal_tok_ = immortal_tok_ = dummy_token;
  utt_frames_ = 0;
  if (full)
    frame_ = 0;
  state_ = kStartFeats;
}


void
OnlineNnetFasterDecoder::MakeLattice(const Token *start,
                                 const Token *end,
                                 fst::MutableFst<LatticeArc> *out_fst) const {
  out_fst->DeleteStates();
  if (start == NULL) return;
  bool is_final = false;
  double this_cost = start->cost_ + fst_.Final(start->arc_.nextstate).Value();
  if (this_cost != std::numeric_limits<double>::infinity())
    is_final = true;
  std::vector<LatticeArc> arcs_reverse;  // arcs in reverse order.
  for (const Token *tok = start; tok != end; tok = tok->prev_) {
    BaseFloat tot_cost = tok->cost_ -
        (tok->prev_ ? tok->prev_->cost_ : 0.0),
        graph_cost = tok->arc_.weight.Value(),
        ac_cost = tot_cost - graph_cost;
    LatticeArc l_arc(tok->arc_.ilabel,
                     tok->arc_.olabel,
                     LatticeWeight(graph_cost, ac_cost),
                     tok->arc_.nextstate);
    arcs_reverse.push_back(l_arc);
  }
  if(arcs_reverse.back().nextstate == fst_.Start()) {
    arcs_reverse.pop_back();  // that was a "fake" token... gives no info.
  }
  StateId cur_state = out_fst->AddState();
  out_fst->SetStart(cur_state);
  for (ssize_t i = static_cast<ssize_t>(arcs_reverse.size())-1; i >= 0; i--) {
    LatticeArc arc = arcs_reverse[i];
    arc.nextstate = out_fst->AddState();
    out_fst->AddArc(cur_state, arc);
    cur_state = arc.nextstate;
  }
  if (is_final) {
    Weight final_weight = fst_.Final(start->arc_.nextstate);
    out_fst->SetFinal(cur_state, LatticeWeight(final_weight.Value(), 0.0));
  } else {
    out_fst->SetFinal(cur_state, LatticeWeight::One());
  }
  RemoveEpsLocal(out_fst);
}


void OnlineNnetFasterDecoder::UpdateImmortalToken() {
  unordered_set<Token*> emitting;
  for (const Elem *e = toks_.GetList(); e != NULL; e = e->tail) {
    Token* tok = e->val;
    while (tok->arc_.ilabel == 0) //deal with non-emitting ones ...
      tok = tok->prev_;
    if (tok != NULL)
      emitting.insert(tok);
  }
  Token* the_one = NULL;
  while (1) {
    if (emitting.size() == 1) {
      the_one = *(emitting.begin());
      break;
    }
    if (emitting.size() == 0)
      break;
    unordered_set<Token*> prev_emitting;
    unordered_set<Token*>::iterator it;
    for (it = emitting.begin(); it != emitting.end(); ++it) {
      Token* tok = *it;
      Token* prev_token = tok->prev_;
      while ((prev_token != NULL) && (prev_token->arc_.ilabel == 0))
        prev_token = prev_token->prev_; //deal with non-emitting ones
      if (prev_token == NULL)
        continue;
      prev_emitting.insert(prev_token);
    } // for
    emitting = prev_emitting;
  } // while
  if (the_one != NULL) {
    prev_immortal_tok_ = immortal_tok_;
    immortal_tok_ = the_one;
    return;
  }
}


bool
OnlineNnetFasterDecoder::PartialTraceback(fst::MutableFst<LatticeArc> *out_fst) {
  UpdateImmortalToken();
  if(immortal_tok_ == prev_immortal_tok_)
    return false; //no partial traceback at that point of time
  MakeLattice(immortal_tok_, prev_immortal_tok_, out_fst);
  return true;
}


void
OnlineNnetFasterDecoder::FinishTraceBack(fst::MutableFst<LatticeArc> *out_fst) {
  Token *best_tok = NULL;
  bool is_final = ReachedFinal();
  if (!is_final) {
    for (const Elem *e = toks_.GetList(); e != NULL; e = e->tail)
      if (best_tok == NULL || *best_tok < *(e->val) )
        best_tok = e->val;
  } else {
    double best_cost = std::numeric_limits<double>::infinity();
    for (const Elem *e = toks_.GetList(); e != NULL; e = e->tail) {
      double this_cost = e->val->cost_ + fst_.Final(e->key).Value();
      if (this_cost != std::numeric_limits<double>::infinity() &&
          this_cost < best_cost) {
        best_cost = this_cost;
        best_tok = e->val;
      }
    }
  }
  MakeLattice(best_tok, immortal_tok_, out_fst);
}


void
OnlineNnetFasterDecoder::TracebackNFrames(int32 nframes,
                                      fst::MutableFst<LatticeArc> *out_fst) {
  Token *best_tok = NULL;
  for (const Elem *e = toks_.GetList(); e != NULL; e = e->tail)
    if (best_tok == NULL || *best_tok < *(e->val) )
      best_tok = e->val;
  if (best_tok == NULL) {
    out_fst->DeleteStates();
    return;
  }

  bool is_final = false;
  double this_cost = best_tok->cost_ +
      fst_.Final(best_tok->arc_.nextstate).Value();
                             
  if (this_cost != std::numeric_limits<double>::infinity())
    is_final = true;
  std::vector<LatticeArc> arcs_reverse;  // arcs in reverse order.
  for (Token *tok = best_tok; (tok != NULL) && (nframes > 0); tok = tok->prev_) {
    if (tok->arc_.ilabel != 0) // count only the non-epsilon arcs
      --nframes;
    BaseFloat tot_cost = tok->cost_ -
        (tok->prev_ ? tok->prev_->cost_ : 0.0);
    BaseFloat graph_cost = tok->arc_.weight.Value();
    BaseFloat ac_cost = tot_cost - graph_cost;
    LatticeArc larc(tok->arc_.ilabel,
                     tok->arc_.olabel,
                     LatticeWeight(graph_cost, ac_cost),
                     tok->arc_.nextstate);
    arcs_reverse.push_back(larc);
  }
  if(arcs_reverse.back().nextstate == fst_.Start())
    arcs_reverse.pop_back();  // that was a "fake" token... gives no info.
  StateId cur_state = out_fst->AddState();
  out_fst->SetStart(cur_state);
  for (ssize_t i = static_cast<ssize_t>(arcs_reverse.size())-1; i >= 0; i--) {
    LatticeArc arc = arcs_reverse[i];
    arc.nextstate = out_fst->AddState();
    out_fst->AddArc(cur_state, arc);
    cur_state = arc.nextstate;
  }
  if (is_final) {
    Weight final_weight = fst_.Final(best_tok->arc_.nextstate);
    out_fst->SetFinal(cur_state, LatticeWeight(final_weight.Value(), 0.0));
  } else {
    out_fst->SetFinal(cur_state, LatticeWeight::One());
  }
  RemoveEpsLocal(out_fst);
}


OnlineNnetFasterDecoder::DecodeState
OnlineNnetFasterDecoder::Decode(DecodableInterface *decodable) {
  if (state_ == kEndFeats) // new utterance
    return state_;

  ProcessNonemitting(std::numeric_limits<float>::max());

  int32 batch_frame = 0;
  Timer timer;
  double64 tstart = timer.Elapsed(), tstart_batch = tstart;
  BaseFloat factor = -1;
  for (; frame_ < decodable->NumFramesReady() && batch_frame < opts_.batch_size;
		  ++frame_, ++utt_frames_, ++batch_frame) {
    if (batch_frame != 0 && (batch_frame % opts_.update_interval) == 0) {
      // adjust the beam if needed
      BaseFloat tend = timer.Elapsed();
      BaseFloat elapsed = (tend - tstart) * 1000;
      // warning: hardcoded 10ms frames assumption!
      factor = elapsed / (opts_.rt_max * opts_.update_interval * 10);
      BaseFloat min_factor = (opts_.rt_min / opts_.rt_max);
      if (factor > 1 || factor < min_factor) {
        BaseFloat update_factor = (factor > 1)?
            -std::min(opts_.beam_update * factor, opts_.max_beam_update):
             std::min(opts_.beam_update / factor, opts_.max_beam_update);
        effective_beam_ += effective_beam_ * update_factor;
        effective_beam_ = std::min(effective_beam_, max_beam_);
      }
      tstart = tend;
    }

    if (batch_frame != 0 && (frame_ % 200) == 0)
      // one log message at every 2 seconds assuming 10ms frames
      KALDI_VLOG(3) << "Beam: " << effective_beam_
          << "; Speed: "
          << ((timer.Elapsed() - tstart_batch) * 1000) / (batch_frame*10)
          << " xRT";

    BaseFloat weight_cutoff;
    if (opts_.cutoff == "hybrid")
    		weight_cutoff = ProcessEmitting(decodable);
    else
    		weight_cutoff = ProcessEmittingCtc(decodable);

    ProcessNonemitting(weight_cutoff);
  }

  state_ = kEndBatch;
  if (decodable->IsLastFrame(frame_ - 1))
	  state_ = kEndFeats;

  return state_;
}

} // namespace kaldi
