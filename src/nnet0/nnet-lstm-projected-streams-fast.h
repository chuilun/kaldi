// nnet0/nnet-lstm-projected-streams-fast.h

// Copyright 2014  Jiayu DU (Jerry), Wei Li
// Copyright 2015  Shanghai Jiao Tong University (author: Wei Deng)

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



#ifndef KALDI_NNET_NNET_LSTM_PROJECTED_STREAMS_FAST_H_
#define KALDI_NNET_NNET_LSTM_PROJECTED_STREAMS_FAST_H_

#include "nnet0/nnet-component.h"
#include "nnet0/nnet-utils.h"
#include "cudamatrix/cu-math.h"

/*************************************
 * x: input neuron
 * g: squashing neuron near input
 * i: Input gate
 * f: Forget gate
 * o: Output gate
 * c: memory Cell (CEC)
 * h: squashing neuron near output
 * m: output neuron of Memory block
 * r: recurrent projection neuron
 * y: output neuron of LSTMP
 *************************************/

namespace kaldi {

namespace lm {
class LmModelSync;
}

namespace nnet0 {

class LstmProjectedStreamsFast : public UpdatableComponent {
	friend class NnetModelSync;
	friend class lm::LmModelSync;
 public:
  LstmProjectedStreamsFast(int32 input_dim, int32 output_dim) :
    UpdatableComponent(input_dim, output_dim),
    ncell_(0),
    nrecur_(output_dim),
    nstream_(0),
    ntruncated_bptt_size_(0),
    clip_gradient_(0.0), clip_cell_(50.0),
	learn_rate_coef_(1.0), bias_learn_rate_coef_(1.0), max_norm_(0.0)
    //, dropout_rate_(0.0)
  { }

  ~LstmProjectedStreamsFast()
  { }

  Component* Copy() const { return new LstmProjectedStreamsFast(*this); }
  ComponentType GetType() const { return kLstmProjectedStreamsFast; }

  static void InitMatParam(CuMatrix<BaseFloat> &m, float scale) {
    m.SetRandUniform();  // uniform in [0, 1]
    m.Add(-0.5);         // uniform in [-0.5, 0.5]
    m.Scale(2 * scale);  // uniform in [-scale, +scale]
  }

  static void InitVecParam(CuVector<BaseFloat> &v, float scale) {
    Vector<BaseFloat> tmp(v.Dim());
    for (int i=0; i < tmp.Dim(); i++) {
      tmp(i) = (RandUniform() - 0.5) * 2 * scale;
    }
    v = tmp;
  }

  void InitData(std::istream &is) {
    // define options
    float param_scale = 0.02;
    float fgate_param_scale = param_scale;
    // parse config
    std::string token;
    while (!is.eof()) {
      ReadToken(is, false, &token);
      if (token == "<CellDim>")
        ReadBasicType(is, false, &ncell_);
      else if (token == "<ClipGradient>")
        ReadBasicType(is, false, &clip_gradient_);
      else if (token == "<ClipCell>")
        ReadBasicType(is, false, &clip_cell_);
      //else if (token == "<DropoutRate>")
      //  ReadBasicType(is, false, &dropout_rate_);
      else if (token == "<ParamScale>")
        ReadBasicType(is, false, &param_scale);
      else if (token == "<FgateBias>") ReadBasicType(is, false, &fgate_param_scale);
      else if (token == "<LearnRateCoef>") ReadBasicType(is, false, &learn_rate_coef_);
      else if (token == "<BiasLearnRateCoef>") ReadBasicType(is, false, &bias_learn_rate_coef_);
      else if (token == "<MaxNorm>") ReadBasicType(is, false, &max_norm_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
               << " (CellDim|ClipGradient|ParamScale)";
               //<< " (CellDim|ClipGradient|DropoutRate|ParamScale)";
      is >> std::ws;
    }

    // init weight and bias (Uniform)
    w_gifo_x_.Resize(4*ncell_, input_dim_, kUndefined);
    w_gifo_r_.Resize(4*ncell_, nrecur_, kUndefined);
    w_r_m_.Resize(nrecur_, ncell_, kUndefined);

    InitMatParam(w_gifo_x_, param_scale);
    InitMatParam(w_gifo_r_, param_scale);
    InitMatParam(w_r_m_, param_scale);

    bias_.Resize(4*ncell_, kUndefined);
    peephole_i_c_.Resize(ncell_, kUndefined);
    peephole_f_c_.Resize(ncell_, kUndefined);
    peephole_o_c_.Resize(ncell_, kUndefined);

    InitVecParam(bias_, param_scale);
    bias_.Range(2*ncell_,ncell_).Set(fgate_param_scale);

    InitVecParam(peephole_i_c_, param_scale);
    InitVecParam(peephole_f_c_, param_scale);
    InitVecParam(peephole_o_c_, param_scale);

    // init delta buffers
    w_gifo_x_corr_.Resize(4*ncell_, input_dim_, kSetZero);
    w_gifo_r_corr_.Resize(4*ncell_, nrecur_, kSetZero);
    bias_corr_.Resize(4*ncell_, kSetZero);

    peephole_i_c_corr_.Resize(ncell_, kSetZero);
    peephole_f_c_corr_.Resize(ncell_, kSetZero);
    peephole_o_c_corr_.Resize(ncell_, kSetZero);

    w_r_m_corr_.Resize(nrecur_, ncell_, kSetZero);

    KALDI_ASSERT(clip_gradient_ >= 0.0);
    KALDI_ASSERT(clip_cell_ >= 0.0);
  }

  void ReadData(std::istream &is, bool binary) {
    // optional learning-rate coefs
    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<LearnRateCoef>");
      ReadBasicType(is, binary, &learn_rate_coef_);
      ExpectToken(is, binary, "<BiasLearnRateCoef>");
      ReadBasicType(is, binary, &bias_learn_rate_coef_);
    }
    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<MaxNorm>");
      ReadBasicType(is, binary, &max_norm_);
    }

    ExpectToken(is, binary, "<CellDim>");
    ReadBasicType(is, binary, &ncell_);
    ExpectToken(is, binary, "<ClipGradient>");
    ReadBasicType(is, binary, &clip_gradient_);
    // optional cell activation cliping value
    if ('<' == Peek(is, binary)) {
        ExpectToken(is, binary, "<ClipCell>");
        ReadBasicType(is, binary, &clip_cell_);
    }
    //ExpectToken(is, binary, "<DropoutRate>");
    //ReadBasicType(is, binary, &dropout_rate_);

    w_gifo_x_.Read(is, binary);
    w_gifo_r_.Read(is, binary);
    bias_.Read(is, binary);

    peephole_i_c_.Read(is, binary);
    peephole_f_c_.Read(is, binary);
    peephole_o_c_.Read(is, binary);

    w_r_m_.Read(is, binary);

    // init delta buffers
    w_gifo_x_corr_.Resize(4*ncell_, input_dim_, kSetZero);
    w_gifo_r_corr_.Resize(4*ncell_, nrecur_, kSetZero);
    bias_corr_.Resize(4*ncell_, kSetZero);

    peephole_i_c_corr_.Resize(ncell_, kSetZero);
    peephole_f_c_corr_.Resize(ncell_, kSetZero);
    peephole_o_c_corr_.Resize(ncell_, kSetZero);

    w_r_m_corr_.Resize(nrecur_, ncell_, kSetZero);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<LearnRateCoef>");
    WriteBasicType(os, binary, learn_rate_coef_);
    WriteToken(os, binary, "<BiasLearnRateCoef>");
    WriteBasicType(os, binary, bias_learn_rate_coef_);
    WriteToken(os, binary, "<MaxNorm>");
    WriteBasicType(os, binary, max_norm_);

    WriteToken(os, binary, "<CellDim>");
    WriteBasicType(os, binary, ncell_);
    WriteToken(os, binary, "<ClipGradient>");
    WriteBasicType(os, binary, clip_gradient_);
    if (clip_cell_ != 50.0) {
        WriteToken(os, binary, "<ClipCell>");
        WriteBasicType(os, binary, clip_cell_);
    }
    //WriteToken(os, binary, "<DropoutRate>");
    //WriteBasicType(os, binary, dropout_rate_);

    w_gifo_x_.Write(os, binary);
    w_gifo_r_.Write(os, binary);
    bias_.Write(os, binary);

    peephole_i_c_.Write(os, binary);
    peephole_f_c_.Write(os, binary);
    peephole_o_c_.Write(os, binary);

    w_r_m_.Write(os, binary);
  }

  int32 NumParams() const {
    return ( w_gifo_x_.NumRows() * w_gifo_x_.NumCols() +
         w_gifo_r_.NumRows() * w_gifo_r_.NumCols() +
         bias_.Dim() +
         peephole_i_c_.Dim() +
         peephole_f_c_.Dim() +
         peephole_o_c_.Dim() +
         w_r_m_.NumRows() * w_r_m_.NumCols() );
  }

  int32 GetDim() const {
    return ( w_gifo_x_.SizeInBytes()/sizeof(BaseFloat) +
         w_gifo_r_.SizeInBytes()/sizeof(BaseFloat) +
         bias_.Dim() +
         peephole_i_c_.Dim() +
         peephole_f_c_.Dim() +
         peephole_o_c_.Dim() +
         w_r_m_.SizeInBytes()/sizeof(BaseFloat) );
  }

  void GetParams(Vector<BaseFloat>* wei_copy) const {
    wei_copy->Resize(NumParams());

    int32 offset, len;

    offset = 0;  len = w_gifo_x_.NumRows() * w_gifo_x_.NumCols();
    wei_copy->Range(offset, len).CopyRowsFromMat(w_gifo_x_);

    offset += len; len = w_gifo_r_.NumRows() * w_gifo_r_.NumCols();
    wei_copy->Range(offset, len).CopyRowsFromMat(w_gifo_r_);

    offset += len; len = bias_.Dim();
    wei_copy->Range(offset, len).CopyFromVec(bias_);

    offset += len; len = peephole_i_c_.Dim();
    wei_copy->Range(offset, len).CopyFromVec(peephole_i_c_);

    offset += len; len = peephole_f_c_.Dim();
    wei_copy->Range(offset, len).CopyFromVec(peephole_f_c_);

    offset += len; len = peephole_o_c_.Dim();
    wei_copy->Range(offset, len).CopyFromVec(peephole_o_c_);

    offset += len; len = w_r_m_.NumRows() * w_r_m_.NumCols();
    wei_copy->Range(offset, len).CopyRowsFromMat(w_r_m_);

    return;
  }

  std::string Info() const {
    return std::string("  ") +
      "\n  w_gifo_x_  "   + MomentStatistics(w_gifo_x_) +
      "\n  w_gifo_r_  "   + MomentStatistics(w_gifo_r_) +
      "\n  bias_  "     + MomentStatistics(bias_) +
      "\n  peephole_i_c_  " + MomentStatistics(peephole_i_c_) +
      "\n  peephole_f_c_  " + MomentStatistics(peephole_f_c_) +
      "\n  peephole_o_c_  " + MomentStatistics(peephole_o_c_) +
      "\n  w_r_m_  "    + MomentStatistics(w_r_m_);
  }

  std::string InfoGradient() const {
    // disassemble forward-propagation buffer into different neurons,
    const CuSubMatrix<BaseFloat> YG(propagate_buf_.ColRange(0*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> YI(propagate_buf_.ColRange(1*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> YF(propagate_buf_.ColRange(2*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> YO(propagate_buf_.ColRange(3*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> YC(propagate_buf_.ColRange(4*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> YH(propagate_buf_.ColRange(5*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> YM(propagate_buf_.ColRange(6*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> YR(propagate_buf_.ColRange(7*ncell_, nrecur_));

    // disassemble backpropagate buffer into different neurons,
    const CuSubMatrix<BaseFloat> DG(backpropagate_buf_.ColRange(0*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> DI(backpropagate_buf_.ColRange(1*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> DF(backpropagate_buf_.ColRange(2*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> DO(backpropagate_buf_.ColRange(3*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> DC(backpropagate_buf_.ColRange(4*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> DH(backpropagate_buf_.ColRange(5*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> DM(backpropagate_buf_.ColRange(6*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> DR(backpropagate_buf_.ColRange(7*ncell_, nrecur_));

    return std::string("  ") +
      "\n  Gradients:" +
      "\n  w_gifo_x_corr_  "   + MomentStatistics(w_gifo_x_corr_) +
      "\n  w_gifo_r_corr_  "   + MomentStatistics(w_gifo_r_corr_) +
      "\n  bias_corr_  "     + MomentStatistics(bias_corr_) +
      "\n  peephole_i_c_corr_  " + MomentStatistics(peephole_i_c_corr_) +
      "\n  peephole_f_c_corr_  " + MomentStatistics(peephole_f_c_corr_) +
      "\n  peephole_o_c_corr_  " + MomentStatistics(peephole_o_c_corr_) +
      "\n  w_r_m_corr_  "    + MomentStatistics(w_r_m_corr_) +
      "\n  Forward-pass:" +
      "\n  YG  " + MomentStatistics(YG) +
      "\n  YI  " + MomentStatistics(YI) +
      "\n  YF  " + MomentStatistics(YF) +
      "\n  YC  " + MomentStatistics(YC) +
      "\n  YH  " + MomentStatistics(YH) +
      "\n  YO  " + MomentStatistics(YO) +
      "\n  YM  " + MomentStatistics(YM) +
      "\n  YR  " + MomentStatistics(YR) +
      "\n  Backward-pass:" +
      "\n  DG  " + MomentStatistics(DG) +
      "\n  DI  " + MomentStatistics(DI) +
      "\n  DF  " + MomentStatistics(DF) +
      "\n  DC  " + MomentStatistics(DC) +
      "\n  DH  " + MomentStatistics(DH) +
      "\n  DO  " + MomentStatistics(DO) +
      "\n  DM  " + MomentStatistics(DM) +
      "\n  DR  " + MomentStatistics(DR);
  }

  void SetLstmContext(const Matrix<BaseFloat> &recurrent, const Matrix<BaseFloat> &cell) {
	  KALDI_ASSERT(nstream_ == recurrent.NumRows());
	  KALDI_ASSERT(nstream_ == cell.NumRows());

	  CuSubMatrix<BaseFloat> yr(prev_nnet_state_.ColRange(7*ncell_, nrecur_));
	  yr.CopyFromMat(recurrent);
	  CuSubMatrix<BaseFloat> yc(prev_nnet_state_.ColRange(4*ncell_, ncell_));
	  yc.CopyFromMat(cell);
  }

  void GetLstmContext(Matrix<BaseFloat> &recurrent, Matrix<BaseFloat> &cell) {
		KALDI_ASSERT(nstream_ == recurrent.NumRows());
		KALDI_ASSERT(nstream_ == cell.NumRows());

		CuSubMatrix<BaseFloat> yr(prev_nnet_state_.ColRange(7*ncell_, nrecur_));
		yr.CopyToMat(&recurrent);
		CuSubMatrix<BaseFloat> yc(prev_nnet_state_.ColRange(4*ncell_, ncell_));
		yc.CopyToMat(&cell);
  }

  void GetRCDim(int &r, int &c) {
        r = nrecur_;
        c = ncell_;
  }

  void UpdateLstmStreamsState(const std::vector<int32> &update_state_flag) {
	  KALDI_ASSERT(nstream_ == update_state_flag.size());
	  std::vector<int32> idx(nstream_);
	  for (int i = 0; i < nstream_; i++)
		  idx[i] = update_state_flag[i] == 1 ? i : -1;
	  keep_state_indices_.CopyFromVec(idx);
  }

  void ResetLstmStreams(const std::vector<int32> &stream_reset_flag, int32 ntruncated_bptt_size) {
    // allocate prev_nnet_state_ if not done yet,
    if (nstream_ != stream_reset_flag.size()) {
      // Karel: we just got number of streams! (before the 1st batch comes)
      nstream_ = stream_reset_flag.size();
      prev_nnet_state_.Resize(nstream_, 7*ncell_ + 1*nrecur_, kSetZero);
      KALDI_LOG << "Running training with " << nstream_ << " streams.";
    }
    // reset flag: 1 - reset stream network state
    KALDI_ASSERT(prev_nnet_state_.NumRows() == stream_reset_flag.size());
    for (int s = 0; s < stream_reset_flag.size(); s++) {
      if (stream_reset_flag[s] == 1) {
        prev_nnet_state_.Row(s).SetZero();
      }
    }

    if (ntruncated_bptt_size_ != ntruncated_bptt_size)
    {
    	ntruncated_bptt_size_ = ntruncated_bptt_size;
    	KALDI_LOG << "Backpropagate Truncated BPTT size: " << ntruncated_bptt_size_;
    }

  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    int DEBUG = 0;

    static bool do_stream_reset = false;
    if (nstream_ == 0) {
      do_stream_reset = true;
      nstream_ = 1; // Karel: we are in nnet-forward, so 1 stream,
      prev_nnet_state_.Resize(nstream_, 7*ncell_ + 1*nrecur_, kSetZero);
      KALDI_LOG << "Running nnet-forward with per-utterance LSTM-state reset";
    }
    if (do_stream_reset) prev_nnet_state_.SetZero();
    KALDI_ASSERT(nstream_ > 0);

    KALDI_ASSERT(in.NumRows() % nstream_ == 0);
    int32 T = in.NumRows() / nstream_;
    int32 S = nstream_;

    bool reset = propagate_buf_.NumRows() != (T+2)*S ? true:false;

    // 0:forward pass history, [1, T]:current sequence, T+1:dummy
    propagate_buf_.Resize((T+2)*S, 7 * ncell_ + nrecur_, kSetZero);
    propagate_buf_.RowRange(0*S,S).CopyFromMat(prev_nnet_state_);

    // disassemble entire neuron activation buffer into different neurons
    if (y_g.size() != T+2 || reset)
    {
    	Destroy();

    	YG = new CuSubMatrix<BaseFloat>(propagate_buf_.ColRange(0*ncell_, ncell_));
        YI = new CuSubMatrix<BaseFloat>(propagate_buf_.ColRange(1*ncell_, ncell_));
        YF = new CuSubMatrix<BaseFloat>(propagate_buf_.ColRange(2*ncell_, ncell_));
        YO = new CuSubMatrix<BaseFloat>(propagate_buf_.ColRange(3*ncell_, ncell_));
        YC = new CuSubMatrix<BaseFloat>(propagate_buf_.ColRange(4*ncell_, ncell_));
        YH = new CuSubMatrix<BaseFloat>(propagate_buf_.ColRange(5*ncell_, ncell_));
        YM = new CuSubMatrix<BaseFloat>(propagate_buf_.ColRange(6*ncell_, ncell_));
        YR = new CuSubMatrix<BaseFloat>(propagate_buf_.ColRange(7*ncell_, nrecur_));

        YGIFO = new CuSubMatrix<BaseFloat>(propagate_buf_.ColRange(0, 4*ncell_));

        y_g.resize(T+2);
        y_i.resize(T+2);
        y_f.resize(T+2);
        y_o.resize(T+2);
        y_c.resize(T+2);
        y_h.resize(T+2);
        y_m.resize(T+2);
        y_r.resize(T+2);
        y_gifo.resize(T+2);

        for (int t = 0; t <= T+1; t++)
        {
        	// multistream buffers for current time-step
        	y_g[t] = new CuSubMatrix<BaseFloat>(YG->RowRange(t*S,S));
        	y_i[t] = new CuSubMatrix<BaseFloat>(YI->RowRange(t*S,S));
        	y_f[t] = new CuSubMatrix<BaseFloat>(YF->RowRange(t*S,S));
        	y_o[t] = new CuSubMatrix<BaseFloat>(YO->RowRange(t*S,S));
        	y_c[t] = new CuSubMatrix<BaseFloat>(YC->RowRange(t*S,S));
        	y_h[t] = new CuSubMatrix<BaseFloat>(YH->RowRange(t*S,S));
        	y_m[t] = new CuSubMatrix<BaseFloat>(YM->RowRange(t*S,S));
        	y_r[t] = new CuSubMatrix<BaseFloat>(YR->RowRange(t*S,S));

        	y_gifo[t] = new CuSubMatrix<BaseFloat>(YGIFO->RowRange(t*S,S));

        }
    }


    // x -> g, i, f, o, not recurrent, do it all in once
    YGIFO->RowRange(1*S,T*S).AddMatMat(1.0, in, kNoTrans, w_gifo_x_, kTrans, 0.0);
    //// LSTM forward dropout
    //// Google paper 2014: Recurrent Neural Network Regularization
    //// by Wojciech Zaremba, Ilya Sutskever, Oriol Vinyals
    //if (dropout_rate_ != 0.0) {
    //  dropout_mask_.Resize(in.NumRows(), 4*ncell_, kUndefined);
    //  dropout_mask_.SetRandUniform();   // [0,1]
    //  dropout_mask_.Add(-dropout_rate_);  // [-dropout_rate, 1-dropout_rate_],
    //  dropout_mask_.ApplyHeaviside();   // -tive -> 0.0, +tive -> 1.0
    //  YGIFO.RowRange(1*S,T*S).MulElements(dropout_mask_);
    //}

    // bias -> g, i, f, o
    YGIFO->RowRange(1*S,T*S).AddVecToRows(1.0, bias_);

    for (int t = 1; t <= T; t++) {

      // r(t-1) -> g, i, f, o
      y_gifo[t]->AddMatMat(1.0, *y_r[t-1], kNoTrans, w_gifo_r_, kTrans,  1.0);

      // c(t-1) -> i(t) via peephole
      y_i[t]->AddMatDiagVec(1.0, *y_c[t-1], kNoTrans, peephole_i_c_, 1.0);

      // c(t-1) -> f(t) via peephole
      y_f[t]->AddMatDiagVec(1.0, *y_c[t-1], kNoTrans, peephole_f_c_, 1.0);

      // i, f sigmoid squashing
      y_i[t]->Sigmoid(*y_i[t]);
      y_f[t]->Sigmoid(*y_f[t]);

      // g tanh squashing
      y_g[t]->Tanh(*y_g[t]);

      // g -> c
      y_c[t]->AddMatMatElements(1.0, *y_g[t], *y_i[t], 0.0);

      // c(t-1) -> c(t) via forget-gate
      y_c[t]->AddMatMatElements(1.0, *y_c[t-1], *y_f[t], 1.0);

      y_c[t]->ApplyFloor(-clip_cell_);   // optional clipping of cell activation
      y_c[t]->ApplyCeiling(clip_cell_);  // google paper Interspeech2014: LSTM for LVCSR

      // h tanh squashing
      y_h[t]->Tanh(*y_c[t]);

      // c(t) -> o(t) via peephole (non-recurrent) & o squashing
      y_o[t]->AddMatDiagVec(1.0, *y_c[t], kNoTrans, peephole_o_c_, 1.0);

      // o sigmoid squashing
      y_o[t]->Sigmoid(*y_o[t]);

      // h -> m via output gate
      y_m[t]->AddMatMatElements(1.0, *y_h[t], *y_o[t], 0.0);

      // m -> r
      y_r[t]->AddMatMat(1.0, *y_m[t], kNoTrans, w_r_m_, kTrans, 0.0);

      if (DEBUG) {
        std::cerr << "forward-pass frame " << t << "\n";
        std::cerr << "activation of g: " << *y_g[t];
        std::cerr << "activation of i: " << *y_i[t];
        std::cerr << "activation of f: " << *y_f[t];
        std::cerr << "activation of o: " << *y_o[t];
        std::cerr << "activation of c: " << *y_c[t];
        std::cerr << "activation of h: " << *y_h[t];
        std::cerr << "activation of m: " << *y_m[t];
        std::cerr << "activation of r: " << *y_r[t];
      }
    }

    // recurrent projection layer is also feed-forward as LSTM output
    out->CopyFromMat(YR->RowRange(1*S,T*S));

    // now the last frame state becomes previous network state for next batch
    if (keep_state_indices_.Dim() != nstream_)
    	prev_nnet_state_.CopyFromMat(propagate_buf_.RowRange(T*S,S));
    else
    	prev_nnet_state_.CopyRows(propagate_buf_.RowRange(T*S,S), keep_state_indices_);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
              const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {

    int DEBUG = 0;
    float bptt = 1.0;

    int32 T = in.NumRows() / nstream_;
    int32 S = nstream_;

    bool reset = backpropagate_buf_.NumRows() != (T+2)*S ? true:false;

    // 0:dummy, [1,T] frames, T+1 backward pass history
    backpropagate_buf_.Resize((T+2)*S, 7 * ncell_ + nrecur_, kSetZero);

    if (d_g.size() != T+2 || reset)
    {
        // disassemble backpropagate buffer into neurons
        DG = new CuSubMatrix<BaseFloat>(backpropagate_buf_.ColRange(0*ncell_, ncell_));
        DI = new CuSubMatrix<BaseFloat>(backpropagate_buf_.ColRange(1*ncell_, ncell_));
        DF = new CuSubMatrix<BaseFloat>(backpropagate_buf_.ColRange(2*ncell_, ncell_));
        DO = new CuSubMatrix<BaseFloat>(backpropagate_buf_.ColRange(3*ncell_, ncell_));
        DC = new CuSubMatrix<BaseFloat>(backpropagate_buf_.ColRange(4*ncell_, ncell_));
        DH = new CuSubMatrix<BaseFloat>(backpropagate_buf_.ColRange(5*ncell_, ncell_));
        DM = new CuSubMatrix<BaseFloat>(backpropagate_buf_.ColRange(6*ncell_, ncell_));
        DR = new CuSubMatrix<BaseFloat>(backpropagate_buf_.ColRange(7*ncell_, nrecur_));

        DGIFO = new CuSubMatrix<BaseFloat>(backpropagate_buf_.ColRange(0, 4*ncell_));

        d_g.resize(T+2);
        d_i.resize(T+2);
        d_f.resize(T+2);
        d_o.resize(T+2);
        d_c.resize(T+2);
        d_h.resize(T+2);
        d_m.resize(T+2);
        d_r.resize(T+2);
        d_gifo.resize(T+2);

        for (int t = T+1; t >= 0; t--)
        {
        	// multistream buffers for current time-step
        	d_g[t] = new CuSubMatrix<BaseFloat>(DG->RowRange(t*S,S));
        	d_i[t] = new CuSubMatrix<BaseFloat>(DI->RowRange(t*S,S));
        	d_f[t] = new CuSubMatrix<BaseFloat>(DF->RowRange(t*S,S));
        	d_o[t] = new CuSubMatrix<BaseFloat>(DO->RowRange(t*S,S));
        	d_c[t] = new CuSubMatrix<BaseFloat>(DC->RowRange(t*S,S));
        	d_h[t] = new CuSubMatrix<BaseFloat>(DH->RowRange(t*S,S));
        	d_m[t] = new CuSubMatrix<BaseFloat>(DM->RowRange(t*S,S));
        	d_r[t] = new CuSubMatrix<BaseFloat>(DR->RowRange(t*S,S));

        	d_gifo[t] = new CuSubMatrix<BaseFloat>(DGIFO->RowRange(t*S,S));

        }
    }

    // projection layer to LSTM output is not recurrent, so backprop it all in once
    DR->RowRange(1*S,T*S).CopyFromMat(out_diff);

    for (int t = T; t >= 1; t--) {

      if (ntruncated_bptt_size_ > 0)
    	  bptt = (t<T && (T-t)%ntruncated_bptt_size_==0) ? 0.0 : 1.0;
    	  //bptt = t%ntruncated_bptt_size_==0 ? 0.0 : 1.0;
      // r
      //   Version 1 (precise gradients):
      //   backprop error from g(t+1), i(t+1), f(t+1), o(t+1) to r(t)
      if (ntruncated_bptt_size_ >= 0)
      d_r[t]->AddMatMat(bptt, *d_gifo[t+1], kNoTrans, w_gifo_r_, kNoTrans, 1.0);

      /*
      //   Version 2 (Alex Graves' PhD dissertation):
      //   only backprop g(t+1) to r(t)
      CuSubMatrix<BaseFloat> w_g_r_(w_gifo_r_.RowRange(0, ncell_));
      d_r[t]->AddMatMat(bptt, *d_g[t+1], kNoTrans, w_g_r_, kNoTrans, 1.0);
      */

      /*
      //   Version 3 (Felix Gers' PhD dissertation):
      //   truncate gradients of g(t+1), i(t+1), f(t+1), o(t+1) once they leak out memory block
      //   CEC(with forget connection) is the only "error-bridge" through time
      */

      // r -> m
      d_m[t]->AddMatMat(1.0, *d_r[t], kNoTrans, w_r_m_, kNoTrans, 0.0);

      // m -> h via output gate
      d_h[t]->AddMatMatElements(1.0, *d_m[t], *y_o[t], 0.0);
      d_h[t]->DiffTanh(*y_h[t], *d_h[t]);

      // o
      d_o[t]->AddMatMatElements(1.0, *d_m[t], *y_h[t], 0.0);
      d_o[t]->DiffSigmoid(*y_o[t], *d_o[t]);

      // c
      // 1. diff from h(t)
      // 2. diff from c(t+1) (via forget-gate between CEC)
      // 3. diff from i(t+1) (via peephole)
      // 4. diff from f(t+1) (via peephole)
      // 5. diff from o(t)   (via peephole, not recurrent)
      d_c[t]->AddMat(1.0, *d_h[t]);
      d_c[t]->AddMatMatElements(bptt, *d_c[t+1], *y_f[t+1], 1.0);
      d_c[t]->AddMatDiagVec(bptt, *d_i[t+1], kNoTrans, peephole_i_c_, 1.0);
      d_c[t]->AddMatDiagVec(bptt, *d_f[t+1], kNoTrans, peephole_f_c_, 1.0);
      d_c[t]->AddMatDiagVec(1.0, *d_o[t]  , kNoTrans, peephole_o_c_, 1.0);

      // f
      d_f[t]->AddMatMatElements(1.0, *d_c[t], *y_c[t-1], 0.0);
      d_f[t]->DiffSigmoid(*y_f[t], *d_f[t]);

      // i
      d_i[t]->AddMatMatElements(1.0, *d_c[t], *y_g[t], 0.0);
      d_i[t]->DiffSigmoid(*y_i[t], *d_i[t]);

      // c -> g via input gate
      d_g[t]->AddMatMatElements(1.0, *d_c[t], *y_i[t], 0.0);
      d_g[t]->DiffTanh(*y_g[t], *d_g[t]);

      // debug info
      if (DEBUG) {
        std::cerr << "backward-pass frame " << t << "\n";
        std::cerr << "derivative wrt input r " << *d_r[t];
        std::cerr << "derivative wrt input m " << *d_m[t];
        std::cerr << "derivative wrt input h " << *d_h[t];
        std::cerr << "derivative wrt input o " << *d_o[t];
        std::cerr << "derivative wrt input c " << *d_c[t];
        std::cerr << "derivative wrt input f " << *d_f[t];
        std::cerr << "derivative wrt input i " << *d_i[t];
        std::cerr << "derivative wrt input g " << *d_g[t];
      }
    }

    // g,i,f,o -> x, do it all in once
    in_diff->AddMatMat(1.0, DGIFO->RowRange(1*S,T*S), kNoTrans, w_gifo_x_, kNoTrans, 0.0);

  }

  void Gradient(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff)
  {

	    // we use following hyperparameters from the option class
	    const BaseFloat lr = opts_.learn_rate * learn_rate_coef_;
	    //const BaseFloat lr_bias = opts_.learn_rate * bias_learn_rate_coef_;
	    //const BaseFloat mmt = opts_.momentum;
	    const BaseFloat l2 = opts_.l2_penalty;
	    //const BaseFloat l1 = opts_.l1_penalty;
	    // we will also need the number of frames in the mini-batch
	    const int32 num_frames = input.NumRows();

	    int DEBUG = 0;

	    int32 T = input.NumRows() / nstream_;
	    int32 S = nstream_;

	    //// backward pass dropout
	    //if (dropout_rate_ != 0.0) {
	    //  in_diff->MulElements(dropout_mask_);
	    //}

	    // calculate delta
	    const BaseFloat mmt = opts_.momentum;

	    // weight x -> g, i, f, o
	    w_gifo_x_corr_.AddMatMat(1.0, DGIFO->RowRange(1*S,T*S), kTrans,
	                                  input                  , kNoTrans, mmt);
	    // recurrent weight r -> g, i, f, o
	    w_gifo_r_corr_.AddMatMat(1.0, DGIFO->RowRange(1*S,T*S), kTrans,
	                                  YR->RowRange(0*S,T*S)   , kNoTrans, mmt);
	    // bias of g, i, f, o
	    bias_corr_.AddRowSumMat(1.0, DGIFO->RowRange(1*S,T*S), mmt);

	    // recurrent peephole c -> i
	    peephole_i_c_corr_.AddDiagMatMat(1.0, DI->RowRange(1*S,T*S), kTrans,
	                                          YC->RowRange(0*S,T*S), kNoTrans, mmt);
	    // recurrent peephole c -> f
	    peephole_f_c_corr_.AddDiagMatMat(1.0, DF->RowRange(1*S,T*S), kTrans,
	                                          YC->RowRange(0*S,T*S), kNoTrans, mmt);
	    // peephole c -> o
	    peephole_o_c_corr_.AddDiagMatMat(1.0, DO->RowRange(1*S,T*S), kTrans,
	                                          YC->RowRange(1*S,T*S), kNoTrans, mmt);

	    w_r_m_corr_.AddMatMat(1.0, DR->RowRange(1*S,T*S), kTrans,
	                               YM->RowRange(1*S,T*S), kNoTrans, mmt);

	    if (clip_gradient_ > 0.0) {
	      w_gifo_x_corr_.ApplyFloor(-clip_gradient_);
	      w_gifo_x_corr_.ApplyCeiling(clip_gradient_);
	      w_gifo_r_corr_.ApplyFloor(-clip_gradient_);
	      w_gifo_r_corr_.ApplyCeiling(clip_gradient_);
	      bias_corr_.ApplyFloor(-clip_gradient_);
	      bias_corr_.ApplyCeiling(clip_gradient_);
	      w_r_m_corr_.ApplyFloor(-clip_gradient_);
	      w_r_m_corr_.ApplyCeiling(clip_gradient_);
	      peephole_i_c_corr_.ApplyFloor(-clip_gradient_);
	      peephole_i_c_corr_.ApplyCeiling(clip_gradient_);
	      peephole_f_c_corr_.ApplyFloor(-clip_gradient_);
	      peephole_f_c_corr_.ApplyCeiling(clip_gradient_);
	      peephole_o_c_corr_.ApplyFloor(-clip_gradient_);
	      peephole_o_c_corr_.ApplyCeiling(clip_gradient_);
	    }

	    if (DEBUG) {
	      std::cerr << "gradients(with optional momentum): \n";
	      std::cerr << "w_gifo_x_corr_ " << w_gifo_x_corr_;
	      std::cerr << "w_gifo_r_corr_ " << w_gifo_r_corr_;
	      std::cerr << "bias_corr_ " << bias_corr_;
	      std::cerr << "w_r_m_corr_ " << w_r_m_corr_;
	      std::cerr << "peephole_i_c_corr_ " << peephole_i_c_corr_;
	      std::cerr << "peephole_f_c_corr_ " << peephole_f_c_corr_;
	      std::cerr << "peephole_o_c_corr_ " << peephole_o_c_corr_;
	    }

	    // l2 regularization
	    if (l2 != 0.0) {
	    	w_gifo_x_.AddMat(-lr*l2*num_frames, w_gifo_x_);
	    	w_gifo_r_.AddMat(-lr*l2*num_frames, w_gifo_r_);
	    	bias_.AddVec(-lr*l2*num_frames, bias_);

	    	peephole_i_c_.AddVec(-lr*l2*num_frames, peephole_i_c_);
	    	peephole_f_c_.AddVec(-lr*l2*num_frames, peephole_f_c_);
	    	peephole_o_c_.AddVec(-lr*l2*num_frames, peephole_o_c_);

	    	w_r_m_.AddMat(-lr*l2*num_frames, w_r_m_);
	    }

	    if (y_g.size() != T+2)
	    	Destroy();
  }

  void UpdateGradient()
  {
	    const BaseFloat lr = opts_.learn_rate * learn_rate_coef_;
        const BaseFloat lr_bias = opts_.learn_rate * bias_learn_rate_coef_;

	    w_gifo_x_.AddMat(-lr, w_gifo_x_corr_);
	    w_gifo_r_.AddMat(-lr, w_gifo_r_corr_);
	    bias_.AddVec(-lr_bias, bias_corr_, 1.0);

	    peephole_i_c_.AddVec(-lr, peephole_i_c_corr_, 1.0);
	    peephole_f_c_.AddVec(-lr, peephole_f_c_corr_, 1.0);
	    peephole_o_c_.AddVec(-lr, peephole_o_c_corr_, 1.0);

	    w_r_m_.AddMat(-lr, w_r_m_corr_);

        if (clip_cell_ > 0.0) {
          w_gifo_x_.ApplyFloor(-clip_cell_);
          w_gifo_x_.ApplyCeiling(clip_cell_);
          w_gifo_r_.ApplyFloor(-clip_cell_);
          w_gifo_r_.ApplyCeiling(clip_cell_);
          bias_.ApplyFloor(-clip_cell_);
          bias_.ApplyCeiling(clip_cell_);
          w_r_m_.ApplyFloor(-clip_cell_);
          w_r_m_.ApplyCeiling(clip_cell_);
          peephole_i_c_.ApplyFloor(-clip_cell_);
          peephole_i_c_.ApplyCeiling(clip_cell_);
          peephole_f_c_.ApplyFloor(-clip_cell_);
          peephole_f_c_.ApplyCeiling(clip_cell_);
          peephole_o_c_.ApplyFloor(-clip_cell_);
          peephole_o_c_.ApplyCeiling(clip_cell_);
        }
  }

  void ResetGradient()
  {
      w_gifo_x_corr_.SetZero();
      w_gifo_r_corr_.SetZero();
      bias_corr_.SetZero();
      peephole_i_c_corr_.SetZero();
      peephole_f_c_corr_.SetZero();
      peephole_o_c_corr_.SetZero();
      w_r_m_corr_.SetZero();
  }


  int WeightCopy(void *host, int direction, int copykind)
  {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
        Timer tim;

        int32 dst_pitch, src_pitch, width,  size;
        int pos = 0;
        void *src, *dst;
        MatrixDim dim;
        cudaMemcpyKind kind;
        switch(copykind)
        {
            case 0:
                kind = cudaMemcpyHostToHost;
                break;
            case 1:
                kind = cudaMemcpyHostToDevice;
                break;
            case 2:
                kind = cudaMemcpyDeviceToHost;
                break;
            case 3:
                kind = cudaMemcpyDeviceToDevice;
                break;
            default:
                KALDI_ERR << "Default based unified address space";
                break;
        }

        dim = w_gifo_x_.Dim();
		src_pitch = dim.stride*sizeof(BaseFloat);
		dst_pitch = src_pitch;
		width = dim.cols*sizeof(BaseFloat);
		dst = (void*) (direction==0 ? ((char *)host+pos) : (char *)w_gifo_x_.Data());
		src = (void*) (direction==0 ? (char *)w_gifo_x_.Data() : ((char *)host+pos));
		cudaMemcpy2D(dst, dst_pitch, src, src_pitch, width, dim.rows, kind);
		pos += w_gifo_x_.SizeInBytes();

		dim = w_gifo_r_.Dim();
		src_pitch = dim.stride*sizeof(BaseFloat);
		dst_pitch = src_pitch;
		width = dim.cols*sizeof(BaseFloat);
		dst = (void*) (direction==0 ? ((char *)host+pos) : (char *)w_gifo_r_.Data());
		src = (void*) (direction==0 ? (char *)w_gifo_r_.Data() : ((char *)host+pos));
		cudaMemcpy2D(dst, dst_pitch, src, src_pitch, width, dim.rows, kind);
		pos += w_gifo_r_.SizeInBytes();

		size = bias_.Dim()*sizeof(BaseFloat);
		dst = (void*) (direction==0 ? ((char *)host+pos) : (char *)bias_.Data());
		src = (void*) (direction==0 ? (char *)bias_.Data() : ((char *)host+pos));
		cudaMemcpy(dst, src, size, kind);
		pos += size;

		size = peephole_i_c_.Dim()*sizeof(BaseFloat);
		dst = (void*) (direction==0 ? ((char *)host+pos) : (char *)peephole_i_c_.Data());
		src = (void*) (direction==0 ? (char *)peephole_i_c_.Data() : ((char *)host+pos));
		cudaMemcpy(dst, src, size, kind);
		pos += size;

		size = peephole_f_c_.Dim()*sizeof(BaseFloat);
		dst = (void*) (direction==0 ? ((char *)host+pos) : (char *)peephole_f_c_.Data());
		src = (void*) (direction==0 ? (char *)peephole_f_c_.Data() : ((char *)host+pos));
		cudaMemcpy(dst, src, size, kind);
		pos += size;

		size = peephole_o_c_.Dim()*sizeof(BaseFloat);
		dst = (void*) (direction==0 ? ((char *)host+pos) : (char *)peephole_o_c_.Data());
		src = (void*) (direction==0 ? (char *)peephole_o_c_.Data() : ((char *)host+pos));
		cudaMemcpy(dst, src, size, kind);
		pos += size;

		dim = w_r_m_.Dim();
		src_pitch = dim.stride*sizeof(BaseFloat);
		dst_pitch = src_pitch;
		width = dim.cols*sizeof(BaseFloat);
		dst = (void*) (direction==0 ? ((char *)host+pos) : (char *)w_r_m_.Data());
		src = (void*) (direction==0 ? (char *)w_r_m_.Data() : ((char *)host+pos));
		cudaMemcpy2D(dst, dst_pitch, src, src_pitch, width, dim.rows, kind);
		pos += w_r_m_.SizeInBytes();


  	  CU_SAFE_CALL(cudaGetLastError());

  	  CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  	  return pos;
  }else
#endif
  	{
  		// not implemented for CPU yet
  		return 0;
  	}
  }

  void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
    const BaseFloat lr  = opts_.learn_rate;

    int DEBUG = 0;

    int32 T = input.NumRows() / nstream_;
    int32 S = nstream_;

    //// backward pass dropout
    //if (dropout_rate_ != 0.0) {
    //  in_diff->MulElements(dropout_mask_);
    //}

    // calculate delta
    const BaseFloat mmt = opts_.momentum;

    // weight x -> g, i, f, o
    w_gifo_x_corr_.AddMatMat(1.0, DGIFO->RowRange(1*S,T*S), kTrans,
                                  input                     , kNoTrans, mmt);
    // recurrent weight r -> g, i, f, o
    w_gifo_r_corr_.AddMatMat(1.0, DGIFO->RowRange(1*S,T*S), kTrans,
                                  YR->RowRange(0*S,T*S)   , kNoTrans, mmt);
    // bias of g, i, f, o
    bias_corr_.AddRowSumMat(1.0, DGIFO->RowRange(1*S,T*S), mmt);

    // recurrent peephole c -> i
    peephole_i_c_corr_.AddDiagMatMat(1.0, DI->RowRange(1*S,T*S), kTrans,
                                          YC->RowRange(0*S,T*S), kNoTrans, mmt);
    // recurrent peephole c -> f
    peephole_f_c_corr_.AddDiagMatMat(1.0, DF->RowRange(1*S,T*S), kTrans,
                                          YC->RowRange(0*S,T*S), kNoTrans, mmt);
    // peephole c -> o
    peephole_o_c_corr_.AddDiagMatMat(1.0, DO->RowRange(1*S,T*S), kTrans,
                                          YC->RowRange(1*S,T*S), kNoTrans, mmt);

    w_r_m_corr_.AddMatMat(1.0, DR->RowRange(1*S,T*S), kTrans,
                               YM->RowRange(1*S,T*S), kNoTrans, mmt);

    if (clip_gradient_ > 0.0) {
      w_gifo_x_corr_.ApplyFloor(-clip_gradient_);
      w_gifo_x_corr_.ApplyCeiling(clip_gradient_);
      w_gifo_r_corr_.ApplyFloor(-clip_gradient_);
      w_gifo_r_corr_.ApplyCeiling(clip_gradient_);
      bias_corr_.ApplyFloor(-clip_gradient_);
      bias_corr_.ApplyCeiling(clip_gradient_);
      w_r_m_corr_.ApplyFloor(-clip_gradient_);
      w_r_m_corr_.ApplyCeiling(clip_gradient_);
      peephole_i_c_corr_.ApplyFloor(-clip_gradient_);
      peephole_i_c_corr_.ApplyCeiling(clip_gradient_);
      peephole_f_c_corr_.ApplyFloor(-clip_gradient_);
      peephole_f_c_corr_.ApplyCeiling(clip_gradient_);
      peephole_o_c_corr_.ApplyFloor(-clip_gradient_);
      peephole_o_c_corr_.ApplyCeiling(clip_gradient_);
    }

    if (DEBUG) {
      std::cerr << "gradients(with optional momentum): \n";
      std::cerr << "w_gifo_x_corr_ " << w_gifo_x_corr_;
      std::cerr << "w_gifo_r_corr_ " << w_gifo_r_corr_;
      std::cerr << "bias_corr_ " << bias_corr_;
      std::cerr << "w_r_m_corr_ " << w_r_m_corr_;
      std::cerr << "peephole_i_c_corr_ " << peephole_i_c_corr_;
      std::cerr << "peephole_f_c_corr_ " << peephole_f_c_corr_;
      std::cerr << "peephole_o_c_corr_ " << peephole_o_c_corr_;
    }

    w_gifo_x_.AddMat(-lr, w_gifo_x_corr_);
    w_gifo_r_.AddMat(-lr, w_gifo_r_corr_);
    bias_.AddVec(-lr, bias_corr_, 1.0);

    peephole_i_c_.AddVec(-lr, peephole_i_c_corr_, 1.0);
    peephole_f_c_.AddVec(-lr, peephole_f_c_corr_, 1.0);
    peephole_o_c_.AddVec(-lr, peephole_o_c_corr_, 1.0);

    w_r_m_.AddMat(-lr, w_r_m_corr_);

//    /*
//      Here we deal with the famous "vanishing & exploding difficulties" in RNN learning.
//
//      *For gradients vanishing*
//      LSTM architecture introduces linear CEC as the "error bridge" across long time distance
//      solving vanishing problem.
//
//      *For gradients exploding*
//      LSTM is still vulnerable to gradients explosing in BPTT(with large weight & deep time expension).
//      To prevent this, we tried L2 regularization, which didn't work well
//
//      Our approach is a *modified* version of Max Norm Regularization:
//      For each nonlinear neuron,
//      1. fan-in weights & bias model a seperation hyper-plane: W x + b = 0
//      2. squashing function models a differentiable nonlinear slope around this hyper-plane.
//
//      Conventional max norm regularization scale W to keep its L2 norm bounded,
//      As a modification, we scale down large (W & b) *simultaneously*, this:
//      1. keeps all fan-in weights small, prevents gradients from exploding during backward-pass.
//      2. keeps the location of the hyper-plane unchanged, so we don't wipe out already learned knowledge.
//      3. shrinks the "normal" of the hyper-plane, smooths the nonlinear slope, improves generalization.
//      4. makes the network *well-conditioned* (weights are constrained in a reasonible range).
//
//      We've observed faster convergence and performance gain by doing this.
//    */
//
//    int DEBUG = 0;
//    BaseFloat max_norm = 1.0;   // weights with large L2 norm may cause exploding in deep BPTT expensions
//                  // TODO: move this config to opts_
//    CuMatrix<BaseFloat> L2_gifo_x(w_gifo_x_);
//    CuMatrix<BaseFloat> L2_gifo_r(w_gifo_r_);
//    L2_gifo_x.MulElements(w_gifo_x_);
//    L2_gifo_r.MulElements(w_gifo_r_);
//
//    CuVector<BaseFloat> L2_norm_gifo(L2_gifo_x.NumRows());
//    L2_norm_gifo.AddColSumMat(1.0, L2_gifo_x, 0.0);
//    L2_norm_gifo.AddColSumMat(1.0, L2_gifo_r, 1.0);
//    L2_norm_gifo.Range(1*ncell_, ncell_).AddVecVec(1.0, peephole_i_c_, peephole_i_c_, 1.0);
//    L2_norm_gifo.Range(2*ncell_, ncell_).AddVecVec(1.0, peephole_f_c_, peephole_f_c_, 1.0);
//    L2_norm_gifo.Range(3*ncell_, ncell_).AddVecVec(1.0, peephole_o_c_, peephole_o_c_, 1.0);
//    L2_norm_gifo.ApplyPow(0.5);
//
//    CuVector<BaseFloat> shrink(L2_norm_gifo);
//    shrink.Scale(1.0/max_norm);
//    shrink.ApplyFloor(1.0);
//    shrink.InvertElements();
//
//    w_gifo_x_.MulRowsVec(shrink);
//    w_gifo_r_.MulRowsVec(shrink);
//    bias_.MulElements(shrink);
//
//    peephole_i_c_.MulElements(shrink.Range(1*ncell_, ncell_));
//    peephole_f_c_.MulElements(shrink.Range(2*ncell_, ncell_));
//    peephole_o_c_.MulElements(shrink.Range(3*ncell_, ncell_));
//
//    if (DEBUG) {
//      if (shrink.Min() < 0.95) {   // we dont want too many trivial logs here
//        std::cerr << "gifo shrinking coefs: " << shrink;
//      }
//    }
//
  }

  void Destroy()
  {

	  //delete[] y_g.front();

	  if (y_g.size() > 0)
	  {
		  for (int t = 0; t< y_g.size(); t++)
		  {
			  delete y_g[t]; delete y_i[t]; delete y_f[t]; delete y_o[t];
			  delete y_c[t]; delete y_h[t]; delete y_m[t]; delete y_r[t]; delete y_gifo[t];
		  }
		  delete YG; delete YI; delete YF; delete YO; delete YC; delete YH; delete YM; delete YR; delete YGIFO;
		  y_g.resize(0);
	  }

	  if (d_g.size() > 0)
	  {
		  for (int t = 0; t< d_g.size(); t++)
		  {
			  delete d_g[t]; delete d_i[t]; delete d_f[t]; delete d_o[t];
			  delete d_c[t]; delete d_h[t]; delete d_m[t]; delete d_r[t]; delete d_gifo[t];
		  }
		  delete DG, delete DI; delete DF; delete DO; delete DC; delete DH; delete DM; delete DR; delete DGIFO;
		  d_g.resize(0);
	  }
		/*
		  delete y_g[t], y_i[t], y_f[t], y_o[t],
	  	  	  	  	  y_c[t], y_h[t], y_m[t], y_r[t], y_gifo[t];
		  delete d_g[t], d_i[t], d_f[t], d_o[t],
		  	  	  	  d_c[t], d_h[t], d_m[t], d_r[t], d_gifo[t];
		*/
  }

 protected:
  // dims
  int32 ncell_;
  int32 nrecur_;  ///< recurrent projection layer dim
  int32 nstream_;
  int32 ntruncated_bptt_size_;

  CuMatrix<BaseFloat> prev_nnet_state_;
  CuArray<MatrixIndexT> keep_state_indices_;

  // gradient-clipping value,
  BaseFloat clip_gradient_;
  // cell activation clipping value,
  BaseFloat clip_cell_;

  // non-recurrent dropout
  //BaseFloat dropout_rate_;
  //CuMatrix<BaseFloat> dropout_mask_;

  // feed-forward connections: from x to [g, i, f, o]
  CuMatrix<BaseFloat> w_gifo_x_;
  CuMatrix<BaseFloat> w_gifo_x_corr_;

  // recurrent projection connections: from r to [g, i, f, o]
  CuMatrix<BaseFloat> w_gifo_r_;
  CuMatrix<BaseFloat> w_gifo_r_corr_;

  // biases of [g, i, f, o]
  CuVector<BaseFloat> bias_;
  CuVector<BaseFloat> bias_corr_;

  // peephole from c to i, f, g
  // peephole connections are block-internal, so we use vector form
  CuVector<BaseFloat> peephole_i_c_;
  CuVector<BaseFloat> peephole_f_c_;
  CuVector<BaseFloat> peephole_o_c_;

  CuVector<BaseFloat> peephole_i_c_corr_;
  CuVector<BaseFloat> peephole_f_c_corr_;
  CuVector<BaseFloat> peephole_o_c_corr_;

  // projection layer r: from m to r
  CuMatrix<BaseFloat> w_r_m_;
  CuMatrix<BaseFloat> w_r_m_fix_;
  CuMatrix<BaseFloat> w_r_m_corr_;

  // propagate buffer: output of [g, i, f, o, c, h, m, r]
  CuMatrix<BaseFloat> propagate_buf_;

  // back-propagate buffer: diff-input of [g, i, f, o, c, h, m, r]
  CuMatrix<BaseFloat> backpropagate_buf_;

  std::vector<CuSubMatrix<BaseFloat>* > y_g, y_i, y_f, y_o,
  	  	  	  	  	  	  	  	  	  	  y_c, y_h, y_m, y_r, y_gifo;
  std::vector<CuSubMatrix<BaseFloat>* > d_g, d_i, d_f, d_o,
    	  	  	  	  	  	  	  	  	  	  d_c, d_h, d_m, d_r, d_gifo;

  CuSubMatrix<BaseFloat> *YG, *YI, *YF, *YO,
  	  	  	  	  	  	  	  	  	  	  *YC, *YH, *YM, *YR, *YGIFO;

  CuSubMatrix<BaseFloat> *DG, *DI, *DF, *DO,
   	  	  	  	  	  	  	  	  	  	  *DC, *DH, *DM, *DR, *DGIFO;

  BaseFloat learn_rate_coef_;
  BaseFloat bias_learn_rate_coef_;
  BaseFloat max_norm_;

};
} // namespace nnet0
} // namespace kaldi

#endif
