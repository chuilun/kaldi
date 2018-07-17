// nnet0/nnet-gru-streams.h

// Copyright 2015  xutao

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



#ifndef KALDI_NNET_NNET_GRU_PROJECTED_STREAMS_H_
#define KALDI_NNET_NNET_GRU_PROJECTED_STREAMS_H_

#include "nnet0/nnet-component.h"
#include "nnet0/nnet-utils.h"
#include "cudamatrix/cu-math.h"

/*************************************
 * x: input neuron
 * r: input gate
 * z: reset gate
 * c: cell
 * h: squashing neuron near output
 * p: recurrent projection neuron
 * y: output neuron of GRU
 *************************************/

namespace kaldi {
namespace nnet0 {

class GruProjectedStreams : public UpdatableComponent {
	friend class NnetModelSync;
 public:
  GruProjectedStreams(int32 input_dim, int32 output_dim) :
    UpdatableComponent(input_dim, output_dim),
    ncell_(0),nrecur_(output_dim),
    nstream_(0),
    ntruncated_bptt_size_(0),
    clip_gradient_(0.0),
    learn_rate_coef_(1.0), bias_learn_rate_coef_(1.0), max_norm_(0.0)
  { }

  ~GruProjectedStreams()
  { }

  Component* Copy() const { return new GruProjectedStreams(*this); }
  ComponentType GetType() const { return kGruProjectedStreams; }

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
    // parse config
    std::string token;
    while (!is.eof()) {
      ReadToken(is, false, &token);
      if (token == "<CellDim>")
        ReadBasicType(is, false, &ncell_);
      else if (token == "<ClipGradient>")
        ReadBasicType(is, false, &clip_gradient_);
      //else if (token == "<DropoutRate>")
      //  ReadBasicType(is, false, &dropout_rate_);
      else if (token == "<ParamScale>")
        ReadBasicType(is, false, &param_scale);
      else if (token == "<LearnRateCoef>") ReadBasicType(is, false, &learn_rate_coef_);
      else if (token == "<BiasLearnRateCoef>") ReadBasicType(is, false, &bias_learn_rate_coef_);
      else if (token == "<MaxNorm>") ReadBasicType(is, false, &max_norm_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
               << " (CellDim|ClipGradient|ParamScale)";
               //<< " (CellDim|ClipGradient|DropoutRate|ParamScale)";
      is >> std::ws;
    }

    // init weight and bias (Uniform)
    w_rzc_x_.Resize(3*ncell_, input_dim_, kUndefined);
    w_rz_r_.Resize(2*ncell_, nrecur_, kUndefined);
    w_c_r_.Resize(ncell_, ncell_, kUndefined);
    w_p_m_.Resize(nrecur_, ncell_, kUndefined);
    InitMatParam(w_rzc_x_, param_scale);
    InitMatParam(w_rz_r_, param_scale);
    InitMatParam(w_c_r_, param_scale) ;
    InitMatParam(w_p_m_, param_scale);

    bias_.Resize(3*ncell_, kUndefined);
    InitVecParam(bias_, param_scale);
    CuSubVector<BaseFloat> rz_gate(bias_.Range(0, 2*ncell_));
    rz_gate.Set(1.0);

    // init delta buffers
    w_rzc_x_corr_.Resize(3*ncell_, input_dim_, kSetZero);
    w_rz_r_corr_.Resize(2*ncell_, nrecur_, kSetZero) ;
    w_c_r_corr_.Resize(ncell_, ncell_, kSetZero) ;
    w_p_m_corr_.Resize(nrecur_, ncell_, kSetZero) ;
    bias_corr_.Resize(3*ncell_, kSetZero);

    KALDI_ASSERT(clip_gradient_ >= 0.0);
  }

  void ReadData(std::istream &is, bool binary) {
    ExpectToken(is, binary, "<CellDim>");
    ReadBasicType(is, binary, &ncell_);
    ExpectToken(is, binary, "<ClipGradient>");
    ReadBasicType(is, binary, &clip_gradient_);
    //ExpectToken(is, binary, "<DropoutRate>");
    //ReadBasicType(is, binary, &dropout_rate_);

    w_rzc_x_.Read(is, binary);
    w_rz_r_.Read(is, binary);
    w_c_r_.Read(is, binary);
    w_p_m_.Read(is, binary);
    bias_.Read(is, binary);


    // init delta buffers
    w_rzc_x_corr_.Resize(3*ncell_, input_dim_, kSetZero);
    w_rz_r_corr_.Resize(2*ncell_, nrecur_, kSetZero);
    w_c_r_corr_.Resize(ncell_, ncell_, kSetZero) ;
    w_p_m_corr_.Resize(nrecur_, ncell_, kSetZero);
    bias_corr_.Resize(3*ncell_, kSetZero);

  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<CellDim>");
    WriteBasicType(os, binary, ncell_);
    WriteToken(os, binary, "<ClipGradient>");
    WriteBasicType(os, binary, clip_gradient_);
    //WriteToken(os, binary, "<DropoutRate>");
    //WriteBasicType(os, binary, dropout_rate_);

    w_rzc_x_.Write(os, binary);
    w_rz_r_.Write(os, binary);
    w_c_r_.Write(os, binary) ;
    w_p_m_.Write(os, binary);
    bias_.Write(os, binary);

  }

  int32 NumParams() const {
    return ( w_rzc_x_.NumRows() * w_rzc_x_.NumCols() +
         w_rz_r_.NumRows() * w_rz_r_.NumCols() +
         w_c_r_.NumRows() * w_c_r_.NumCols() +
        w_p_m_.NumRows() * w_p_m_.NumCols() +
         bias_.Dim() ) ;
  }

  void GetParams(Vector<BaseFloat>* wei_copy) const {
    wei_copy->Resize(NumParams());

    int32 offset, len;

    offset = 0;  len = w_rzc_x_.NumRows() * w_rzc_x_.NumCols();
    wei_copy->Range(offset, len).CopyRowsFromMat(w_rzc_x_);

    offset += len; len = w_rz_r_.NumRows() * w_rz_r_.NumCols();
    wei_copy->Range(offset, len).CopyRowsFromMat(w_rz_r_);

    offset += len ; len = w_c_r_.NumRows() * w_c_r_.NumCols() ;
    wei_copy->Range(offset, len).CopyRowsFromMat(w_c_r_);

    offset += len ; len = w_p_m_.NumRows() * w_p_m_.NumCols() ;
    wei_copy->Range(offset, len).CopyRowsFromMat(w_p_m_);

    offset += len; len = bias_.Dim();
    wei_copy->Range(offset, len).CopyFromVec(bias_);

    return;
  }

  std::string Info() const {
    return std::string("  ") +
      "\n  w_rzc_x_  "   + MomentStatistics(w_rzc_x_) +
      "\n  w_rz_r_  "   + MomentStatistics(w_rz_r_) +
      "\n  w_c_r_ " + MomentStatistics(w_c_r_)+
      "\n  w_p_m_" + MomentStatistics(w_p_m_)+
      "\n  bias_  "     + MomentStatistics(bias_) ;
  }

  std::string InfoGradient() const {
    // disassemble forward-propagation buffer into different neurons,
    const CuSubMatrix<BaseFloat> YR(propagate_buf_.ColRange(0*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> YZ(propagate_buf_.ColRange(1*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> YC(propagate_buf_.ColRange(2*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> YH(propagate_buf_.ColRange(3*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> YP(propagate_buf_.ColRange(4*ncell_, nrecur_));
    // disassemble backpropagate buffer into different neurons,
    const CuSubMatrix<BaseFloat> DR(backpropagate_buf_.ColRange(0*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> DZ(backpropagate_buf_.ColRange(1*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> DC(backpropagate_buf_.ColRange(2*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> DH(backpropagate_buf_.ColRange(3*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> DP(backpropagate_buf_.ColRange(4*ncell_, nrecur_));
    return std::string("  ") +
      "\n  Gradients:" +
      "\n  w_rzc_x_corr_  "   + MomentStatistics(w_rzc_x_corr_) +
      "\n  w_rz_r_corr_  "   + MomentStatistics(w_rz_r_corr_) +
      "\n  w_c_r_corr_"  + MomentStatistics(w_c_r_corr_) +
      "\n  w_p_m_corr_" + MomentStatistics(w_p_m_corr_)+
      "\n  bias_corr_  "     + MomentStatistics(bias_corr_) +
      "\n  Forward-pass:" +
      "\n  YR  " + MomentStatistics(YR) +
      "\n  YZ  " + MomentStatistics(YZ) +
      "\n  YC  " + MomentStatistics(YC) +
      "\n  YH  " + MomentStatistics(YH) +
      "\n  YP  " + MomentStatistics(YP) +
      "\n  Backward-pass:" +
      "\n  DR  " + MomentStatistics(DR) +
      "\n  DZ  " + MomentStatistics(DZ) +
      "\n  DC  " + MomentStatistics(DC) +
      "\n  DH  " + MomentStatistics(DH) +
      "\n  DP  " + MomentStatistics(DP) ;
  }

  void ResetGRUProjectedStreams(const std::vector<int32> &stream_reset_flag, int32 ntruncated_bptt_size) {
    // allocate prev_nnet_state_ if not done yet,
    if (nstream_ == 0) {
      // Karel: we just got number of streams! (before the 1st batch comes)
      nstream_ = stream_reset_flag.size();
      prev_nnet_state_.Resize(nstream_, 5*ncell_ , kSetZero);
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
      prev_nnet_state_.Resize(nstream_, 5*ncell_, kSetZero);
      KALDI_LOG << "Running nnet-forward with per-utterance GRU-state reset";
    }
    if (do_stream_reset) prev_nnet_state_.SetZero();
    KALDI_ASSERT(nstream_ > 0);

    KALDI_ASSERT(in.NumRows() % nstream_ == 0);
    int32 T = in.NumRows() / nstream_;
    int32 S  = nstream_;
    // 0:forward pass history, [1, T]:current sequence, T+1:dummy
    propagate_buf_.Resize((T+2)*S, 5 * ncell_ , kSetZero);
    propagate_buf_.RowRange(0*S,S).CopyFromMat(prev_nnet_state_);
    YRP_.Resize((T+2)*S, ncell_, kSetZero);
    DRP_.Resize((T+2)*S, ncell_, kSetZero);
    // disassemble entire neuron activation buffer into different neurons
    CuSubMatrix<BaseFloat> YR(propagate_buf_.ColRange(0*ncell_, ncell_));
    CuSubMatrix<BaseFloat> YZ(propagate_buf_.ColRange(1*ncell_, ncell_));
    CuSubMatrix<BaseFloat> YC(propagate_buf_.ColRange(2*ncell_, ncell_));
    CuSubMatrix<BaseFloat> YH(propagate_buf_.ColRange(3*ncell_, ncell_));
    CuSubMatrix<BaseFloat> YP(propagate_buf_.ColRange(4*ncell_, nrecur_));

    CuSubMatrix<BaseFloat> YRZC(propagate_buf_.ColRange(0, 3*ncell_));
    CuSubMatrix<BaseFloat> YRZ(propagate_buf_.ColRange(0,2*ncell_));

    // x -> r, z, c not recurrent, do it all in once
    YRZC.RowRange(1*S,T*S).AddMatMat(1.0, in, kNoTrans, w_rzc_x_, kTrans, 0.0);
    // bias -> r, z, c
    YRZC.RowRange(1*S,T*S).AddVecToRows(1.0, bias_);

    for (int t = 1; t <= T; t++) {
      // multistream buffers for current time-step
      CuSubMatrix<BaseFloat> y_r(YR.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_z(YZ.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_c(YC.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_h(YH.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_p(YP.RowRange(t*S,S));

      CuSubMatrix<BaseFloat> y_rz(YRZ.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_rp(YRP_.RowRange(t*S,S));

      // p(t-1) -> r(t), z(t)
      y_rz.AddMatMat(1.0, YP.RowRange((t-1)*S,S), kNoTrans, w_rz_r_, kTrans,  1.0);
      // i, f sigmoid squashing
      y_rz.Sigmoid(y_rz);
      // r(t),p(t-1) -> rp(t)
      y_rp.AddMatMatElements(1.0, y_r, YH.RowRange((t-1)*S, S), 0.0);
      //rp -> c
      y_c.AddMatMat(1.0, y_rp, kNoTrans, w_c_r_, kTrans, 1.0);
      y_c.Tanh(y_c);

      y_h.AddMatMatElements(1.0, y_z, YH.RowRange((t-1)*S, S), 0.0);
      y_h.AddMatMatElements( -1.0, y_z, y_c, 1.0 );
      y_h.AddMat(1.0, y_c) ;
      //h->p
      y_p.AddMatMat(1.0, y_h, kNoTrans, w_p_m_, kTrans, 0.0);
      if (DEBUG) {
        std::cerr << "forward-pass frame " << t << "\n";
        std::cerr << "activation of r: " << y_r;
        std::cerr << "activation of z: " << y_z;
        std::cerr << "activation of c: " << y_c;
        std::cerr << "activation of h: " << y_h;
        std::cerr << "activation of p: " << y_p;
      }
    }
    out->CopyFromMat(YP.RowRange(1*S,T*S));

    // now the last frame state becomes previous network state for next batch
    prev_nnet_state_.CopyFromMat(propagate_buf_.RowRange(T*S,S));
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
              const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {

    int DEBUG = 0;

    int32 T = in.NumRows() / nstream_;
    int32 S = nstream_;

    // disassemble propagated buffer into neurons
    CuSubMatrix<BaseFloat> YR(propagate_buf_.ColRange(0*ncell_, ncell_));
    CuSubMatrix<BaseFloat> YZ(propagate_buf_.ColRange(1*ncell_, ncell_));
    CuSubMatrix<BaseFloat> YC(propagate_buf_.ColRange(2*ncell_, ncell_));
    CuSubMatrix<BaseFloat> YH(propagate_buf_.ColRange(3*ncell_, ncell_));
    CuSubMatrix<BaseFloat> YP(propagate_buf_.ColRange(4*ncell_, nrecur_));

    CuSubMatrix<BaseFloat> YRZC(propagate_buf_.ColRange(0, 3*ncell_));
    CuSubMatrix<BaseFloat> YRZ(propagate_buf_.ColRange(0,2*ncell_));

    // 0:dummy, [1,T] frames, T+1 backward pass history
    backpropagate_buf_.Resize((T+2)*S, 5 * ncell_ , kSetZero);

    // disassemble backpropagate buffer into neurons
    CuSubMatrix<BaseFloat> DR(backpropagate_buf_.ColRange(0*ncell_, ncell_));
    CuSubMatrix<BaseFloat> DZ(backpropagate_buf_.ColRange(1*ncell_, ncell_));
    CuSubMatrix<BaseFloat> DC(backpropagate_buf_.ColRange(2*ncell_, ncell_));
    CuSubMatrix<BaseFloat> DH(backpropagate_buf_.ColRange(3*ncell_, ncell_));
    CuSubMatrix<BaseFloat> DP(backpropagate_buf_.ColRange(4*ncell_,nrecur_));
    CuSubMatrix<BaseFloat> DRZC(backpropagate_buf_.ColRange(0, 3*ncell_));
    CuSubMatrix<BaseFloat> DRZ(backpropagate_buf_.ColRange(0,2*ncell_));

    // projection layer to GRU output is not recurrent, so backprop it all in once
    DP.RowRange(1*S,T*S).CopyFromMat(out_diff);

    for (int t = T; t >= 1; t--) {
      CuSubMatrix<BaseFloat> y_r(YR.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_z(YZ.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_c(YC.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_h(YH.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_p(YP.RowRange(t*S,S));

      CuSubMatrix<BaseFloat> d_r(DR.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> d_z(DZ.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> d_c(DC.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> d_h(DH.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> d_p(DP.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> d_rp(DRP_.RowRange(t*S,S));

      //   backprop error from r(t+1), z(t+1)  to p(t)
      d_p.AddMatMat(1.0, DRZ.RowRange((t+1)*S,S), kNoTrans, w_rz_r_, kNoTrans, 1.0);
      //   backprop error from c(t+1), h(t+1), p(t) to h(t)
      d_h.AddMatMat(1.0, d_p, kNoTrans, w_p_m_, kNoTrans, 0.0);
      d_h.AddMatMatElements(1.0, YZ.RowRange((t+1)*S, S), DH.RowRange((t+1)*S,S), 1.0);
      
      DRP_.RowRange((t+1)*S,S).AddMatMat(1.0, DC.RowRange((t+1)*S,S), kNoTrans, w_c_r_, kNoTrans, 0.0);
      DRP_.RowRange((t+1)*S,S).MulElements(YR.RowRange((t+1)*S,S)) ;

      d_h.AddMat(1.0, DRP_.RowRange((t+1)*S,S)) ;

      //h -> z
      d_z.AddMatMatElements(1.0, YH.RowRange((t-1)*S, S), d_h, 0.0) ;
      d_z.AddMatMatElements(-1.0, y_c, d_h, 1.0);
      d_z.DiffSigmoid(y_z, d_z) ;

      //h -> c
      //edition1
      d_c.AddMatMatElements(-1.0, y_z, d_h, 0.0);
      d_c.AddMat(1.0, d_h);
      d_c.DiffTanh(y_c, d_c);
      //c -> r
      d_rp.AddMatMat(1.0, d_c, kNoTrans, w_c_r_, kNoTrans, 0.0);
      d_r.AddMatMatElements(1.0, YH.RowRange((t-1)*S,S) , d_rp, 0.0);
      d_r.DiffSigmoid(y_r, d_r) ;
      
      // debug info
      if (DEBUG) {
        std::cerr << "backward-pass frame " << t << "\n";
        std::cerr << "derivative wrt input r " << d_r;
        std::cerr << "derivative wrt input z " << d_z;
        std::cerr << "derivative wrt input c " << d_c;
        std::cerr << "derivative wrt input h " << d_h;
      }
    }

    // r , z , c -> x, do it all in once
    in_diff->AddMatMat(1.0, DRZC.RowRange(1*S,T*S), kNoTrans, w_rzc_x_, kNoTrans, 0.0);

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

	    // disassemble propagated buffer into neurons
	    CuSubMatrix<BaseFloat> YR(propagate_buf_.ColRange(0*ncell_, ncell_));
	    CuSubMatrix<BaseFloat> YZ(propagate_buf_.ColRange(1*ncell_, ncell_));
	    CuSubMatrix<BaseFloat> YC(propagate_buf_.ColRange(2*ncell_, ncell_));
	    CuSubMatrix<BaseFloat> YH(propagate_buf_.ColRange(3*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YP(propagate_buf_.ColRange(4*ncell_, nrecur_));

	    CuSubMatrix<BaseFloat> YRZC(propagate_buf_.ColRange(0, 3*ncell_));
	    CuSubMatrix<BaseFloat> YRZ(propagate_buf_.ColRange(0,2*ncell_));

	    // disassemble backpropagate buffer into neurons
	    CuSubMatrix<BaseFloat> DR(backpropagate_buf_.ColRange(0*ncell_, ncell_));
	    CuSubMatrix<BaseFloat> DZ(backpropagate_buf_.ColRange(1*ncell_, ncell_));
	    CuSubMatrix<BaseFloat> DC(backpropagate_buf_.ColRange(2*ncell_, ncell_));
	    CuSubMatrix<BaseFloat> DH(backpropagate_buf_.ColRange(3*ncell_, ncell_));
        CuSubMatrix<BaseFloat> DP(backpropagate_buf_.ColRange(4*ncell_, nrecur_));

	    CuSubMatrix<BaseFloat> DRZC(backpropagate_buf_.ColRange(0, 3*ncell_));
	    CuSubMatrix<BaseFloat> DRZ(backpropagate_buf_.ColRange(0,2*ncell_));


	    // calculate delta
	    const BaseFloat mmt = opts_.momentum;

	    // weight x -> r , z , c
	    w_rzc_x_corr_.AddMatMat(1.0, DRZC.RowRange(1*S,T*S), kTrans,
	                                  input                     , kNoTrans, mmt);
	    // recurrent weight h -> r , z
	    w_rz_r_corr_.AddMatMat(1.0, DRZ.RowRange(1*S,T*S), kTrans,
	                                  YP.RowRange(0*S,T*S)   , kNoTrans, mmt );
	    // recurrent weight h -> c
	    w_c_r_corr_.AddMatMat(1.0, DC.RowRange(1*S,T*S), kTrans,
	                                  YRP_.RowRange(1*S,T*S), kNoTrans, mmt);

        // recurrent weight h -> p
        w_p_m_corr_.AddMatMat(1.0, DP.RowRange(1*S,T*S), kTrans,
                                        YH.RowRange(1*S,T*S), kNoTrans, mmt);
	    // bias of r,z,c
	    bias_corr_.AddRowSumMat(1.0, DRZC.RowRange(1*S,T*S), mmt);


	    if (clip_gradient_ > 0.0) {

	      w_rzc_x_corr_.ApplyFloor(-clip_gradient_);
	      w_rzc_x_corr_.ApplyCeiling(clip_gradient_);
	      w_rz_r_corr_.ApplyFloor(-clip_gradient_);
	      w_rz_r_corr_.ApplyCeiling(clip_gradient_);
	      w_c_r_corr_.ApplyFloor(-clip_gradient_);
	      w_c_r_corr_.ApplyCeiling(clip_gradient_);
          w_p_m_corr_.ApplyFloor(-clip_gradient_);
          w_p_m_corr_.ApplyCeiling(clip_gradient_);
	      bias_corr_.ApplyFloor(-clip_gradient_);
	      bias_corr_.ApplyCeiling(clip_gradient_);

	    }

	    if (DEBUG) {
	      std::cerr << "gradients(with optional momentum): \n";
	      std::cerr << "w_rzc_x_corr_ " << w_rzc_x_corr_;
	      std::cerr << "w_rz_r_corr_ " << w_rz_r_corr_;
	      std::cerr << "bias_corr_ " << bias_corr_;
	      std::cerr << "w_c_r_corr_ " << w_c_r_corr_;

	    }

		 // l2 regularization
		 if (l2 != 0.0) {
			 w_rzc_x_.AddMat(-lr*l2*num_frames, w_rzc_x_);
			 w_rz_r_.AddMat(-lr*l2*num_frames, w_rz_r_);
			 w_c_r_.AddMat(-lr*l2*num_frames, w_c_r_);
             w_p_m_.AddMat(-lr*l2*num_frames,w_p_m_);
		 	//bias_.AddVec(-lr*l2*num_frames, bias_);
		 }

  }

  void UpdateGradient()
  {
	    const BaseFloat lr = opts_.learn_rate * learn_rate_coef_;
        const BaseFloat lr_bias = opts_.learn_rate * bias_learn_rate_coef_;

        w_rzc_x_.AddMat(-lr, w_rzc_x_corr_);
        w_rz_r_.AddMat(-lr, w_rz_r_corr_);
        w_c_r_.AddMat(-lr, w_c_r_corr_);
        w_p_m_.AddMat(-lr, w_p_m_corr_);
        bias_.AddVec(-lr_bias, bias_corr_, 1.0);
  }

  void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
    const BaseFloat lr  = opts_.learn_rate;

    w_rzc_x_.AddMat(-lr, w_rzc_x_corr_);
    w_rz_r_.AddMat(-lr, w_rz_r_corr_);
    w_c_r_.AddMat(-lr, w_c_r_corr_);
    w_p_m_.AddMat(-lr, w_p_m_corr_);
    bias_.AddVec(-lr, bias_corr_, 1.0);

  }

 private:
  // dims
  int32 ncell_;
  int32 nrecur_;  ///< recurrent projection layer dim
  int32 nstream_;
  int32 ntruncated_bptt_size_;

  CuMatrix<BaseFloat> prev_nnet_state_;

  CuMatrix<BaseFloat> YRP_;

  CuMatrix<BaseFloat> DRP_;
  // gradient-clipping value,
  BaseFloat clip_gradient_;

  // non-recurrent dropout
  //BaseFloat dropout_rate_;
  //CuMatrix<BaseFloat> dropout_mask_;

  // feed-forward connections: from x to [g, i, f, o]
  CuMatrix<BaseFloat> w_rzc_x_;
  CuMatrix<BaseFloat> w_rzc_x_corr_;

  // recurrent projection connections: from r to [g, i, f, o]
  CuMatrix<BaseFloat> w_rz_r_;
  CuMatrix<BaseFloat> w_rz_r_corr_;

  CuMatrix<BaseFloat> w_c_r_ ;
  CuMatrix<BaseFloat> w_c_r_corr_ ;

  CuMatrix<BaseFloat> w_p_m_;
  CuMatrix<BaseFloat> w_p_m_corr_;

  // biases of [g, i, f, o]
  CuVector<BaseFloat> bias_;
  CuVector<BaseFloat> bias_corr_;

  // propagate buffer: output of [g, i, f, o, c, h, m, r]
  CuMatrix<BaseFloat> propagate_buf_;

  // back-propagate buffer: diff-input of [g, i, f, o, c, h, m, r]
  CuMatrix<BaseFloat> backpropagate_buf_;

  BaseFloat learn_rate_coef_;
  BaseFloat bias_learn_rate_coef_;
  BaseFloat max_norm_;
};
} // namespace nnet0
} // namespace kaldi

#endif
