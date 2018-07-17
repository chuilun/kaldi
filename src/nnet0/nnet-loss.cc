// nnet0/nnet-loss.cc

// Copyright 2011-2015  Brno University of Technology (author: Karel Vesely)
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

#include "nnet0/nnet-loss.h"
#include "nnet0/nnet-utils.h"
#include "cudamatrix/cu-math.h"
#include "hmm/posterior.h"
#include "util/edit-distance.h"
#include "cudamatrix/ctc-utils.h"

#include <sstream>
#include <iterator>
#include <mpi.h>

namespace kaldi {
namespace nnet0 {


/* Xent */

/**
 * Helper function of Xent::Eval,
 * calculates number of matching elemente in 'v1', 'v2' weighted by 'weights'.
 */
template <typename T>
inline void CountCorrectFramesWeighted(const CuArray<T> &v1, 
                                       const CuArray<T> &v2, 
                                       const CuVectorBase<BaseFloat> &weights, 
                                       double *correct) {
  KALDI_ASSERT(v1.Dim() == v2.Dim());
  KALDI_ASSERT(v1.Dim() == weights.Dim());
  int32 dim = v1.Dim();
  // Get GPU data to host,
  std::vector<T> v1_h(dim), v2_h(dim);
  v1.CopyToVec(&v1_h);
  v2.CopyToVec(&v2_h);
  Vector<BaseFloat> w(dim);
  weights.CopyToVec(&w);
  // Get correct frame count (weighted),
  double corr = 0.0;
  for (int32 i=0; i<dim; i++) {
   corr += w(i) * (v1_h[i] == v2_h[i] ? 1.0 : 0.0);
  }
  // Return,
  (*correct) = corr;
}


void Xent::Eval(const VectorBase<BaseFloat> &frame_weights,
                const CuMatrixBase<BaseFloat> &net_out, 
                const CuMatrixBase<BaseFloat> &targets, 
				CuMatrixBase<BaseFloat> *diff) {
  // check inputs,
  KALDI_ASSERT(net_out.NumCols() == targets.NumCols());
  KALDI_ASSERT(net_out.NumRows() == targets.NumRows());
  KALDI_ASSERT(net_out.NumRows() == frame_weights.Dim());

  KALDI_ASSERT(KALDI_ISFINITE(frame_weights.Sum()));
  KALDI_ASSERT(KALDI_ISFINITE(net_out.Sum()));
  KALDI_ASSERT(KALDI_ISFINITE(targets.Sum()));

  // get frame_weights to GPU,
  frame_weights_ = frame_weights;

  // There may be frames for which the sum of targets is zero.
  // This happens in multi-lingual training when the frame 
  // has target class in the softmax of another language.
  // We 'switch-off' such frames by masking the 'frame_weights_',
  target_sum_.Resize(targets.NumRows());
  target_sum_.AddColSumMat(1.0, targets, 0.0);
  frame_weights_.MulElements(target_sum_);

  // get the number of frames after the masking,
  double num_frames = frame_weights_.Sum();
  KALDI_ASSERT(num_frames >= 0.0);

  // compute derivative wrt. activations of last layer of neurons,
  //*diff = net_out;
  if (diff->NumRows() != net_out.NumRows() || diff->NumCols() != net_out.NumCols())
	  (static_cast<CuMatrix<BaseFloat>*>(diff))->Resize(net_out.NumRows(), net_out.NumCols(), kUndefined);

  diff->CopyFromMat(net_out);
  diff->AddMat(-1.0, targets);
  diff->MulRowsVec(frame_weights_); // weighting,

  // evaluate the frame-level classification,
  double correct; 
  net_out.FindRowMaxId(&max_id_out_); // find max in nn-output
  targets.FindRowMaxId(&max_id_tgt_); // find max in targets
  CountCorrectFramesWeighted(max_id_out_, max_id_tgt_, frame_weights_, &correct);

  // calculate cross_entropy (in GPU),
  xentropy_aux_ = net_out; // y
  xentropy_aux_.Add(1e-20); // avoid log(0)
  xentropy_aux_.ApplyLog(); // log(y)
  xentropy_aux_.MulElements(targets); // t*log(y)
  xentropy_aux_.MulRowsVec(frame_weights_); // w*t*log(y) 
  double cross_entropy = -xentropy_aux_.Sum();
  
  // caluculate entropy (in GPU),
  entropy_aux_ = targets; // t
  entropy_aux_.Add(1e-20); // avoid log(0)
  entropy_aux_.ApplyLog(); // log(t)
  entropy_aux_.MulElements(targets); // t*log(t)
  entropy_aux_.MulRowsVec(frame_weights_); // w*t*log(t) 
  double entropy = -entropy_aux_.Sum();

  KALDI_ASSERT(KALDI_ISFINITE(cross_entropy));
  KALDI_ASSERT(KALDI_ISFINITE(entropy));

  loss_ += cross_entropy;
  entropy_ += entropy;
  correct_ += correct;
  frames_ += num_frames;

  // progressive loss reporting
  {
    static const int32 progress_step = 3600*100; // 1h
    frames_progress_ += num_frames;
    loss_progress_ += cross_entropy;
    entropy_progress_ += entropy;
    correct_progress_ += correct;
    if (frames_progress_ > progress_step) {
      KALDI_VLOG(1) << "ProgressLoss[last " 
                    << static_cast<int>(frames_progress_/100/3600) << "h of " 
                    << static_cast<int>(frames_/100/3600) << "h]: " 
                    << (loss_progress_-entropy_progress_)/frames_progress_ << " (Xent) "
                    << exp((loss_progress_-entropy_progress_)/frames_progress_) << " (PPL) "
					<< correct_progress_*100/frames_progress_ << "% (Facc)";
      // store
      loss_vec_.push_back((loss_progress_-entropy_progress_)/frames_progress_);
      // reset
      frames_progress_ = 0;
      loss_progress_ = 0.0;
      entropy_progress_ = 0.0;
      correct_progress_ = 0.0;
    }
  }
}


void Xent::Eval(const VectorBase<BaseFloat> &frame_weights,
                const CuMatrixBase<BaseFloat> &net_out, 
                const Posterior &post, 
                CuMatrix<BaseFloat> *diff) {
  int32 num_frames = net_out.NumRows(),
    num_pdf = net_out.NumCols();
  KALDI_ASSERT(num_frames == post.size());

  // convert posterior to matrix,
  PosteriorToMatrix(post, num_pdf, &tgt_mat_);

  // call the other eval function,
  Eval(frame_weights, net_out, tgt_mat_, diff);
}

/// Evaluate cross entropy using target-posteriors (supports soft labels),
 void Xent::Eval(const VectorBase<BaseFloat> &frame_weights,
           const CuMatrixBase<BaseFloat> &net_out,
           const std::vector<int32> &target,
		   CuMatrixBase<BaseFloat> *diff) {
	  int32 num_frames = net_out.NumRows(),
			  num_pdf = net_out.NumCols();
	  KALDI_ASSERT(num_frames == target.size());

	  CuArray<int32> target_vec(target);
      tgt_mat_.Resize(num_frames, num_pdf, kUndefined);
	  tgt_mat_.GenTarget(target_vec);

	  // call the other eval function,
	  Eval(frame_weights, net_out, tgt_mat_, diff);
 }

 void Xent::GetTargetWordPosterior(Vector<BaseFloat> &tgt)
 {
	 CuVector<BaseFloat> tgt_post(xentropy_aux_.NumRows(), kUndefined);
	 tgt_post.AddColSumMat(-1.0, xentropy_aux_, 0.0);
	 tgt.Resize(tgt_post.Dim(), kUndefined);
	 tgt.CopyFromVec(tgt_post);
 }

std::string Xent::Report() {
  std::ostringstream oss;
  oss << "AvgLoss: " << (loss_-entropy_)/frames_ << " (Xent), "
      << "Perplexity: " << exp((loss_-entropy_)/frames_) << " (PPL), "
      << "[AvgXent " << loss_/frames_ 
      << ", AvgTargetEnt " << entropy_/frames_ 
      << ", frames " << frames_ << "]" << std::endl;
  if (loss_vec_.size() > 0) {
     oss << "progress: [";
     std::copy(loss_vec_.begin(),loss_vec_.end(),std::ostream_iterator<float>(oss," "));
     oss << "]" << std::endl;
  }
  if (correct_ >= 0.0) {
    oss << "FRAME_ACCURACY >> " << 100.0*correct_/frames_ << "% <<" << std::endl;
  }
  return oss.str(); 
}


/* CBXent */

void CBXent::SetClassBoundary(const std::vector<int32>& class_boundary)
{
		class_boundary_ = class_boundary;

		int32 num_class = class_boundary.size()-1;
        word2class_.resize(class_boundary[num_class]);
		int i,j = 0;
		for (i = 0; i < class_boundary[num_class]; i++)
		{
			if (i>=class_boundary[j] && i<class_boundary[j+1])
				word2class_[i] = j;
			else
				word2class_[i] = ++j;
		}
		class_zt_mean_.resize(num_class+1, 0);
		class_frames_.resize(num_class+1, 0);
		class_zt_variance_.resize(num_class+1, 0);

#if HAVE_CUDA == 1
		streamlist_.resize(num_class+1);
		for (i = 0; i < num_class+1; i++)
			cudaStreamCreateWithFlags(&streamlist_[i], cudaStreamNonBlocking);
#endif
}

template <typename T>
inline void CBCountCorrectFramesWeighted(const CuArray<T> &v1, 
                                       const CuArray<T> &v2, 
                                       const CuVectorBase<BaseFloat> &weights, 
                                       double *correct) {
  KALDI_ASSERT(v1.Dim() == v2.Dim());
  KALDI_ASSERT(v1.Dim()/2 == weights.Dim());
  int32 dim = v1.Dim();
  // Get GPU data to host,
  std::vector<T> v1_h(dim), v2_h(dim);
  v1.CopyToVec(&v1_h);
  v2.CopyToVec(&v2_h);
  Vector<BaseFloat> w(dim/2);
  weights.CopyToVec(&w);
  // Get correct frame count (weighted),
  double corr = 0.0;
  for (int32 i=0; i<dim/2; i++) {
   corr += w(i) * (v1_h[i]==v2_h[i] && v1_h[i+dim/2]==v2_h[i+dim/2]  ? 1.0 : 0.0);
  }
  // Return,
  (*correct) = corr;
}

void CBXent::Eval(const VectorBase<BaseFloat> &frame_weights,
          const CuMatrixBase<BaseFloat> &net_out,
		  const std::vector<int32> &target,
		  CuMatrixBase<BaseFloat> *diff) {

	  int32 num_frames = net_out.NumRows(),
	  num_pdf = net_out.NumCols();

	  KALDI_ASSERT(num_frames == target.size());

      for (int p = 0; p < class_frame_weights_.size(); p++)
      {
            delete class_frame_weights_[p];
            delete class_target_sum_[p];
            delete class_target_[p];
            delete class_netout_[p];
            delete class_diff_[p];
            delete class_xentropy_aux_[p];
            delete class_entropy_aux_[p];
      }

	  if (tgt_mat_.NumRows() != num_frames)
	  {
		  //hos_tgt_mat_.Resize(num_frames, num_pdf, kSetZero);
		  tgt_mat_.Resize(num_frames, num_pdf, kSetZero);
		  xentropy_aux_.Resize(num_frames, num_pdf, kSetZero);
		  entropy_aux_.Resize(num_frames, num_pdf, kSetZero);
		  target_sum_.Resize(2*num_frames);
		  tgt_id_.Resize(2*num_frames);
	  }

	  if (diff->NumRows() != num_frames || diff->NumCols() != num_pdf)
		  (static_cast<CuMatrix<BaseFloat>*>(diff))->Resize(num_frames, num_pdf, kSetZero);

	  frame_weights_ = frame_weights;

	  // convert target to matrix,
	  class_frame_weights_.clear();
	  class_target_sum_.clear();
	  class_target_.clear();
	  class_netout_.clear();
	  class_diff_.clear();
	  class_xentropy_aux_.clear();
	  class_entropy_aux_.clear();
	  class_counts_.clear();
	  class_id_.clear();

      std::vector<int32> tgt_id(2*num_frames);

	  int beg = 0, len, cid, cnt = 0, total_cnt = 0;
	  for (int i = 1; i <= num_frames; i++)
	  {
          cid = word2class_[target[i-1]];
		  tgt_id[i-1] = target[i-1] - class_boundary_[cid];
		  tgt_id[i-1+num_frames] = cid;

		  if(frame_weights(i-1) != 0)	cnt++;

		  if (i == num_frames || word2class_[target[i]] != word2class_[target[i-1]])
		  {
			  cid = word2class_[target[i-1]];
			  len = class_boundary_[cid+1] - class_boundary_[cid];
			  class_frame_weights_.push_back(new CuSubVector<BaseFloat>(frame_weights_.Range(beg, i-beg)));
			  class_target_sum_.push_back(new CuSubVector<BaseFloat>(target_sum_.Range(beg, i-beg)));
			  class_target_.push_back(new CuSubMatrix<BaseFloat>(tgt_mat_.Range(beg, i-beg, class_boundary_[cid], len)));
			  class_netout_.push_back(new CuSubMatrix<BaseFloat>(net_out.Range(beg, i-beg, class_boundary_[cid], len)));
			  class_diff_.push_back(new CuSubMatrix<BaseFloat>(diff->Range(beg, i-beg, class_boundary_[cid], len)));
			  class_xentropy_aux_.push_back(new CuSubMatrix<BaseFloat>(xentropy_aux_.Range(beg, i-beg, class_boundary_[cid], len)));
			  class_entropy_aux_.push_back(new CuSubMatrix<BaseFloat>(entropy_aux_.Range(beg, i-beg, class_boundary_[cid], len)));

			  // constant normalizing
			  class_counts_.push_back(cnt);
			  class_id_.push_back(cid);
			  total_cnt += cnt; cnt = 0;

			  beg = i;
		  }
	  }
      tgt_id_.CopyFromVec(tgt_id);

	  len = num_pdf - class_boundary_.back();
	  class_frame_weights_.push_back(new CuSubVector<BaseFloat>(frame_weights_.Range(0, num_frames)));
	  class_target_sum_.push_back(new CuSubVector<BaseFloat>(target_sum_.Range(num_frames, num_frames)));
	  class_target_.push_back(new CuSubMatrix<BaseFloat>(tgt_mat_.ColRange(class_boundary_.back(), len)));
	  class_netout_.push_back(new CuSubMatrix<BaseFloat>(net_out.ColRange(class_boundary_.back(), len)));
	  class_diff_.push_back(new CuSubMatrix<BaseFloat>(diff->ColRange(class_boundary_.back(), len)));
	  class_xentropy_aux_.push_back(new CuSubMatrix<BaseFloat>(xentropy_aux_.ColRange(class_boundary_.back(), len)));
	  class_entropy_aux_.push_back(new CuSubMatrix<BaseFloat>(entropy_aux_.ColRange(class_boundary_.back(), len)));

	  class_counts_.push_back(total_cnt);
	  class_id_.push_back(class_boundary_.size()-1);


#if HAVE_CUDA == 1
	  SetStream(class_frame_weights_, streamlist_);
	  SetStream(class_target_sum_, streamlist_);
	  SetStream(class_target_, streamlist_);
	  SetStream(class_netout_, streamlist_);
	  SetStream(class_diff_, streamlist_);
	  SetStream(class_xentropy_aux_, streamlist_);
	  SetStream(class_entropy_aux_, streamlist_);
	  if (class_frame_zt_ptr_ != NULL)
		  SetStream(*class_frame_zt_ptr_, streamlist_);
#endif

	  GenTargetStreamed(class_target_, tgt_id_);

	  // call the other eval function,
	  Eval();
    
#if HAVE_CUDA == 1
	  ResetStream(class_frame_weights_);
	  ResetStream(class_target_sum_);
	  ResetStream(class_target_);
	  ResetStream(class_netout_);
	  ResetStream(class_diff_);
	  ResetStream(class_xentropy_aux_);
	  ResetStream(class_entropy_aux_);
	  if (class_frame_zt_ptr_ != NULL)
		  ResetStream(*class_frame_zt_ptr_);
#endif

}


void CBXent::Eval() {

  // There may be frames for which the sum of targets is zero.
  // This happens in multi-lingual training when the frame
  // has target class in the softmax of another language.
  // We 'switch-off' such frames by masking the 'frame_weights_',
  // int size = class_frame_weights_.size();
  double num_frames = 0, correct = 0, cross_entropy = 0, entropy = 0;
  double logzt = 0, logzt_variance = 0;
  CuVector<BaseFloat> frame_zt;
  std::vector<BaseFloat> class_zt_sum;
  //Vector<BaseFloat> tmp(frame_zt_ptr_->Dim());


  AddColSumMatStreamed(static_cast<BaseFloat>(1.0f), class_target_sum_, class_target_, static_cast<BaseFloat>(0.0f));
  MulElementsStreamed(class_target_sum_, class_frame_weights_);

  // get the number of frames after the masking,
  num_frames = VecSumStreamed(class_target_sum_);
  num_frames /= 2;


  if (var_penalty_ != 0)
  {
	  	KALDI_ASSERT(class_frame_zt_ptr_ != NULL);
		// constant normalizing (for each class per frame), zt = sum(exp(y)), log(zt)
		MulElementsStreamed(*class_frame_zt_ptr_, class_frame_weights_);
        //KALDI_ASSERT(KALDI_ISFINITE(frame_zt_ptr_->Sum()));
		logzt = VecSumStreamed(*class_frame_zt_ptr_, &class_zt_sum);
        //logzt -= class_zt_sum.back(); //except last class output
		for (int i = 0; i < class_counts_.size(); i++)
		{
			class_frames_[class_id_[i]] += class_counts_[i];
			class_zt_mean_[class_id_[i]] += class_zt_sum[i];
			class_zt_sum[i] = (class_counts_[i] == 0 ? 0 : class_zt_sum[i]/(-class_counts_[i]));
            //KALDI_ASSERT(class_counts_[i] != 0);
            //class_zt_sum[i] /= (-class_counts_[i]);
		}

		// 2beta*(log(zt) - log(zt)/n)+1
		AddStreamed(*class_frame_zt_ptr_, class_zt_sum);
		MulElementsStreamed(*class_frame_zt_ptr_, class_frame_weights_); // weighting after sub zt mean
		frame_zt = *frame_zt_ptr_; //backup for compute variance
		frame_zt_ptr_->Scale(var_penalty_*2); // *beta*2
		frame_zt_ptr_->Add(static_cast<BaseFloat>(1.0f));
  }


  // compute derivative wrt. activations of last layer of neurons,
  CopyFromMatStreamed(class_netout_, class_diff_);
  if (var_penalty_ != 0)
  {
	  // y*(2beta*(log(zt) - log(zt)/n)+1)
	  MulRowsVecStreamed(class_diff_, *class_frame_zt_ptr_); // constant normalizing contain last class output

	  /*
      // except last class output
      int n = class_diff_.size()-1;
      std::vector<CuSubMatrix<BaseFloat>* > class_diff_zt(n, NULL);
      std::vector<CuSubVector<BaseFloat>* > class_frame_zt(n, NULL);
      for (int i = 0; i < n; i++) {
            class_diff_zt[i] = class_diff_[i];
            class_frame_zt[i] = (*class_frame_zt_ptr_)[i]; 
       }
	   MulRowsVecStreamed(class_diff_zt, class_frame_zt); // constant normalizing except last class output
	   */
        
	  // compute logzt variance
	  frame_zt_ptr_->CopyFromVec(frame_zt);
	  MulElementsStreamed(*class_frame_zt_ptr_, *class_frame_zt_ptr_);
	  logzt_variance = VecSumStreamed(*class_frame_zt_ptr_, &class_zt_sum);
      //logzt_variance -= class_zt_sum.back(); // except last class
	  for (int i = 0; i < class_counts_.size(); i++)
		  class_zt_variance_[class_id_[i]] += class_zt_sum[i];
  }
  AddMatStreamed(static_cast<BaseFloat>(-1.0f), class_diff_, class_target_);
  MulRowsVecStreamed(class_diff_, class_frame_weights_); // weighting,

  // for constant class zt
  if (const_class_zt_.size() == class_boundary_.size())
  {	  /*
      // last class output
      CuSubMatrix<BaseFloat> *tmp = class_netout_.back();
      CuMatrix<BaseFloat> last_netout(*tmp);
      tmp->ApplyLogSoftMaxPerRow(last_netout);
      */

      int n = class_netout_.size();
	  std::vector<BaseFloat> class_zt(n, 0);

	  for (int i = 0; i < n; i++)
		  class_zt[i] = -const_class_zt_[class_id_[i]];
      //class_zt[n-1] = 0;  // last class output
	  AddStreamed(class_netout_, class_zt);
	  ApplyExpStreamed(class_netout_);
  }

  // evaluate the frame-level classification,
  FindMaxIdPerRowStreamed(class_netout_, max_id_out_);
  FindMaxIdPerRowStreamed(class_target_, max_id_tgt_);
  CBCountCorrectFramesWeighted(max_id_out_, max_id_tgt_, frame_weights_, &correct);

  // calculate cross_entropy (in GPU), (avoid log(0)), log(y), t*log(y), w*t*log(y)
  CrossEntropyStreamed(class_xentropy_aux_, class_netout_, class_target_);
  MulRowsVecStreamed(class_xentropy_aux_, class_frame_weights_); // weighting,
  cross_entropy = -MatSumStreamed(class_xentropy_aux_);

  // caluculate entropy (in GPU), (avoid log(0)), log(t), t*log(t), w*t*log(t)
  EntropyStreamed(class_entropy_aux_, class_target_);
  MulRowsVecStreamed(class_entropy_aux_, class_frame_weights_); // weighting,
  entropy = -MatSumStreamed(class_entropy_aux_);

  KALDI_ASSERT(KALDI_ISFINITE(cross_entropy));
  KALDI_ASSERT(KALDI_ISFINITE(entropy));


  loss_ += cross_entropy;
  entropy_ += entropy;
  correct_ += correct;
  frames_ += num_frames;
  ppl_ = exp(loss_/frames_);
  logzt_ += logzt;
  logzt_variance_ += logzt_variance;

  // progressive loss reporting
  {
    static const int32 progress_step = 3600*100; // 1h
    frames_progress_ += num_frames;
    loss_progress_ += cross_entropy;
    entropy_progress_ += entropy;
    correct_progress_ += correct;
    ppl_progress_ = exp(loss_progress_/frames_progress_);
    logzt_progress_ += logzt;
    logzt_variance_progress_ += logzt_variance;
    if (frames_progress_ > progress_step) {
      KALDI_VLOG(1) << "ProgressLoss[last "
                    << static_cast<int>(frames_progress_/100/3600) << "(1h words) of "
                    << static_cast<int>(frames_/100/3600) << "(1h words)]: "
                    << (loss_progress_-entropy_progress_)/frames_progress_ << " (Xent) "
					<< "("<<logzt_progress_/frames_progress_<<", "<<logzt_variance_progress_/frames_progress_<<")"
					<< " (logzt[mean,variance]) "
                    << ppl_progress_ << " (PPL) "
					<< correct_progress_*100/frames_progress_ << "% (Facc)";
      // store
      loss_vec_.push_back((loss_progress_-entropy_progress_)/frames_progress_);
      // reset
      frames_progress_ = 0;
      loss_progress_ = 0.0;
      entropy_progress_ = 0.0;
      correct_progress_ = 0.0;
      logzt_progress_ = 0.0;
      logzt_variance_progress_ = 0.0;
    }
  }
}

void CBXent::GetTargetWordPosterior(Vector<BaseFloat> &tgt)
{
#if HAVE_CUDA == 1
	SetStream(class_target_sum_, streamlist_);
	AddColSumMatStreamed(static_cast<BaseFloat>(1.0f), class_target_sum_, class_xentropy_aux_, static_cast<BaseFloat>(0.0f));
	ResetStream(class_target_sum_);
#endif

	Vector<BaseFloat> tgt_post(target_sum_);
	int size = tgt_post.Dim()/2;
	tgt.Resize(size, kUndefined);
	for (int i = 0; i < size; i++)
		tgt(i) = -(tgt_post(i) + tgt_post(size+i));
}

std::string CBXent::Report() {
  std::ostringstream oss;
  oss << "AvgLoss: " << (loss_-entropy_)/frames_ << " (Xent), "
	  << "("<< logzt_/frames_<<", "<<logzt_variance_/frames_<<")"
	  << " (Logzt[mean,variance]), "
      << "Perplexity: " << ppl_ << " (PPL), "
      << "[AvgXent " << loss_/frames_
      << ", AvgTargetEnt " << entropy_/frames_
      << ", frames " << frames_ << "]" << std::endl;
  if (loss_vec_.size() > 0) {
     oss << "progress: [";
     std::copy(loss_vec_.begin(),loss_vec_.end(),std::ostream_iterator<float>(oss," "));
     oss << "]" << std::endl;
  }
  if (var_penalty_ != 0 && class_zt_variance_.size() > 0) {
     std::vector<double> buffer(class_zt_variance_.size());

     // zt variance
	 oss << "]" << std::endl;
	 for (int i = 0; i < buffer.size(); i++)
        buffer[i] = (class_frames_[i] == 0 ? 0 : class_zt_variance_[i]/class_frames_[i]);
	 oss << "class zt variance: [ ";
	 std::copy(buffer.begin(),buffer.end(),std::ostream_iterator<float>(oss," "));
	 oss << "]" << std::endl;

     // zt mean
	 for (int i = 0; i < buffer.size(); i++)
        buffer[i] = (class_frames_[i] == 0 ? 0 : class_zt_mean_[i]/class_frames_[i]);
	 oss << "class zt mean: [ ";
	 std::copy(buffer.begin(),buffer.end(),std::ostream_iterator<float>(oss," "));
	 oss << "]" << std::endl;

     /*
	 oss << "class frame count: [ ";
	 std::copy(class_frames_.begin(),class_frames_.end(),std::ostream_iterator<float>(oss," "));
	 oss << "]" << std::endl;
     */
  }
  if (correct_ >= 0.0) {
    oss << "PPL >> " << ppl_ << " <<" << std::endl;
  }
  return oss.str();
}

/// Merge lost
void CBXent::Add(CBXent *xent)
{
	  this->frames_ += xent->frames_;
	  this->correct_ += xent->correct_;
	  this->loss_ += xent->loss_;
	  this->entropy_ += xent->entropy_;
	  this->ppl_ = exp(loss_/frames_);
	  this->logzt_ += xent->logzt_;
	  this->logzt_variance_ += xent->logzt_variance_;

	  // partial results during training
	  frames_progress_ += xent->frames_progress_;
	  loss_progress_ += xent->loss_progress_;
	  entropy_progress_+= xent->entropy_progress_;
	  ppl_progress_ = exp(loss_progress_/frames_progress_);
	  logzt_progress_ += xent->logzt_progress_;
	  logzt_variance_progress_ += xent->logzt_variance_progress_;

	  for (int i = 0; i<this->loss_vec_.size() && i < xent->loss_vec_.size(); i++)
		  this->loss_vec_[i] += xent->loss_vec_[i];

	  // variance per class
	  var_penalty_ = xent->var_penalty_;
	  if (var_penalty_ != 0 )
	  {
		  if (this->class_zt_variance_.size() == 0)
			  class_zt_variance_.resize(xent->class_zt_variance_.size());
		  for (int i = 0; i<this->class_zt_variance_.size() && i<xent->class_zt_variance_.size(); i++)
				  this->class_zt_variance_[i] += xent->class_zt_variance_[i];

		  if (this->class_zt_mean_.size() == 0)
			  class_zt_mean_.resize(xent->class_zt_mean_.size());
		  for (int i = 0; i<this->class_zt_mean_.size() && i<xent->class_zt_mean_.size(); i++)
				  this->class_zt_mean_[i] += xent->class_zt_mean_[i];

		  if (this->class_frames_.size() == 0)
			  class_frames_.resize(xent->class_frames_.size());
		  for (int i = 0; i<this->class_frames_.size() && i<xent->class_frames_.size(); i++)
				  this->class_frames_[i] += xent->class_frames_[i];
	  }
}

void CBXent::Merge(int myid, int root)
{

	MPI_Barrier(MPI_COMM_WORLD);

	void *addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&this->frames_));
	MPI_Reduce(addr, (void*)(&this->frames_), 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

	addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&this->correct_));
	MPI_Reduce(addr, (void*)(&this->correct_), 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

	addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&this->loss_));
	MPI_Reduce(addr, (void*)(&this->loss_), 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

	addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&this->entropy_));
	MPI_Reduce(addr, (void*)(&this->entropy_), 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

	this->ppl_ = exp(loss_/frames_);

	if (var_penalty_ != 0)
	{
		addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&this->logzt_));
		MPI_Reduce(addr, (void*)(&this->logzt_), 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
		addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&this->logzt_variance_));
		MPI_Reduce(addr, (void*)(&this->logzt_variance_), 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
		addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&this->class_zt_variance_.front()));
		MPI_Reduce(addr, (void*)(&this->class_zt_variance_.front()), this->class_zt_variance_.size(), MPI_FLOAT, MPI_SUM, root, MPI_COMM_WORLD);
		addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&this->class_zt_mean_.front()));
		MPI_Reduce(addr, (void*)(&this->class_zt_mean_.front()), this->class_zt_mean_.size(), MPI_FLOAT, MPI_SUM, root, MPI_COMM_WORLD);
		addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&this->class_frames_.front()));
		MPI_Reduce(addr, (void*)(&this->class_frames_.front()), this->class_frames_.size(), MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
	}
}

/* Mse */

void Mse::Eval(const VectorBase<BaseFloat> &frame_weights,
               const CuMatrixBase<BaseFloat>& net_out, 
               const CuMatrixBase<BaseFloat>& target, 
               CuMatrixBase<BaseFloat>* diff) {
  // check inputs,
  KALDI_ASSERT(net_out.NumCols() == target.NumCols());
  KALDI_ASSERT(net_out.NumRows() == target.NumRows());
  KALDI_ASSERT(net_out.NumRows() == frame_weights.Dim());

  KALDI_ASSERT(KALDI_ISFINITE(frame_weights.Sum()));
  KALDI_ASSERT(KALDI_ISFINITE(net_out.Sum()));
  KALDI_ASSERT(KALDI_ISFINITE(target.Sum()));

  int32 num_frames = frame_weights.Sum();
  KALDI_ASSERT(num_frames >= 0.0);

  // get frame_weights to GPU,
  frame_weights_ = frame_weights;

  //compute derivative w.r.t. neural nerwork outputs
  //*diff = net_out; // y
  if (diff->NumRows() != net_out.NumRows() || diff->NumCols() != net_out.NumCols())
	  (static_cast<CuMatrix<BaseFloat>*>(diff))->Resize(net_out.NumRows(), net_out.NumCols(), kUndefined);
  diff->CopyFromMat(net_out);

  diff->AddMat(-1.0,target); // (y - t)
  diff->MulRowsVec(frame_weights_); // weighting,

  // Compute MeanSquareError loss of mini-batch
  diff_pow_2_ = *diff;
  diff_pow_2_.MulElements(diff_pow_2_); // (y - t)^2
  diff_pow_2_.MulRowsVec(frame_weights_); // w*(y - t)^2
  double mean_square_error = 0.5 * diff_pow_2_.Sum(); // sum the matrix,

  KALDI_ASSERT(KALDI_ISFINITE(mean_square_error));

  // accumulate
  loss_ += mean_square_error;
  frames_ += num_frames;

  // progressive loss reporting
  {
    static const int32 progress_step = 3600*100; // 1h
    frames_progress_ += num_frames;
    loss_progress_ += mean_square_error;
    if (frames_progress_ > progress_step) {
      KALDI_VLOG(1) << "ProgressLoss[last " 
                    << static_cast<int>(frames_progress_/100/3600) << "h of " 
                    << static_cast<int>(frames_/100/3600) << "h]: " 
                    << loss_progress_/frames_progress_ << " (Mse)";
      // store
      loss_vec_.push_back(loss_progress_/frames_progress_);
      // reset
      frames_progress_ = 0;
      loss_progress_ = 0.0;
    }
  }
}


/// Merge lost
void Xent::Add(LossItf *loss)
{
	Xent *xent = dynamic_cast<Xent*>(loss);
	this->frames_ += xent->frames_;
	this->correct_ += xent->correct_;
	this->loss_ += xent->loss_;
	this->entropy_ += xent->entropy_;

	// partial results during training
	frames_progress_ += xent->frames_progress_;
	loss_progress_ += xent->loss_progress_;
	entropy_progress_+= xent->entropy_progress_;

	for (int i = 0; i<this->loss_vec_.size() && i < xent->loss_vec_.size(); i++)
	  this->loss_vec_[i] += xent->loss_vec_[i];
}

void Xent::Merge(int myid, int root)
{

	MPI_Barrier(MPI_COMM_WORLD);

	void *addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&this->frames_));
	MPI_Reduce(addr, (void*)(&this->frames_), 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

	addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&this->correct_));
	MPI_Reduce(addr, (void*)(&this->correct_), 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);


	addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&this->loss_));
	MPI_Reduce(addr, (void*)(&this->loss_), 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

	addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&this->entropy_));
	MPI_Reduce(addr, (void*)(&this->entropy_), 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
}



void Mse::Eval(const VectorBase<BaseFloat> &frame_weights,
               const CuMatrixBase<BaseFloat>& net_out, 
               const Posterior& post, 
               CuMatrix<BaseFloat>* diff) {
  int32 num_frames = net_out.NumRows(),
    num_nn_outputs = net_out.NumCols();
  KALDI_ASSERT(num_frames == post.size());

  // convert posterior to matrix,
  PosteriorToMatrix(post, num_nn_outputs, &tgt_mat_);

  // call the other eval function,
  Eval(frame_weights, net_out, tgt_mat_, diff);
}
 

std::string Mse::Report() {
  // compute root mean square,
  int32 num_tgt = diff_pow_2_.NumCols();
  BaseFloat root_mean_square = sqrt(loss_/frames_/num_tgt);
  // build the message,
  std::ostringstream oss;
  oss << "AvgLoss: " << loss_/frames_ << " (Mse), " 
      << "[RMS " << root_mean_square << ", frames " << frames_ << "]" << std::endl;
  oss << "progress: [";
  std::copy(loss_vec_.begin(),loss_vec_.end(),std::ostream_iterator<float>(oss," "));
  oss << "]" << std::endl;
  return oss.str();
}

/// Merge lost
void Mse::Add(LossItf *loss)
{
	Mse *mse = dynamic_cast<Mse*>(loss);
	this->frames_ += mse->frames_;
	this->loss_ += mse->loss_;

	// partial results during training
	frames_progress_ += mse->frames_progress_;
	loss_progress_ += mse->loss_progress_;

	for (int i = 0; i<this->loss_vec_.size() && i<mse->loss_vec_.size(); i++)
	  this->loss_vec_[i] += mse->loss_vec_[i];
}

void Mse::Merge(int myid, int root)
{

	MPI_Barrier(MPI_COMM_WORLD);

	void *addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&this->frames_));
	MPI_Reduce(addr, (void*)(&this->frames_), 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

	addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&this->loss_));
	MPI_Reduce(addr, (void*)(&this->loss_), 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

}


/* MultiTaskLoss */

void MultiTaskLoss::InitFromString(const std::string& s) {
  std::vector<std::string> v;
  SplitStringToVector(s, ",:" /* delimiter */, false, &v);

  KALDI_ASSERT((v.size()-1) % 3 == 0); // triplets,
  KALDI_ASSERT(v[0] == "multitask"); // header,

  // parse the definition of multitask loss,
  std::vector<std::string>::iterator it(v.begin()+1); // skip header,
  for ( ; it != v.end(); ++it) {
    // type,
    if (*it == "xent") {
      loss_vec_.push_back(new Xent());
    } else if (*it == "mse") {
      loss_vec_.push_back(new Mse());
    } else {
      KALDI_ERR << "Unknown objective function code : " << *it;
    }
    ++it;
    // dim,
    int32 dim;
    if (!ConvertStringToInteger(*it, &dim)) {
      KALDI_ERR << "Cannot convert 'dim' " << *it << " to integer!";
    }
    loss_dim_.push_back(dim);
    ++it;
    // weight,
    BaseFloat weight;
    if (!ConvertStringToReal(*it, &weight)) {
      KALDI_ERR << "Cannot convert 'weight' " << *it << " to integer!";
    }
    KALDI_ASSERT(weight >= 0.0);
    loss_weights_.push_back(weight);
  }

  // build vector with starting-point offsets,
  loss_dim_offset_.resize(loss_dim_.size()+1, 0); // 1st zero stays,
  for (int32 i = 1; i <= loss_dim_.size(); i++) {
    loss_dim_offset_[i] = loss_dim_offset_[i-1] + loss_dim_[i-1];
  }

  // sanity check,
  KALDI_ASSERT(loss_vec_.size() > 0);
  KALDI_ASSERT(loss_vec_.size() == loss_dim_.size());
  KALDI_ASSERT(loss_vec_.size() == loss_weights_.size());
}

void MultiTaskLoss::Eval(const VectorBase<BaseFloat> &frame_weights, 
            const CuMatrixBase<BaseFloat>& net_out, 
            const Posterior& post,
            CuMatrix<BaseFloat>* diff) {
  int32 num_frames = net_out.NumRows(),
    num_output = net_out.NumCols();
  KALDI_ASSERT(num_frames == post.size());
  KALDI_ASSERT(num_output == loss_dim_offset_.back()); // sum of loss-dims,

  // convert posterior to matrix,
  PosteriorToMatrix(post, num_output, &tgt_mat_);

  // allocate diff matrix,
  diff->Resize(num_frames, num_output);
  
  // call the vector of loss functions,
  CuMatrix<BaseFloat> diff_aux;
  for (int32 i = 0; i < loss_vec_.size(); i++) {
    loss_vec_[i]->Eval(frame_weights, 
      net_out.ColRange(loss_dim_offset_[i], loss_dim_[i]),
      tgt_mat_.ColRange(loss_dim_offset_[i], loss_dim_[i]),
      &diff_aux);
    // Scale the gradients,
    diff_aux.Scale(loss_weights_[i]);
    // Copy to diff,
    diff->ColRange(loss_dim_offset_[i], loss_dim_[i]).CopyFromMat(diff_aux);
  }
}

std::string MultiTaskLoss::Report() {
  // calculate overall loss (weighted),
  BaseFloat overall_loss = AvgLoss();
  // copy the loss-values into a vector,
  std::vector<BaseFloat> loss_values;
  for (int32 i = 0; i < loss_vec_.size(); i++) {
    loss_values.push_back(loss_vec_[i]->AvgLoss());
  }

  // build the message,
  std::ostringstream oss;
  oss << "MultiTaskLoss, with " << loss_vec_.size()
	  << " parallel loss functions." << std::endl;
  // individual loss reports first,
  for (int32 i = 0; i < loss_vec_.size(); i++) {
    oss << "Loss " << i+1 << ", " << loss_vec_[i]->Report() << std::endl;
  }

  // overall loss is last,
  oss << "Loss (OVERALL), " 
      << "AvgLoss: " << overall_loss << " (MultiTaskLoss), "
      << "weights " << loss_weights_ << ", "
      << "values " << loss_values << std::endl;

  return oss.str();
}

BaseFloat MultiTaskLoss::AvgLoss() {
  BaseFloat ans(0.0);
  for (int32 i = 0; i < loss_vec_.size(); i++) {
    BaseFloat val = loss_weights_[i] * loss_vec_[i]->AvgLoss();
    if(!KALDI_ISFINITE(val)) {
      KALDI_WARN << "Loss " << i+1 << ", has bad objective function value '" << val << "', using 0.0 instead.";
      val = 0.0;
    }
    ans += val;
  }
  return ans;
}

/// Merge statistic data
void MultiTaskLoss::Add(LossItf *loss) {

	MultiTaskLoss *multitask = dynamic_cast<MultiTaskLoss*>(loss);
	for (int32 i = 0; i < loss_vec_.size(); i++) {
		loss_vec_[i]->Add(multitask->loss_vec_[i]);
	}
}

void MultiTaskLoss::Merge(int myid, int root) {

	for (int32 i = 0; i < loss_vec_.size(); i++) {
		loss_vec_[i]->Merge(myid, root);
	}
}


/**CTC**/

void Ctc::Eval(const CuMatrixBase<BaseFloat> &net_out, const std::vector<int32> &label, CuMatrix<BaseFloat> *diff) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {

  diff->Resize(net_out.NumRows(), net_out.NumCols());
  int32 num_frames = net_out.NumRows();
  int32 num_classes = net_out.NumCols();

  // label expansion by inserting blank (indexed by 0) at the beginning and end,
  // and between every pair of labels
  int32 len_labels = label.size();
  int32 exp_len_labels = 2*len_labels + 1;

  label_expand_.resize(0);
  label_expand_.resize(exp_len_labels, 0);
  for (int l = 0; l < len_labels; l++) {
    label_expand_[2*l+1] = label[l];
  }

  // compute in log scale
  CuMatrix<BaseFloat> log_nnet_out(net_out);
  log_nnet_out.ApplyLog();

  alpha_.Resize(num_frames, exp_len_labels, kSetZero);
  beta_.Resize(num_frames, exp_len_labels, kSetZero);
  for (int t = 0; t < num_frames; t++) {
    alpha_.ComputeCtcAlpha(log_nnet_out, t, label_expand_, false);
  }
  for (int t = (num_frames - 1); t >= 0; t--) {
    beta_.ComputeCtcBeta(log_nnet_out, t, label_expand_, false);
  }

  // compute the log-likelihood of the label sequence given the inputs logP(z|x)
  BaseFloat tmp1 = alpha_(num_frames-1, exp_len_labels-1);
  BaseFloat tmp2 = alpha_(num_frames-1, exp_len_labels-2);
  BaseFloat pzx = tmp1 + log(1 + ExpA(tmp2 - tmp1));

  // compute the errors
  ctc_err_.Resize(num_frames, num_classes, kSetZero);
  ctc_err_.ComputeCtcError(alpha_, beta_, net_out, label_expand_, pzx);  // here should use the original ??

  // back-propagate the errors through the softmax layer
  ctc_err_.MulElements(net_out);
  CuVector<BaseFloat> row_sum(num_frames, kSetZero);
  row_sum.AddColSumMat(1.0, ctc_err_, 0.0);

  CuMatrix<BaseFloat> net_out_tmp(net_out);
  net_out_tmp.MulRowsVec(row_sum);
  diff->CopyFromMat(ctc_err_);

  diff->AddMat(-1.0, net_out_tmp);

  // update registries
  obj_progress_ += pzx;
  sequences_progress_ += 1;
  sequences_num_ += 1;
  frames_progress_ += num_frames;
  frames_ += num_frames;

  // progressive reporting
  {
    if (sequences_progress_ > report_step_) {
      KALDI_VLOG(1) << "After " << sequences_num_ << " sequences (" << frames_/(100.0 * 3600) << "Hr): "
                    << "Obj(log[Pzx]) = " << obj_progress_/sequences_progress_
                    << "   TokenAcc = " << 100.0*(1.0 - error_num_progress_/ref_num_progress_) << "%";
      // reset
      sequences_progress_ = 0;
      frames_progress_ = 0;
      obj_progress_ = 0.0;
      error_num_progress_ = 0;
      ref_num_progress_ = 0;
    }
  }

  }else
#endif
        {
                // not implemented for CPU yet
                // return 0;
        }

}

void Ctc::EvalParallel(const std::vector<int32> &frame_num_utt, const CuMatrixBase<BaseFloat> &net_out,
                       std::vector< std::vector<int32> > &label, CuMatrix<BaseFloat> *diff) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {

  diff->Resize(net_out.NumRows(), net_out.NumCols());

  int32 num_sequence = frame_num_utt.size();  // number of sequences
  int32 num_frames = net_out.NumRows();
  KALDI_ASSERT(num_frames % num_sequence == 0);  // after padding, number of frames is a multiple of number of sequences

  int32 num_frames_per_sequence = num_frames / num_sequence;
  int32 num_classes = net_out.NumCols();
  int32 max_label_len = 0;
  for (int32 s = 0; s < num_sequence; s++) {
    if (label[s].size() > max_label_len) max_label_len = label[s].size();
  }

  // label expansion
  std::vector<int32> label_lengths_utt(num_sequence);
  int32 exp_len_labels = 2*max_label_len + 1;
  label_expand_.resize(0);
  label_expand_.resize(num_sequence * exp_len_labels, -1);
  for (int32 s = 0; s < num_sequence; s++) {
    std::vector<int32> label_s = label[s];
    label_lengths_utt[s] = 2 * label_s.size() + 1;
    for (int32 l = 0; l < label_s.size(); l++) {
      label_expand_[s*exp_len_labels + 2*l] = 0;
      label_expand_[s*exp_len_labels + 2*l + 1] = label_s[l];
    }
    label_expand_[s*exp_len_labels + 2*label_s.size()] = 0;
  }

  // convert into the log scale
  CuMatrix<BaseFloat> log_nnet_out(net_out);
  log_nnet_out.ApplyLog();

  // do the forward and backward pass, to compute alpha and beta values
  alpha_.Resize(num_frames, exp_len_labels);
  beta_.Resize(num_frames, exp_len_labels);
  alpha_.Set(NumericLimits<BaseFloat>::log_zero_);
  beta_.Set(NumericLimits<BaseFloat>::log_zero_);
  for (int t = 0; t < num_frames_per_sequence; t++) {
    alpha_.ComputeCtcAlphaMSeq(log_nnet_out, t, label_expand_, frame_num_utt);
  }
  for (int t = (num_frames_per_sequence - 1); t >= 0; t--) {
    beta_.ComputeCtcBetaMSeq(log_nnet_out, t, label_expand_, frame_num_utt, label_lengths_utt);
  }
  CuVector<BaseFloat> pzx(num_sequence, kSetZero);
  for (int s = 0; s < num_sequence; s++) {
    int label_len = 2* label[s].size() + 1;
    int frame_num = frame_num_utt[s];
    BaseFloat tmp1 = alpha_((frame_num-1)*num_sequence + s, label_len - 1);
    BaseFloat tmp2 = alpha_((frame_num-1)*num_sequence + s, label_len-2);
    pzx(s) = tmp1 + log(1 + ExpA(tmp2 - tmp1));
  }

  // gradients from CTC
  ctc_err_.Resize(num_frames, num_classes, kSetZero);
  ctc_err_.ComputeCtcErrorMSeq(alpha_, beta_, net_out, label_expand_, frame_num_utt, pzx);  // here should use the original ??

  // back-propagate the errors through the softmax layer
  ctc_err_.MulElements(net_out);
  CuVector<BaseFloat> row_sum(num_frames, kSetZero);
  row_sum.AddColSumMat(1.0, ctc_err_, 0.0);

  CuMatrix<BaseFloat> net_out_tmp(net_out);
  net_out_tmp.MulRowsVec(row_sum);
  diff->CopyFromMat(ctc_err_);

  diff->AddMat(-1.0, net_out_tmp);

  // update registries
  obj_progress_ += pzx.Sum();
  sequences_progress_ += num_sequence;
  sequences_num_ += num_sequence;
  for (int s = 0; s < num_sequence; s++) {
    frames_progress_ += frame_num_utt[s];
    frames_ += frame_num_utt[s];
  }

  // progressive reporting
  {
    if (sequences_progress_ > report_step_) {
      KALDI_VLOG(1) << "After " << sequences_num_ << " sequences (" << frames_/(100.0 * 3600) << "Hr): "
                    << "Obj(log[Pzx]) = " << obj_progress_/sequences_progress_
                    << "  TokenAcc = " << 100.0*(1.0 - error_num_progress_/ref_num_progress_) << "%";
      // reset
      sequences_progress_ = 0;
      frames_progress_ = 0;
      obj_progress_ = 0.0;
      error_num_progress_ = 0;
      ref_num_progress_ = 0;
    }
  }

  }else
#endif
        {
                // not implemented for CPU yet
                // return 0;
        }

}

void Ctc::ErrorRate(const CuMatrixBase<BaseFloat> &net_out, const std::vector<int32> &label, float* err_rate, std::vector<int32> *hyp) {

  // frame-level labels, by selecting the label with the largest probability at each frame
  CuArray<int32> maxid(net_out.NumRows());
  net_out.FindRowMaxId(&maxid);

  int32 dim = maxid.Dim();

  std::vector<int32> data(dim);
  maxid.CopyToVec(&data);

  // remove the repetitions
  int32 i = 1, j = 1;
  while(j < dim) {
    if (data[j] != data[j-1]) {
      data[i] = data[j];
      i++;
    }
    j++;
  }
  // remove the blanks
  std::vector<int32> hyp_seq(0);
  for (int32 n = 0; n < i; n++) {
    if (data[n] != 0) {
      hyp_seq.push_back(data[n]);
    }
  }
  hyp->resize(0);
  *hyp = hyp_seq;

  int32 err, ins, del, sub;
  err =  LevenshteinEditDistance(label, hyp_seq, &ins, &del, &sub);
  *err_rate = (100.0 * err) / label.size();
  error_num_ += err;
  ref_num_ += label.size();
  error_num_progress_ += err;
  ref_num_progress_ += label.size();
}

void Ctc::ErrorRateMSeq(const std::vector<int> &frame_num_utt, const CuMatrixBase<BaseFloat> &net_out, std::vector< std::vector<int> > &label) {

  // frame-level labels
  CuArray<int32> maxid(net_out.NumRows());
  net_out.FindRowMaxId(&maxid);

  int32 dim = maxid.Dim();
  std::vector<int32> data(dim);
  maxid.CopyToVec(&data);

  // compute errors sequence by sequence
  int32 num_seq = frame_num_utt.size();
  for (int32 s = 0; s < num_seq; s++) {
    int32 num_frame = frame_num_utt[s];
    std::vector<int32> raw_hyp_seq(num_frame);
    for (int32 f = 0; f < num_frame; f++) {
      raw_hyp_seq[f] = data[f*num_seq + s];
    }
    int32 i = 1, j = 1;
    while(j < num_frame) {
      if (raw_hyp_seq[j] != raw_hyp_seq[j-1]) {
        raw_hyp_seq[i] = raw_hyp_seq[j];
        i++;
      }
      j++;
    }
    std::vector<int32> hyp_seq(0);
    for (int32 n = 0; n < i; n++) {
      if (raw_hyp_seq[n] != 0) {
        hyp_seq.push_back(raw_hyp_seq[n]);
      }
    }
    int32 err, ins, del, sub;
    err =  LevenshteinEditDistance(label[s], hyp_seq, &ins, &del, &sub);
    error_num_ += err;
    ref_num_ += label[s].size();
    error_num_progress_ += err;
    ref_num_progress_ += label[s].size();
  }
}

/// Merge lost
void Ctc::Add(Ctc *ctc)
{
	  this->error_num_ += ctc->error_num_;
	  this->ref_num_ += ctc->ref_num_;
	  this->frames_ += ctc->frames_;

	  // partial results during training
	  this->error_num_progress_ += ctc->error_num_progress_;
	  this->ref_num_progress_ += ctc->ref_num_progress_;
	  this->obj_progress_ += ctc->obj_progress_;
	  this->sequences_progress_ += ctc->sequences_progress_;
	  this->frames_progress_ += ctc->frames_progress_;

}

void Ctc::Merge(int myid, int root)
{

	MPI_Barrier(MPI_COMM_WORLD);

	void *addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&this->error_num_));
	MPI_Reduce(addr, (void*)(&this->error_num_), 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

	addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&this->ref_num_));
	MPI_Reduce(addr, (void*)(&this->ref_num_), 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

}

std::string Ctc::Report() {
  std::ostringstream oss;
  oss << "\nTOKEN_ACCURACY >> " << 100.0*(1.0 - error_num_/ref_num_) << "% <<";
  return oss.str();
}

} // namespace nnet0
} // namespace kaldi
