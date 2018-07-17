/*
 * nnet-affine-preconditioned-transform.cc
 *
 *  Created on: 2015年4月13日
 *      Author: wd007
 */

#include "nnet-affine-preconditioned-transform.h"


namespace kaldi {
namespace nnet0 {

AffinePreconditionedOnlineTransform::
AffinePreconditionedOnlineTransform(const AffineTransform &orig, int32 rank_in, int32 rank_out,
									int32 update_period, BaseFloat num_samples_history, BaseFloat alpha)
									:AffineTransform(orig.InputDim(), orig.OutputDim())
{
	this->linearity_ = orig.linearity_;
	this->bias_ = orig.bias_;

	this->linearity_corr_ = orig.linearity_corr_;
	this->bias_corr_ = orig.bias_corr_;

	this->opts_ = orig.opts_;

	this->learn_rate_coef_ = orig.learn_rate_coef_;
	this->bias_learn_rate_coef_ = orig.bias_learn_rate_coef_;
	this->max_norm_ = orig.max_norm_;
	
	this->local_lrate = opts_.learn_rate;
	this->local_lrate_bias = opts_.learn_rate;

	this->rank_in_ = rank_in;
	this->rank_out_ = rank_out;
	this->update_period_ = update_period;
	this->num_samples_history_ = num_samples_history;
	this->alpha_ = alpha;
	this->max_change_per_sample_ = 0.1;
	SetPreconditionerConfigs();
}

void AffinePreconditionedOnlineTransform::Gradient(const CuMatrixBase<BaseFloat> &in_value, const CuMatrixBase<BaseFloat> &out_deriv) {
	  // we use following hyperparameters from the option class
	  const BaseFloat lr = opts_.learn_rate * learn_rate_coef_;
	  const BaseFloat lr_bias = opts_.learn_rate * bias_learn_rate_coef_;
	  const BaseFloat mmt = opts_.momentum;
	  const BaseFloat l2 = opts_.l2_penalty;
	  const BaseFloat l1 = opts_.l1_penalty;
	  // we will also need the number of frames in the mini-batch
	  const int32 num_frames = in_value.NumRows();

	  CuMatrix<BaseFloat> in_value_temp;

	    in_value_temp.Resize(in_value.NumRows(),
	                         in_value.NumCols() + 1, kUndefined);
	    in_value_temp.Range(0, in_value.NumRows(),
	                        0, in_value.NumCols()).CopyFromMat(in_value);

	    // Add the 1.0 at the end of each row "in_value_temp"
	    in_value_temp.Range(0, in_value.NumRows(),
	                        in_value.NumCols(), 1).Set(1.0);

	    CuMatrix<BaseFloat> out_deriv_temp(out_deriv);

	    CuMatrix<BaseFloat> row_products(2,in_value.NumRows());
	    CuSubVector<BaseFloat> in_row_products(row_products, 0),
	        out_row_products(row_products, 1);

	    // These "scale" values get will get multiplied into the learning rate (faster
	      // than having the matrices scaled inside the preconditioning code).
	      BaseFloat in_scale, out_scale;

	      preconditioner_in_.PreconditionDirections(&in_value_temp, &in_row_products,
	                                                &in_scale);
	      preconditioner_out_.PreconditionDirections(&out_deriv_temp, &out_row_products,
	                                                 &out_scale);

	      // "scale" is a scaling factor coming from the PreconditionDirections calls
	      // (it's faster to have them output a scaling factor than to have them scale
	      // their outputs).
	      BaseFloat scale = in_scale * out_scale;
	      BaseFloat minibatch_scale = 1.0;

	      if (max_change_per_sample_ > 0.0)
	        minibatch_scale = GetScalingFactor(in_row_products, scale,
	                                           &out_row_products);

	      CuSubMatrix<BaseFloat> in_value_precon_part(in_value_temp,
	                                                  0, in_value_temp.NumRows(),
	                                                  0, in_value_temp.NumCols() - 1);
	      // this "precon_ones" is what happens to the vector of 1's representing
	      // offsets, after multiplication by the preconditioner.
	      //CuVector<BaseFloat> precon_ones(in_value_temp.NumRows());

	      //precon_ones.CopyColFromMat(in_value_temp, in_value_temp.NumCols() - 1);

	      local_lrate = scale * minibatch_scale * lr;
	      local_lrate_bias = scale * minibatch_scale * lr_bias;

	  // compute gradient (incl. momentum)
	  linearity_corr_.AddMatMat(1.0, out_deriv_temp, kTrans, in_value_precon_part, kNoTrans, mmt);
	  bias_corr_.AddRowSumMat(1.0, out_deriv_temp, mmt);
	  // l2 regularization
	  if (l2 != 0.0) {
	    linearity_.AddMat(-lr*l2*num_frames, linearity_);
	  }
	  // l1 regularization
	  if (l1 != 0.0) {
	    cu::RegularizeL1(&linearity_, &linearity_corr_, lr*l1*num_frames, lr);
	  }
}
void AffinePreconditionedOnlineTransform::UpdateGradient() {
	// update
	  linearity_.AddMat(-local_lrate, linearity_corr_);
	  bias_.AddVec(-local_lrate_bias, bias_corr_);
	  // max-norm
	  if (max_norm_ > 0.0) {
	    CuMatrix<BaseFloat> lin_sqr(linearity_);
	    lin_sqr.MulElements(linearity_);
	    CuVector<BaseFloat> l2(OutputDim());
	    l2.AddColSumMat(1.0, lin_sqr, 0.0);
	    l2.ApplyPow(0.5); // we have per-neuron L2 norms
	    CuVector<BaseFloat> scl(l2);
	    scl.Scale(1.0/max_norm_);
	    scl.ApplyFloor(1.0);
	    scl.InvertElements();
	    linearity_.MulRowsVec(scl); // shink to sphere!
	  }
}

void AffinePreconditionedOnlineTransform::Update(const CuMatrixBase<BaseFloat> &in_value, const CuMatrixBase<BaseFloat> &out_deriv) {
  // we use following hyperparameters from the option class
  const BaseFloat lr = opts_.learn_rate * learn_rate_coef_;
  const BaseFloat lr_bias = opts_.learn_rate * bias_learn_rate_coef_;
  const BaseFloat mmt = opts_.momentum;
  const BaseFloat l2 = opts_.l2_penalty;
  const BaseFloat l1 = opts_.l1_penalty;
  // we will also need the number of frames in the mini-batch
  const int32 num_frames = in_value.NumRows();

  CuMatrix<BaseFloat> in_value_temp;

    in_value_temp.Resize(in_value.NumRows(),
                         in_value.NumCols() + 1, kUndefined);
    in_value_temp.Range(0, in_value.NumRows(),
                        0, in_value.NumCols()).CopyFromMat(in_value);

    // Add the 1.0 at the end of each row "in_value_temp"
    in_value_temp.Range(0, in_value.NumRows(),
                        in_value.NumCols(), 1).Set(1.0);

    CuMatrix<BaseFloat> out_deriv_temp(out_deriv);

    CuMatrix<BaseFloat> row_products(2,in_value.NumRows());
    CuSubVector<BaseFloat> in_row_products(row_products, 0),
        out_row_products(row_products, 1);

    // These "scale" values get will get multiplied into the learning rate (faster
      // than having the matrices scaled inside the preconditioning code).
      BaseFloat in_scale, out_scale;

      preconditioner_in_.PreconditionDirections(&in_value_temp, &in_row_products,
                                                &in_scale);
      preconditioner_out_.PreconditionDirections(&out_deriv_temp, &out_row_products,
                                                 &out_scale);

      // "scale" is a scaling factor coming from the PreconditionDirections calls
      // (it's faster to have them output a scaling factor than to have them scale
      // their outputs).
      BaseFloat scale = in_scale * out_scale;
      BaseFloat minibatch_scale = 1.0;

      if (max_change_per_sample_ > 0.0)
        minibatch_scale = GetScalingFactor(in_row_products, scale,
                                           &out_row_products);

      CuSubMatrix<BaseFloat> in_value_precon_part(in_value_temp,
                                                  0, in_value_temp.NumRows(),
                                                  0, in_value_temp.NumCols() - 1);
      // this "precon_ones" is what happens to the vector of 1's representing
      // offsets, after multiplication by the preconditioner.
      //CuVector<BaseFloat> precon_ones(in_value_temp.NumRows());

      //precon_ones.CopyColFromMat(in_value_temp, in_value_temp.NumCols() - 1);

      BaseFloat local_lrate = scale * minibatch_scale * lr;
      BaseFloat local_lrate_bias = scale * minibatch_scale * lr_bias;

  // compute gradient (incl. momentum)
  linearity_corr_.AddMatMat(1.0, out_deriv_temp, kTrans, in_value_precon_part, kNoTrans, mmt);
  bias_corr_.AddRowSumMat(1.0, out_deriv_temp, mmt);
  // l2 regularization
  if (l2 != 0.0) {
    linearity_.AddMat(-lr*l2*num_frames, linearity_);
  }
  // l1 regularization
  if (l1 != 0.0) {
    cu::RegularizeL1(&linearity_, &linearity_corr_, lr*l1*num_frames, lr);
  }
  // update
  linearity_.AddMat(-local_lrate, linearity_corr_);
  bias_.AddVec(-local_lrate_bias, bias_corr_);
  // max-norm
  if (max_norm_ > 0.0) {
    CuMatrix<BaseFloat> lin_sqr(linearity_);
    lin_sqr.MulElements(linearity_);
    CuVector<BaseFloat> l2(OutputDim());
    l2.AddColSumMat(1.0, lin_sqr, 0.0);
    l2.ApplyPow(0.5); // we have per-neuron L2 norms
    CuVector<BaseFloat> scl(l2);
    scl.Scale(1.0/max_norm_);
    scl.ApplyFloor(1.0);
    scl.InvertElements();
    linearity_.MulRowsVec(scl); // shink to sphere!
  }
}

void AffinePreconditionedOnlineTransform::Init(
    BaseFloat learning_rate,
    int input_dim, int output_dim,
    BaseFloat param_stddev, BaseFloat bias_stddev,
    int rank_in, int rank_out, int update_period,
    BaseFloat num_samples_history, BaseFloat alpha,
    BaseFloat max_change_per_sample) {
	//UpdatableComponent::Init(learning_rate);
	//linear_params_.Resize(output_dim, input_dim);
	//bias_params_.Resize(output_dim);
	KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0 &&
				   bias_stddev >= 0.0);
	//linear_params_.SetRandn(); // sets to random normally distributed noise.
	//linear_params_.Scale(param_stddev);
	//bias_params_.SetRandn();
	//bias_params_.Scale(bias_stddev);
	rank_in_ = rank_in;
	rank_out_ = rank_out;
	update_period_ = update_period;
	num_samples_history_ = num_samples_history;
	alpha_ = alpha;
	SetPreconditionerConfigs();
	KALDI_ASSERT(max_change_per_sample >= 0.0);
	max_change_per_sample_ = max_change_per_sample;
}


BaseFloat AffinePreconditionedOnlineTransform::GetScalingFactor(
    const CuVectorBase<BaseFloat> &in_products,
    BaseFloat learning_rate_scale,
    CuVectorBase<BaseFloat> *out_products) {
	  const BaseFloat lr = opts_.learn_rate * learn_rate_coef_;
  static int scaling_factor_printed = 0;
  int32 minibatch_size = in_products.Dim();

  out_products->MulElements(in_products);
  out_products->ApplyPow(0.5);
  BaseFloat prod_sum = out_products->Sum();
  BaseFloat tot_change_norm = learning_rate_scale * lr * prod_sum,
      max_change_norm = max_change_per_sample_ * minibatch_size;
  // tot_change_norm is the product of norms that we are trying to limit
  // to max_value_.
  KALDI_ASSERT(tot_change_norm - tot_change_norm == 0.0 && "NaN in backprop");
  KALDI_ASSERT(tot_change_norm >= 0.0);
  if (tot_change_norm <= max_change_norm) return 1.0;
  else {
    BaseFloat factor = max_change_norm / tot_change_norm;
    if (scaling_factor_printed < 10) {
      KALDI_LOG << "Limiting step size using scaling factor "
                << factor << ", for component index "; // << Index();
      scaling_factor_printed++;
    }
    return factor;
  }
}

void AffinePreconditionedOnlineTransform::SetPreconditionerConfigs() {
  preconditioner_in_.SetRank(rank_in_);
  preconditioner_in_.SetNumSamplesHistory(num_samples_history_);
  preconditioner_in_.SetAlpha(alpha_);
  preconditioner_in_.SetUpdatePeriod(update_period_);
  preconditioner_out_.SetRank(rank_out_);
  preconditioner_out_.SetNumSamplesHistory(num_samples_history_);
  preconditioner_out_.SetAlpha(alpha_);
  preconditioner_out_.SetUpdatePeriod(update_period_);
}


} // namespace nnet0
} // namespace kaldi
