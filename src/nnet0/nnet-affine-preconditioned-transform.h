/*
 * nnet-affine-preconditioned-transform.h
 *
 *  Created on: 2015年4月13日
 *      Author: wd007
 */

#ifndef NNET_NNET_AFFINE_PRECONDITIONED_TRANSFORM_H_
#define NNET_NNET_AFFINE_PRECONDITIONED_TRANSFORM_H_

#include "nnet2/nnet-precondition-online.h"
#include "nnet0/nnet-affine-transform.h"

namespace kaldi {
namespace nnet0 {

class AffinePreconditionedOnlineTransform : public AffineTransform{

public:

		AffinePreconditionedOnlineTransform(int32 dim_in, int32 dim_out)
	    : AffineTransform(dim_in, dim_out)
	  {
		BaseFloat param_stddev = 1.0 / std::sqrt(dim_in),
				bias_stddev = 1.0;		// both not quite meaningful now
		int rank_in = 20, rank_out = 80, update_period = 4;
		BaseFloat num_sample_alpha = 2000.0, alpha = 4.0, max_change_per_sample = 0.1;
		Init(opts_.learn_rate, dim_in, dim_out, param_stddev, bias_stddev,
				rank_in, rank_out, update_period, num_sample_alpha, alpha, max_change_per_sample);	
	  }
	  ~AffinePreconditionedOnlineTransform()
	  { }

	  ComponentType GetType() const { return kAffinePreconditionedOnlineTransform; }

	  void Init(BaseFloat learning_rate,
	      int input_dim, int output_dim,
	      BaseFloat param_stddev, BaseFloat bias_stddev,
	      int rank_in, int rank_out, int update_period,
	      BaseFloat num_samples_history, BaseFloat alpha,
	      BaseFloat max_change_per_sample);

	  // This constructor is used when converting neural networks partway through
	   // training, from AffineComponent or AffineComponentPreconditioned to
	   // AffineComponentPreconditionedOnline.
	  AffinePreconditionedOnlineTransform(const AffineTransform &orig,
	                                       int32 rank_in, int32 rank_out,
	                                       int32 update_period,
	                                       BaseFloat eta, BaseFloat alpha);

	  virtual void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff);
	  virtual void Gradient(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff);
	  virtual void UpdateGradient();

private:

	  // Configs for preconditioner.  The input side tends to be better conditioned ->
	  // smaller rank needed, so make them separately configurable.
	  int32 rank_in_;
	  int32 rank_out_;
	  int32 update_period_;
	  BaseFloat num_samples_history_;
	  BaseFloat alpha_;
	  BaseFloat local_lrate;
	  BaseFloat local_lrate_bias;

	  nnet2::OnlinePreconditioner preconditioner_in_;

	  nnet2::OnlinePreconditioner preconditioner_out_;

	  BaseFloat max_change_per_sample_;
	  // If > 0, max_change_per_sample_ this is the maximum amount of parameter
	  // change (in L2 norm) that we allow per sample, averaged over the minibatch.
	  // This was introduced in order to control instability.
	  // Instead of the exact L2 parameter change, for
	  // efficiency purposes we limit a bound on the exact
	  // change.  The limit is applied via a constant <= 1.0
	  // for each minibatch, A suitable value might be, for
	  // example, 10 or so; larger if there are more
	  // parameters.

	  /// The following function is only called if max_change_per_sample_ > 0, it returns a
	  /// scaling factor alpha <= 1.0 (1.0 in the normal case) that enforces the
	  /// "max-change" constraint.  "in_products" is the inner product with itself
	  /// of each row of the matrix of preconditioned input features; "out_products"
	  /// is the same for the output derivatives.  gamma_prod is a product of two
	  /// scalars that are output by the preconditioning code (for the input and
	  /// output), which we will need to multiply into the learning rate.
	  /// out_products is a pointer because we modify it in-place.
	  BaseFloat GetScalingFactor(const CuVectorBase<BaseFloat> &in_products,
	                             BaseFloat gamma_prod,
	                             CuVectorBase<BaseFloat> *out_products);

	  // Sets the configs rank, alpha and eta in the preconditioner objects,
	  // from the class variables.
	  void SetPreconditionerConfigs();



};


} // namespace nnet0
} // namespace kaldi



#endif /* NNET_NNET_AFFINE_PRECONDITIONED_TRANSFORM_H_ */
