// nnet0/nnet-compute-sequential-parallel.cc

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

#include <deque>
#include "hmm/posterior.h"
#include "lat/lattice-functions.h"
#include "thread/kaldi-semaphore.h"
#include "thread/kaldi-mutex.h"
#include "thread/kaldi-thread.h"

#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include "nnet0/nnet-utils.h"

#include "cudamatrix/cu-device.h"
#include "base/kaldi-types.h"


#include "nnet0/nnet-affine-transform.h"
#include "nnet0/nnet-affine-preconditioned-transform.h"
#include "nnet0/nnet-model-merge-function.h"
#include "nnet0/nnet-activation.h"
#include "nnet0/nnet-example.h"

#include "nnet0/nnet-compute-sequential-parallel.h"

namespace kaldi {
namespace nnet0 {

class SeqTrainParallelClass: public MultiThreadable {

private:
    const NnetSequentialUpdateOptions *opts;
    NnetModelSync *model_sync;

	std::string feature_transform,
				model_filename,
				si_model_filename,
				transition_model_filename;

	std::string den_lat_rspecifier,
				num_ali_rspecifier,
				feature_rspecifier;

	ExamplesRepository *repository_;
    NnetSequentialStats *stats_;

    const NnetTrainOptions *trn_opts;
    const PdfPriorOptions *prior_opts;
    const NnetParallelOptions *parallel_opts;

    BaseFloat 	acoustic_scale,
    			lm_scale,
		old_acoustic_scale,
    			kld_scale,
				frame_smooth;
    kaldi::int32 max_frames;
    bool drop_frames;
    std::string use_gpu;
    int32 num_threads;
    int32 time_shift;

	bool one_silence_class;
	BaseFloat boost;
	std::string silence_phones_str;
	std::string criterion;


 public:
  // This constructor is only called for a temporary object
  // that we pass to the RunMultiThreaded function.
	SeqTrainParallelClass(const NnetSequentialUpdateOptions *opts,
			NnetModelSync *model_sync,
			std::string feature_transform,
			std::string	model_filename,
			std::string transition_model_filename,
			std::string den_lat_rspecifier,
			std::string num_ali_rspecifier,
			ExamplesRepository *repository,
			Nnet *nnet,
			NnetSequentialStats *stats):
				opts(opts),
				model_sync(model_sync),
				feature_transform(feature_transform),
				model_filename(model_filename),
				transition_model_filename(transition_model_filename),
				den_lat_rspecifier(den_lat_rspecifier),
				num_ali_rspecifier(num_ali_rspecifier),
				repository_(repository),
				stats_(stats)
 	 		{
				trn_opts = opts->trn_opts;
				prior_opts = opts->prior_opts;
				parallel_opts = opts->parallel_opts;

				acoustic_scale = opts->acoustic_scale;
				lm_scale = opts->lm_scale;
				old_acoustic_scale = opts->old_acoustic_scale;
				kld_scale = opts->kld_scale;
				frame_smooth = opts->frame_smooth;
				max_frames = opts->max_frames;
				drop_frames = opts->drop_frames;
				use_gpu = opts->use_gpu;
				si_model_filename = opts->si_model_filename;

				num_threads = parallel_opts->num_threads;
				time_shift = opts->targets_delay;

				one_silence_class = opts->one_silence_class;  // Affects MPE/SMBR>
				boost = opts->boost; // for MMI, boosting factor (would be Boosted MMI)... e.g. 0.1.
				silence_phones_str= opts->silence_phones_str; // colon-separated list of integer ids of silence phones,
				                                  // for MPE/SMBR only.
				criterion = opts->criterion;

 	 		}


	void LatticeAcousticRescore(const Matrix<BaseFloat> &log_like,
	                            const TransitionModel &trans_model,
	                            const std::vector<int32> &state_times,
	                            Lattice *lat) {
	  kaldi::uint64 props = lat->Properties(fst::kFstProperties, false);
	  if (!(props & fst::kTopSorted))
	    KALDI_ERR << "Input lattice must be topologically sorted.";

	  KALDI_ASSERT(!state_times.empty());
	  std::vector<std::vector<int32> > time_to_state(log_like.NumRows());
	  for (size_t i = 0; i < state_times.size(); i++) {
	    KALDI_ASSERT(state_times[i] >= 0);
	    if (state_times[i] < log_like.NumRows())  // end state may be past this..
	      time_to_state[state_times[i]].push_back(i);
	    else
	      KALDI_ASSERT(state_times[i] == log_like.NumRows()
	                   && "There appears to be lattice/feature mismatch.");
	  }

	  for (int32 t = 0; t < log_like.NumRows(); t++) {
	    for (size_t i = 0; i < time_to_state[t].size(); i++) {
	      int32 state = time_to_state[t][i];
	      for (fst::MutableArcIterator<Lattice> aiter(lat, state); !aiter.Done();
	           aiter.Next()) {
	        LatticeArc arc = aiter.Value();
	        int32 trans_id = arc.ilabel;
	        if (trans_id != 0) {  // Non-epsilon input label on arc
	          int32 pdf_id = trans_model.TransitionIdToPdf(trans_id);
	          arc.weight.SetValue2(-log_like(t, pdf_id) + arc.weight.Value2());
	          aiter.SetValue(arc);
	        }
	      }
	    }
	  }
	}

	  // Note, frames are numbered from zero. Here "tid" means token id, the indexes of the
	  // CTC label tokens. When we compile the search graph, the tokens are indexed from 1
	  // because 0 is always occupied by <eps>. However, in the softmax layer of the RNN
	  // model, CTC tokens are indexed from 0. Thus, we simply shift "tid" by 1, to solve
	  // the mismatch.

	void LatticeAcousticRescoreCTC(const Matrix<BaseFloat> &log_like,
	                            const std::vector<int32> &state_times,
	                            Lattice *lat) {
	  kaldi::uint64 props = lat->Properties(fst::kFstProperties, false);
	  if (!(props & fst::kTopSorted))
	    KALDI_ERR << "Input lattice must be topologically sorted.";

	  KALDI_ASSERT(!state_times.empty());
	  std::vector<std::vector<int32> > time_to_state(log_like.NumRows());
	  for (size_t i = 0; i < state_times.size(); i++) {
	    KALDI_ASSERT(state_times[i] >= 0);
	    if (state_times[i] < log_like.NumRows())  // end state may be past this..
	      time_to_state[state_times[i]].push_back(i);
	    else
	      KALDI_ASSERT(state_times[i] == log_like.NumRows()
	                   && "There appears to be lattice/feature mismatch.");
	  }

	  for (int32 t = 0; t < log_like.NumRows(); t++) {
	    for (size_t i = 0; i < time_to_state[t].size(); i++) {
	      int32 state = time_to_state[t][i];
	      for (fst::MutableArcIterator<Lattice> aiter(lat, state); !aiter.Done();
	           aiter.Next()) {
	        LatticeArc arc = aiter.Value();
	        int32 trans_id = arc.ilabel;
	        if (trans_id != 0) {  // Non-epsilon input label on arc
	          int32 pdf_id = trans_id -1; //trans_model.TransitionIdToPdf(trans_id);
	          arc.weight.SetValue2(-log_like(t, pdf_id) + arc.weight.Value2());
	          aiter.SetValue(arc);
	        }
	      }
	    }
	  }
	}

	void inline MPEObj(Matrix<BaseFloat> &nnet_out_h, MatrixBase<BaseFloat> &nnet_diff_h,
				TransitionModel *trans_model, SequentialNnetExample *example,
				std::vector<int32> &silence_phones, double &total_frame_acc, int32 num_done,
				Matrix<BaseFloat> &soft_nnet_out_h, Matrix<BaseFloat> &si_nnet_out_h)
	{
		std::string utt = example->utt;
		const std::vector<int32> &num_ali = example->num_ali;
		Lattice &den_lat = example->den_lat;
		std::vector<int32> &state_times = example->state_times;

		CuMatrix<BaseFloat> nnet_diff;
		int num_frames, num_pdfs;
	    double utt_frame_acc;


		num_frames = nnet_out_h.NumRows();
		num_pdfs = nnet_out_h.NumCols();

		if (this->kld_scale > 0)
			si_nnet_out_h.AddMat(-1.0, soft_nnet_out_h);


		// 4) rescore the latice
		if (trans_model != NULL)
			LatticeAcousticRescore(nnet_out_h, *trans_model, state_times, &den_lat);
		else
			LatticeAcousticRescoreCTC(nnet_out_h, state_times, &den_lat);

		if (acoustic_scale != 1.0 || lm_scale != 1.0)
			fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale), &den_lat);

	      kaldi::Posterior post;
	    if (trans_model != NULL)
	    {
	      if (this->criterion == "smbr") {  // use state-level accuracies, i.e. sMBR estimation
	        utt_frame_acc = LatticeForwardBackwardMpeVariants(
	            *trans_model, silence_phones, den_lat, num_ali, "smbr",
	            one_silence_class, &post);
	      } else {  // use phone-level accuracies, i.e. MPFE (minimum phone frame error)
	        utt_frame_acc = LatticeForwardBackwardMpeVariants(
	            *trans_model, silence_phones, den_lat, num_ali, "mpfe",
	            one_silence_class, &post);
	      }
	    }
	    else
	    {
		      if (this->criterion == "smbr") {  // use state-level accuracies, i.e. sMBR estimation
		        utt_frame_acc = LatticeForwardBackwardMpeVariantsCTC(
		            silence_phones, den_lat, num_ali, "smbr",
		            one_silence_class, &post);
		      } else {  // use phone-level accuracies, i.e. MPFE (minimum phone frame error)
		        utt_frame_acc = LatticeForwardBackwardMpeVariantsCTC(
		            silence_phones, den_lat, num_ali, "mpfe",
		            one_silence_class, &post);
		      }
	    }

	      // 6) convert the Posterior to a matrix,
	    if (trans_model != NULL)
	    	PosteriorToMatrixMapped(post, *trans_model, &nnet_diff);
	    else
	    	PosteriorToMatrixMappedCTC(post, num_pdfs, &nnet_diff);

	      nnet_diff.Scale(-1.0); // need to flip the sign of derivative,

	      nnet_diff.CopyToMat(&nnet_diff_h);

	       // 8) subtract the pdf-Viterbi-path
	      if (this->frame_smooth > 0)
	      {
	    	  for(int32 t=0; t<nnet_diff_h.NumRows(); t++) {
	    		  int32 pdf = trans_model!=NULL ? trans_model->TransitionIdToPdf(num_ali[t]) : num_ali[t]-1;
	    	  	  soft_nnet_out_h(t, pdf) -= 1.0;
	    	  }

	    	   nnet_diff_h.Scale(1-frame_smooth);
	    	   nnet_diff_h.AddMat(frame_smooth*0.1, soft_nnet_out_h);
	       }

			if (this->kld_scale > 0)
	        {
	        	nnet_diff_h.Scale(1.0-kld_scale);
	        	//-kld_scale means gradient descent direction.
				nnet_diff_h.AddMat(-kld_scale, si_nnet_out_h);
	        }



	      KALDI_VLOG(1) << "Lattice #" << num_done + 1 << " processed"
	                    << " (" << utt << "): found " << den_lat.NumStates()
	                    << " states and " << fst::NumArcs(den_lat) << " arcs.";

	      KALDI_VLOG(1) << "Utterance " << utt << ": Average frame accuracy = "
	                    << (utt_frame_acc/num_frames) << " over " << num_frames
	                    << " frames,"
	                    << " diff-range(" << nnet_diff.Min() << "," << nnet_diff.Max() << ")";

	      // increase time counter
	      total_frame_acc += utt_frame_acc;
	}



	void inline MMIObj(Matrix<BaseFloat> &nnet_out_h, MatrixBase<BaseFloat> &nnet_diff_h,
				TransitionModel *trans_model, SequentialNnetExample *example,
				double &total_mmi_obj, double &total_post_on_ali, int32 &num_frm_drop, int32 num_done,
				Matrix<BaseFloat> &soft_nnet_out_h, Matrix<BaseFloat> &si_nnet_out_h)
	{
		std::string utt = example->utt;
		const std::vector<int32> &num_ali = example->num_ali;
		Lattice &den_lat = example->den_lat;
		std::vector<int32> &state_times = example->state_times;

		int num_frames; // num_pdfs;
	    double lat_like; // total likelihood of the lattice
	    double lat_ac_like; // acoustic likelihood weighted by posterior.
	    double mmi_obj = 0.0, post_on_ali = 0.0;


		num_frames = nnet_out_h.NumRows();
		//num_pdfs = nnet_out_h.NumCols();

		if (this->kld_scale > 0)
			si_nnet_out_h.AddMat(-1.0, soft_nnet_out_h);


		// 4) rescore the latice
		if (trans_model != NULL)
			LatticeAcousticRescore(nnet_out_h, *trans_model, state_times, &den_lat);
		else
			LatticeAcousticRescoreCTC(nnet_out_h, state_times, &den_lat);

		if (acoustic_scale != 1.0 || lm_scale != 1.0)
			fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale), &den_lat);

	       // 5) get the posteriors
	       kaldi::Posterior post;
	       lat_like = kaldi::LatticeForwardBackward(den_lat, &post, &lat_ac_like);

	       // 6) convert the Posterior to a matrix
	       //nnet_diff_h.Resize(num_frames, num_pdfs, kSetZero);
	       for (int32 t = 0; t < post.size(); t++) {
	         for (int32 arc = 0; arc < post[t].size(); arc++) {
	           int32 pdf = trans_model != NULL ? trans_model->TransitionIdToPdf(post[t][arc].first) : post[t][arc].first-1;
	           nnet_diff_h(t, pdf) += post[t][arc].second;
	         }
	       }


	       // 7) Calculate the MMI-objective function
	       // Calculate the likelihood of correct path from acoustic score,
	       // the denominator likelihood is the total likelihood of the lattice.
	       double path_ac_like = 0.0;
	       for(int32 t=0; t<num_frames; t++) {
	         int32 pdf = trans_model != NULL ? trans_model->TransitionIdToPdf(num_ali[t]) : num_ali[t]-1;
	         path_ac_like += nnet_out_h(t,pdf);
	       }
	       path_ac_like *= acoustic_scale;
	       mmi_obj = path_ac_like - lat_like;
	       //
	       // Note: numerator likelihood does not include graph score,
	       // while denominator likelihood contains graph scores.
	       // The result is offset at the MMI-objective.
	       // However the offset is constant for given alignment,
	       // so it is not harmful.

	       // Sum the den-posteriors under the correct path:
	       post_on_ali = 0.0;
	       for(int32 t=0; t<num_frames; t++) {
	         int32 pdf = trans_model != NULL ? trans_model->TransitionIdToPdf(num_ali[t]) : num_ali[t]-1;
	         double posterior = nnet_diff_h(t, pdf);
	         post_on_ali += posterior;
	       }

	       // Report
	       KALDI_VLOG(1) << "Lattice #" << num_done + 1 << " processed"
	                     << " (" << utt << "): found " << den_lat.NumStates()
	                     << " states and " << fst::NumArcs(den_lat) << " arcs.";

	       KALDI_VLOG(1) << "Utterance " << utt << ": Average MMI obj. value = "
	                     << (mmi_obj/num_frames) << " over " << num_frames
	                     << " frames."
	                     << " (Avg. den-posterior on ali " << post_on_ali/num_frames << ")";


	       // 7a) Search for the frames with num/den mismatch
	       int32 frm_drop = 0;
	       std::vector<int32> frm_drop_vec;
	       for(int32 t=0; t<num_frames; t++) {
	         int32 pdf = trans_model != NULL ? trans_model->TransitionIdToPdf(num_ali[t]) : num_ali[t]-1;
	         double posterior = nnet_diff_h(t, pdf);
	         if(posterior < 1e-20) {
	           frm_drop++;
	           frm_drop_vec.push_back(t);
	         }
	       }

	       // 8) subtract the pdf-Viterbi-path
	       for(int32 t=0; t<nnet_diff_h.NumRows(); t++) {
	         int32 pdf = trans_model != NULL ? trans_model->TransitionIdToPdf(num_ali[t]) : num_ali[t]-1;
	         nnet_diff_h(t, pdf) -= 1.0;

	         //frame-smoothing
	         if (this->frame_smooth > 0)
	        	 soft_nnet_out_h(t, pdf) -= 1.0;
	       }

	       if (this->frame_smooth > 0)
	       {
	    	   nnet_diff_h.Scale(1-frame_smooth);
	    	   nnet_diff_h.AddMat(frame_smooth*0.1, soft_nnet_out_h);
	       }

			if (this->kld_scale > 0)
	        {
	        	nnet_diff_h.Scale(1.0-kld_scale);
	        	//-kld_scale means gradient descent direction.
				nnet_diff_h.AddMat(-kld_scale, si_nnet_out_h);

				KALDI_VLOG(1) << "likelihood of correct path = "  << path_ac_like
								<< " total likelihood of the lattice = " << lat_like;
	        }

	       // 9) Drop mismatched frames from the training by zeroing the derivative
	       if(drop_frames) {
	         for(int32 i=0; i<frm_drop_vec.size(); i++) {
	           nnet_diff_h.Row(frm_drop_vec[i]).Set(0.0);
	         }
	         num_frm_drop += frm_drop;
	       }
	       // Report the frame dropping
	       if (frm_drop > 0) {
	         std::stringstream ss;
	         ss << (drop_frames?"Dropped":"[dropping disabled] Would drop")
	            << " frames in " << utt << " " << frm_drop << "/" << num_frames << ",";
	         //get frame intervals from vec frm_drop_vec
	         ss << " intervals :";
	         //search for streaks of consecutive numbers:
	         int32 beg_streak=frm_drop_vec[0];
	         int32 len_streak=0;
	         int32 i;
	         for(i=0; i<frm_drop_vec.size(); i++,len_streak++) {
	           if(beg_streak + len_streak != frm_drop_vec[i]) {
	             ss << " " << beg_streak << ".." << frm_drop_vec[i-1] << "frm";
	             beg_streak = frm_drop_vec[i];
	             len_streak = 0;
	           }
	         }
	         ss << " " << beg_streak << ".." << frm_drop_vec[i-1] << "frm";
	         //print
	         KALDI_WARN << ss.str();
	       }

	       // 10) backpropagate through the nnet
	       //nnet_diff.Resize(num_frames, num_pdfs, kUndefined);
	       //nnet_diff.CopyFromMat(nnet_diff_h);

			       total_mmi_obj += mmi_obj;
	       total_post_on_ali += post_on_ali;

	}
	  // This does the main function of the class.
	void operator ()()
	{

		model_sync->LockModel();
		int thread_idx = this->thread_id_;

	    // Select the GPU
	#if HAVE_CUDA == 1
	    if (parallel_opts->num_procs > 1)
	    {
	    	//int32 thread_idx = model_sync->GetThreadIdx();
	    	KALDI_LOG << "MyId: " << parallel_opts->myid << "  ThreadId: " << thread_idx;
	    	CuDevice::Instantiate().MPISelectGpu(model_sync->gpuinfo_, model_sync->win, thread_idx, this->num_threads);
	    	for (int i = 0; i< this->num_threads*parallel_opts->num_procs; i++)
	    	{
	    		KALDI_LOG << model_sync->gpuinfo_[i].hostname << "  myid: " << model_sync->gpuinfo_[i].myid
	    					<< "  gpuid: " << model_sync->gpuinfo_[i].gpuid;
	    	}
	    }
	    else
	    	CuDevice::Instantiate().SelectGpu();

	    //CuDevice::Instantiate().DisableCaching();
	#endif

	    model_sync->UnlockModel();

	    // Read the class-frame-counts, compute priors
	    PdfPrior log_prior(*prior_opts);


		int32 time_shift = opts->targets_delay;
		const PdfPriorOptions *prior_opts = opts->prior_opts;
		int32 num_stream = opts->num_stream;
		int32 batch_size = opts->batch_size;
		int32 frame_limit = opts->frame_limit;
		int32 skip_frames = opts->skip_frames;


	    int32 num_done = 0, num_frm_drop = 0;

	    int32 rank_in = 20, rank_out = 80, update_period = 4;
	    BaseFloat num_samples_history = 2000.0;
	    BaseFloat alpha = 4.0;

	    kaldi::int64 total_frames = 0;
	    double total_mmi_obj = 0.0;
	    double total_post_on_ali = 0.0;
	    double total_frame_acc = 0.0;

	    // Read transition model
	    TransitionModel *trans_model = NULL;
	    if (transition_model_filename != "")
	    {
	    	trans_model = new TransitionModel();
	    	ReadKaldiObject(transition_model_filename, trans_model);
	    }
        else skip_frames = 1; // for CTC

		Nnet nnet_transf;
	    if (feature_transform != "") {
	      nnet_transf.Read(feature_transform);
	    }

	    Nnet nnet;
	    nnet.Read(model_filename);
	    // using activations directly: remove softmax, if present
	    if (nnet.GetComponent(nnet.NumComponents()-1).GetType() ==
	        kaldi::nnet0::Component::kSoftmax) {
	      KALDI_LOG << "Removing softmax from the nnet " << model_filename;
	      nnet.RemoveComponent(nnet.NumComponents()-1);
	    } else {
	      KALDI_LOG << "The nnet was without softmax " << model_filename;
	    }
	    //if (opts->num_procs > 1 || opts->use_psgd)
	    if (opts->use_psgd)
	    	nnet.SwitchToOnlinePreconditioning(rank_in, rank_out, update_period, num_samples_history, alpha);

	    nnet.SetTrainOptions(*trn_opts);

	    Nnet si_nnet, softmax;
	    if (this->kld_scale > 0)
	    {
	    	si_nnet.Read(si_model_filename);
	    }

	    if (this->kld_scale > 0 || frame_smooth > 0)
	    {
	    	KALDI_LOG << "KLD model Appending the softmax ...";
	    	softmax.AppendComponent(new Softmax(nnet.OutputDim(),nnet.OutputDim()));
        }

	    std::vector<int32> silence_phones;
	    if (this->criterion != "mmi")
	    {
		    if (!kaldi::SplitStringToIntegers(silence_phones_str, ":", false, &silence_phones))
		      KALDI_ERR << "Invalid silence-phones string " << silence_phones_str;
		    kaldi::SortAndUniq(&silence_phones);
		    if (silence_phones.empty())
		      KALDI_LOG << "No silence phones specified.";
	    }

	    model_sync->Initialize(&nnet);

	    Timer time;
	    double time_now = 0;

		CuMatrix<BaseFloat> cu_feats, feats_transf, nnet_out, nnet_diff, si_nnet_out, soft_nnet_out;
		Matrix<BaseFloat> nnet_out_h, nnet_diff_h, si_nnet_out_h, soft_nnet_out_h, *p_nnet_diff_h = NULL;


		ModelMergeFunction *p_merge_func = model_sync->GetModelMergeFunction();

	    std::vector<std::string> keys(num_stream);
	    std::vector<Matrix<BaseFloat> > feats(num_stream);
	    std::vector<int> curt(num_stream, 0);
	    std::vector<int> lent(num_stream, 0);
	    std::vector<int> frame_num_utt(num_stream, 0);
	    std::vector<int> new_utt_flags;

	    std::vector<Matrix<BaseFloat> > utt_nnet_out_h(num_stream),
	    		utt_si_nnet_out_h(num_stream), utt_soft_nnet_out_h(num_stream);
	    std::vector<int> utt_curt(num_stream, 0);
	    std::vector<SequentialNnetExample *> utt_examples(num_stream);
	    std::vector<bool> utt_copied(num_stream, 0);

	    std::vector<Matrix<BaseFloat> > diff_utt_feats(num_stream);
	    std::vector<int> diff_curt(num_stream, 0);

	    // bptt batch buffer
	    int32 feat_dim = nnet.InputDim();
	    int32 out_dim = nnet.OutputDim();

	    Matrix<BaseFloat> feat;
	    Matrix<BaseFloat> diff_feat;
	    Matrix<BaseFloat> nnet_out_host, si_nnet_out_host, soft_nnet_out_host;


		//double t1, t2, t3, t4;
		int32 update_frames = 0;
		int32 num_frames = 0;
		int32 cur_stream_num = 0;
		int32 num_dump = 0, num_skip;
		num_skip = opts->skip_inner ? skip_frames : 1;
        num_skip = opts->skip_frames == skip_frames ? num_skip : opts->skip_frames; // for CTC
        frame_limit *= num_skip;

		SequentialNnetExample *example = NULL;

		while (1)
		{
				if (num_stream >= 1)
				{
					int32 s = 0, max_frame_num = 0, cur_frames = 0;
					cur_stream_num = 0; num_frames = 0;
					p_nnet_diff_h = &diff_feat;
					

					if (NULL == example)
						example = dynamic_cast<SequentialNnetExample*>(repository_->ProvideExample());

					while (s < num_stream && cur_frames < frame_limit && NULL != example)
					{

						std::string key = example->utt;
						Matrix<BaseFloat> &mat = example->input_frames;

						if ((s+1)*mat.NumRows() > frame_limit || (s+1)*max_frame_num > frame_limit) break;

						if (max_frame_num < mat.NumRows()) max_frame_num = mat.NumRows();

						keys[s] = key;
						feats[s] = mat;
						curt[s] = 0;
						lent[s] = feats[s].NumRows();
						num_frames += lent[s];

						// skip frames
						frame_num_utt[s] = example->num_ali.size();
						utt_nnet_out_h[s].Resize(frame_num_utt[s], out_dim, kSetZero);
					    if (this->kld_scale > 0) utt_si_nnet_out_h[s].Resize(frame_num_utt[s], out_dim, kSetZero);
					    if (this->kld_scale > 0 || frame_smooth > 0) utt_soft_nnet_out_h[s].Resize(frame_num_utt[s], out_dim, kSetZero);
						diff_utt_feats[s].Resize(frame_num_utt[s], out_dim, kSetZero);
						utt_copied[s] = false;
						utt_curt[s] = 0;
						diff_curt[s] = 0;
						utt_examples[s] = example;

						s++;
						cur_frames = max_frame_num * s;

						example = dynamic_cast<SequentialNnetExample*>(repository_->ProvideExample());
					}

					cur_stream_num = s;
					new_utt_flags.resize(cur_stream_num, 1);

					// we are done if all streams are exhausted
					if (cur_stream_num == 0) break;

					feat.Resize(max_frame_num * cur_stream_num, feat_dim, kSetZero);
					nnet_out_host.Resize((max_frame_num+num_skip-1)/num_skip * cur_stream_num, out_dim, kUndefined);

					//int truc_frame_num = batch_size>0 ? (max_frame_num+batch_size-1)/batch_size * batch_size : max_frame_num;
					diff_feat.Resize((max_frame_num+num_skip-1)/num_skip * cur_stream_num, out_dim, kSetZero);

					// fill a multi-stream bptt batch
					 // * feat: first shifted to achieve targets delay; then padded to batch_size
					 for (int s = 0; s < cur_stream_num; s++) {
						 for (int t = 0; t < lent[s]; t++) {
							// feat shifting & padding
							if (curt[s] + time_shift < lent[s]) {
								feat.Row(t * cur_stream_num + s).CopyFromVec(feats[s].Row(curt[s]+time_shift));
							} else {
								//int last = (frame_num_utt[s]-1)*skip_frames; //lent[s]-1
								feat.Row(t * cur_stream_num + s).CopyFromVec(feats[s].Row(lent[s]-1));
							}
							curt[s]++;
						}
					}

					 // apply optional feature transform
					nnet_transf.Feedforward(CuMatrix<BaseFloat>(feat), &feats_transf);

					// for streams with new utterance, history states need to be reset
					nnet.ResetLstmStreams(new_utt_flags, batch_size);

					// forward pass
					nnet.Propagate(feats_transf, &nnet_out);

					if (this->kld_scale > 0)
					{
                        // for streams with new utterance, history states need to be reset
                        si_nnet.ResetLstmStreams(new_utt_flags, batch_size);
						si_nnet.Propagate(feats_transf, &si_nnet_out);
						si_nnet_out_host.Resize((max_frame_num+num_skip-1)/num_skip * cur_stream_num, out_dim, kUndefined);
						si_nnet_out.CopyToMat(&si_nnet_out_host);
					}

					if (this->kld_scale > 0 || frame_smooth > 0)
					{
						softmax.Propagate(nnet_out, &soft_nnet_out);
						soft_nnet_out_host.Resize((max_frame_num+num_skip-1)/num_skip * cur_stream_num, out_dim, kUndefined);
						soft_nnet_out.CopyToMat(&soft_nnet_out_host);
					}
					/*
						      if (!KALDI_ISFINITE(nnet_out.Sum())) { // check there's no nan/inf,
        						KALDI_ERR << "NaN or inf found in final output nnet-output for " << utt_examples[0]->utt ;
      							}   
					*/

					// subtract the log_prior
                    if(prior_opts->class_frame_counts != "")
                    {
                    	log_prior.SubtractOnLogpost(&nnet_out);
                    }


					nnet_out.CopyToMat(&nnet_out_host);

					for (int s = 0; s < cur_stream_num; s++) {
						for (int t = 0; t < lent[s]; t++) {
							   // feat shifting & padding
							   for (int k = 0; k < skip_frames; k++) {
								   if (utt_curt[s] < frame_num_utt[s]) {
									   utt_nnet_out_h[s].Row(utt_curt[s]).CopyFromVec(nnet_out_host.Row(t * cur_stream_num + s));
									   if (this->kld_scale > 0)
										   utt_si_nnet_out_h[s].Row(utt_curt[s]).CopyFromVec(si_nnet_out_host.Row(t * cur_stream_num + s));
									   if (this->kld_scale > 0 || frame_smooth > 0)
										   utt_soft_nnet_out_h[s].Row(utt_curt[s]).CopyFromVec(soft_nnet_out_host.Row(t * cur_stream_num + s));
									   utt_curt[s]++;
								   }
								}
						   }
					}

					for (int s = 0; s < cur_stream_num; s++)
					{
						//SubMatrix<BaseFloat> obj_feats(diff_feat.RowRange(s*max_frame_num, lent[s]));
						/*
						      if (!KALDI_ISFINITE(utt_feats[s].Sum())) { // check there's no nan/inf,
        						KALDI_ERR << "NaN or inf found in final output nnet-host-output for " << utt_examples[s]->utt << " s: " << s;
      							}*/   

						if (this->criterion == "mmi")
							MMIObj(utt_nnet_out_h[s], diff_utt_feats[s],
					      				trans_model, utt_examples[s],
					      				total_mmi_obj, total_post_on_ali, num_frm_drop, num_done,
										utt_soft_nnet_out_h[s], utt_si_nnet_out_h[s]);
						else
							MPEObj(utt_nnet_out_h[s], diff_utt_feats[s],
				      				trans_model, utt_examples[s],
									silence_phones, total_frame_acc, num_done,
									utt_soft_nnet_out_h[s], utt_si_nnet_out_h[s]);

						num_done++;

						delete utt_examples[s];
					}

					for (int s = 0; s < cur_stream_num; s++) {
						for (int t = 0; t < lent[s]; t++) {
							// feat shifting & padding
							for (int k = 0; k < skip_frames; k++) {
								if (diff_curt[s] + time_shift < frame_num_utt[s]) {
									diff_feat.Row(t * cur_stream_num + s).AddVec(1.0, diff_utt_feats[s].Row(diff_curt[s]+time_shift));
								    diff_curt[s]++;
									//diff_feat.Row(t * cur_stream_num + s).CopyFromVec(diff_utt_feats[s].Row(diff_curt[s]+time_shift));
								} else {
									//diff_feat.Row(t * cur_stream_num + s).SetZero();
								}
							}
						}
					}



				}


				if (num_stream < 1 && ((example = dynamic_cast<SequentialNnetExample*>(repository_->ProvideExample())) != NULL))
				{
						//time.Reset();
						std::string utt = example->utt;
						Matrix<BaseFloat> &mat = example->input_frames;
						//t1 = time.Elapsed();
						//time.Reset();

						  // get actual dims for this utt and nnet
						  num_frames = mat.NumRows();

						  // 3) propagate the feature to get the log-posteriors (nnet w/o sofrmax)
						  //lstm  time-shift, copy the last frame of LSTM input N-times,
						  if (time_shift > 0) {
							int32 last_row = mat.NumRows() - 1; // last row,
							mat.Resize(mat.NumRows() + time_shift, mat.NumCols(), kCopyData);
							for (int32 r = last_row+1; r<mat.NumRows(); r++) {
							  mat.CopyRowFromVec(mat.Row(last_row), r); // copy last row,
							}
						  }
						  // push features to GPU
						  cu_feats = mat;
						  // possibly apply transform
						  nnet_transf.Feedforward(cu_feats, &feats_transf);
						  // propagate through the nnet (assuming w/o softmax)
						  nnet.Propagate(feats_transf, &nnet_out);

						  // time-shift, remove N first frames of LSTM output,
						  if (time_shift > 0) {
							  CuMatrix<BaseFloat> tmp(nnet_out);
							  nnet_out = tmp.RowRange(time_shift, tmp.NumRows() - time_shift);
						  }

						  if (this->kld_scale > 0)
						  {
							  si_nnet.Propagate(feats_transf, &si_nnet_out);

							  // time-shift, remove N first frames of LSTM output,
							  if (time_shift > 0) {
								  CuMatrix<BaseFloat> tmp(si_nnet_out);
								  si_nnet_out = tmp.RowRange(time_shift, tmp.NumRows() - time_shift);
							  }
							  si_nnet_out_h.Resize(si_nnet_out.NumRows(), si_nnet_out.NumCols(), kUndefined);
							  si_nnet_out.CopyToMat(&si_nnet_out_h);
						  }

						  if (this->kld_scale > 0 || frame_smooth > 0)
						  {
							  softmax.Propagate(nnet_out, &soft_nnet_out);
							  soft_nnet_out_h.Resize(soft_nnet_out.NumRows(), soft_nnet_out.NumCols(), kUndefined);
							  soft_nnet_out.CopyToMat(&soft_nnet_out_h);
						  }

						  // subtract the log_prior
						  if(prior_opts->class_frame_counts != "")
						  {
							  log_prior.SubtractOnLogpost(&nnet_out);
						  }

						  nnet_out_h.Resize(nnet_out.NumRows(), nnet_out.NumCols(), kUndefined);
						  nnet_out.CopyToMat(&nnet_out_h);

						  nnet_diff_h.Resize(nnet_out_h.NumRows(), nnet.OutputDim(), kSetZero);

						  if (this->criterion == "mmi")
							  MMIObj(nnet_out_h, nnet_diff_h,
					      				trans_model, example,
					      				total_mmi_obj, total_post_on_ali, num_frm_drop, num_done,
										soft_nnet_out_h, si_nnet_out_h);
						  else
							  MPEObj(nnet_out_h, nnet_diff_h,
					      				trans_model, example,
										silence_phones, total_frame_acc, num_done,
										soft_nnet_out_h, si_nnet_out_h);

					      num_done++;
					      p_nnet_diff_h = &nnet_diff_h;
					      delete example;
			}

				if (example == NULL && cur_stream_num == 0) break;

				// push to gpu
				nnet_diff = *p_nnet_diff_h;

                if (model_sync->reset_gradient_[thread_idx] && parallel_opts->merge_func == "globalgradient")
                {
                    nnet.ResetGradient();
                    model_sync->reset_gradient_[thread_idx] = false;
                }


		       if (parallel_opts->num_threads > 1 && update_frames >= opts->update_frames)
		       {
		    	   nnet.Backpropagate(nnet_diff, NULL, false);
		    	   nnet.Gradient();

		    	   //t2 = time.Elapsed();
		    	   //time.Reset();

		    	   if (parallel_opts->asgd_lock)
		    		   model_sync->LockModel();

		    	   model_sync->SetWeight(&nnet);
		    	   nnet.UpdateGradient();
		    	   model_sync->GetWeight(&nnet);

		    	   if (parallel_opts->asgd_lock)
		    		   model_sync->UnlockModel();

		    	   update_frames = 0;
			
		    	   //t3 = time.Elapsed();
		       }
		       else
		       {
		    	   nnet.Backpropagate(nnet_diff, NULL, true);

		    	   //t2 = time.Elapsed();
		    	   //time.Reset();
		       }

		       //KALDI_WARN << "prepare data: "<< t1 <<" forward & backward: "<< t2 <<" update: "<< t3;
		       // relase the buffer, we don't need anymore

		       //multi-machine
		       if (parallel_opts->num_procs > 1)
		    	{
		    		model_sync->LockModel();

		    		if (p_merge_func->CurrentMergeCache() + num_frames > parallel_opts->merge_size)
		    		{
		    			if (p_merge_func->leftMerge() <= 1 && !p_merge_func->isLastMerge())
		    			{
		    				p_merge_func->MergeStatus(1);
		    			}

		    			if (p_merge_func->leftMerge() > 1 || !p_merge_func->isLastMerge())
		    			{
		    				model_sync->GetWeight(&nnet);

		    			    p_merge_func->Merge(0);
		    			    KALDI_VLOG(1) << "Model merge NO." << parallel_opts->num_merge - p_merge_func->leftMerge()
		    			    				<< " Current mergesize = " << p_merge_func->CurrentMergeCache() << " frames.";
		    			    p_merge_func->MergeCacheReset();

		    			    model_sync->SetWeight(&nnet);
                            model_sync->ResetGradient();
		    			}
		    		}

		    		p_merge_func->AddMergeCache((int) num_frames);

		    		model_sync->UnlockModel();

		    	}

		       // increase time counter
		       total_frames += num_frames;
		       update_frames += num_frames;

		       if (num_done % 100 == 0)
		       {
		         time_now = time.Elapsed();
		         KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
		                       << time_now/60 << " min; processed " << total_frames/time_now
		                       << " frames per second.";

		       }
			
		        // track training process
			    if (this->thread_id_ == 0 && parallel_opts->myid == 0 && opts->dump_time > 0)
				{
                    int num_procs = parallel_opts->num_procs > 1 ? parallel_opts->num_procs : 1;
					if ((total_frames*parallel_opts->num_threads*num_procs)/(3600*100*opts->dump_time) > num_dump)
					{
						char name[512];
						num_dump++;
						sprintf(name, "%s_%d_%ld", model_filename.c_str(), num_dump, total_frames);
                        nnet.AppendComponent(new Softmax(nnet.OutputDim(),nnet.OutputDim()));
						nnet.Write(string(name), true);
                        nnet.RemoveComponent(nnet.NumComponents()-1);
					}
				}

		       fflush(stderr); 
               fsync(fileno(stderr));
	  }

		model_sync->LockStates();
		stats_->total_mmi_obj += total_mmi_obj;
		stats_->total_post_on_ali += total_post_on_ali;
		stats_->total_frame_acc += total_frame_acc;

		stats_->total_frames += total_frames;
		stats_->num_frm_drop += num_frm_drop;
		stats_->num_done += num_done;
		model_sync->UnlockStates();


		//last merge
			model_sync->LockModel();

			bool last_thread = true;
			for (int i = 0; i < parallel_opts->num_threads; i++)
			{
				if (i != thread_idx && !model_sync->isfinished_[i]){
						last_thread = false;
						break;
				}
			}

			if (parallel_opts->num_procs > 1)
			{
				if (last_thread)
				{
					if (!p_merge_func->isLastMerge())
						p_merge_func->MergeStatus(0);

					model_sync->GetWeight(&nnet);

					p_merge_func->Merge(0);
						KALDI_VLOG(1) << "Model merge NO." << parallel_opts->num_merge-p_merge_func->leftMerge()
									   << " Current mergesize = " << p_merge_func->CurrentMergeCache() << " frames.";
						model_sync->SetWeight(&nnet);
				}
			}

			if (last_thread)
			{
				KALDI_VLOG(1) << "Last thread upload model to host.";
					model_sync->CopyToHost(&nnet);
			}

			model_sync->isfinished_[thread_idx] = true;
			model_sync->UnlockModel();
	}

};


void NnetSequentialUpdateParallel(const NnetSequentialUpdateOptions *opts,
		std::string feature_transform,
		std::string	model_filename,
		std::string transition_model_filename,
		std::string feature_rspecifier,
		std::string den_lat_rspecifier,
		std::string num_ali_rspecifier,
		std::string sweep_frames_rspecifier,
		Nnet *nnet,
		NnetSequentialStats *stats)
{
		ExamplesRepository repository;
		NnetModelSync model_sync(nnet, opts->parallel_opts);

		SeqTrainParallelClass c(opts, &model_sync,
								feature_transform, model_filename, transition_model_filename, den_lat_rspecifier, num_ali_rspecifier,
								&repository, nnet, stats);


	  {

	    	SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
	    	RandomAccessLatticeReader den_lat_reader(den_lat_rspecifier);
	    	RandomAccessInt32VectorReader num_ali_reader(num_ali_rspecifier);
	    	RandomAccessInt32VectorReader sweep_frames_reader(sweep_frames_rspecifier);

	    // The initialization of the following class spawns the threads that
	    // process the examples.  They get re-joined in its destructor.
	    MultiThreader<SeqTrainParallelClass> m(opts->parallel_opts->num_threads, c);

	    // prepare sample
		NnetExample *example;
		std::vector<NnetExample*> examples;
		std::vector<int> sweep_frames, loop_frames;
		if (!kaldi::SplitStringToIntegers(opts->sweep_frames_str, ":", false, &sweep_frames))
			KALDI_ERR << "Invalid sweep-frames string " << opts->sweep_frames_str;
		for (int i = 0; i < sweep_frames.size(); i++) {
			if (sweep_frames[i] >= opts->skip_frames)
				KALDI_ERR << "invalid sweep frames indexes";
		}

		int nframes = sweep_frames.size();
		int idx = 0;
		loop_frames = sweep_frames;
		// loop sweep skip frames
		for (; !feature_reader.Done(); feature_reader.Next()) {
			if (!opts->sweep_loop) {
				loop_frames.resize(1);
				loop_frames[0] = sweep_frames[idx];
				idx = (idx+1)%nframes;
			}

			example = new SequentialNnetExample(&feature_reader, &den_lat_reader,
					&num_ali_reader, &sweep_frames_reader, &model_sync, stats, opts);
			example->SetSweepFrames(loop_frames, opts->skip_inner);
			if (example->PrepareData(examples)) {
				for (int i = 0; i < examples.size(); i++) {
					repository.AcceptExample(examples[i]);
				}
				if (examples[0] != example)
					delete example;
			}
			else
				delete example;
		}
		repository.ExamplesDone();

	  }

}


} // namespace nnet
} // namespace kaldi


