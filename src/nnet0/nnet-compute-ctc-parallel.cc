// nnet0/nnet-compute-parallel.cc

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

#include "cudamatrix/cu-device.h"
#include "base/kaldi-types.h"


#include "nnet0/nnet-affine-transform.h"
#include "nnet0/nnet-affine-preconditioned-transform.h"
#include "nnet0/nnet-model-merge-function.h"
#include "nnet0/nnet-activation.h"
#include "nnet0/nnet-example.h"

#include "nnet0/nnet-compute-ctc-parallel.h"

namespace kaldi {
namespace nnet0 {

class TrainCtcParallelClass: public MultiThreadable {

private:
    const NnetCtcUpdateOptions *opts;
    NnetModelSync *model_sync;

	std::string feature_transform,
				model_filename,
				si_model_filename,
				targets_rspecifier;

	ExamplesRepository *repository_;
    NnetStats *stats_;

    const NnetTrainOptions *trn_opts;
    const NnetDataRandomizerOptions *rnd_opts;
    const NnetParallelOptions *parallel_opts;

    BaseFloat 	kld_scale;

    std::string use_gpu;
    std::string objective_function;
    int32 num_threads;
    bool crossvalidate;



 public:
  // This constructor is only called for a temporary object
  // that we pass to the RunMultiThreaded function.
    TrainCtcParallelClass(const NnetCtcUpdateOptions *opts,
			NnetModelSync *model_sync,
			std::string	model_filename,
			std::string targets_rspecifier,
			ExamplesRepository *repository,
			Nnet *nnet,
			NnetStats *stats):
				opts(opts),
				model_sync(model_sync),
				model_filename(model_filename),
				targets_rspecifier(targets_rspecifier),
				repository_(repository),
				stats_(stats)
 	 		{
				trn_opts = opts->trn_opts;
				rnd_opts = opts->rnd_opts;
				parallel_opts = opts->parallel_opts;

				kld_scale = opts->kld_scale;
				objective_function = opts->objective_function;
				use_gpu = opts->use_gpu;
				feature_transform = opts->feature_transform;
				si_model_filename = opts->si_model_filename;

				num_threads = parallel_opts->num_threads;
				crossvalidate = opts->crossvalidate;
 	 		}

	void monitor(Nnet *nnet, kaldi::int64 total_frames, int32 num_frames)
	{
        // 1st minibatch : show what happens in network
        if (kaldi::g_kaldi_verbose_level >= 1 && total_frames == 0) { // vlog-1
          KALDI_VLOG(1) << "### After " << total_frames << " frames,";
          KALDI_VLOG(1) << nnet->InfoPropagate();
          if (!crossvalidate) {
            KALDI_VLOG(1) << nnet->InfoBackPropagate();
            KALDI_VLOG(1) << nnet->InfoGradient();
          }
        }

        // monitor the NN training
        if (kaldi::g_kaldi_verbose_level >= 2) { // vlog-2
          if ((total_frames/25000) != ((total_frames+num_frames)/25000)) { // print every 25k frames
            KALDI_VLOG(2) << "### After " << total_frames << " frames,";
            KALDI_VLOG(2) << nnet->InfoPropagate();
            if (!crossvalidate) {
              KALDI_VLOG(2) << nnet->InfoGradient();
            }
          }
        }
	}

	  // This does the main function of the class.
	void operator ()()
	{
		int thread_idx = this->thread_id_;

		model_sync->LockModel();

	    // Select the GPU
	#if HAVE_CUDA == 1
	    if (parallel_opts->num_procs > 1)
	    {
	    	//thread_idx = model_sync->GetThreadIdx();
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

		Nnet nnet_transf;
	    if (feature_transform != "") {
	      nnet_transf.Read(feature_transform);
	    }

	    Nnet nnet;
	    nnet.Read(model_filename);

	    nnet.SetTrainOptions(*trn_opts);

	    int32 rank_in = 20, rank_out = 80, update_period = 4;
	   	    BaseFloat num_samples_history = 2000.0;
	   	    BaseFloat alpha = 4.0;
	    //if (opts->num_procs > 1 || opts->use_psgd)
	    if (opts->use_psgd)
	    	nnet.SwitchToOnlinePreconditioning(rank_in, rank_out, update_period, num_samples_history, alpha);


	    if (opts->dropout_retention > 0.0) {
	      nnet_transf.SetDropoutRetention(opts->dropout_retention);
	      nnet.SetDropoutRetention(opts->dropout_retention);
	    }
	    if (crossvalidate) {
	      nnet_transf.SetDropoutRetention(1.0);
	      nnet.SetDropoutRetention(1.0);
	    }

	    Nnet si_nnet;
	    if (this->kld_scale > 0)
	    {
	    	si_nnet.Read(si_model_filename);
	    }

	    model_sync->Initialize(&nnet);

	    RandomizerMask randomizer_mask(*rnd_opts);
	    MatrixRandomizer feature_randomizer(*rnd_opts);
	    PosteriorRandomizer targets_randomizer(*rnd_opts);
	    VectorRandomizer weights_randomizer(*rnd_opts);

	    Xent xent;
	    Mse mse;
	    // Initialize CTC optimizer
	    Ctc ctc;

		CuMatrix<BaseFloat> feats_transf, nnet_out, nnet_diff;
		//CuMatrix<BaseFloat> si_nnet_out, soft_nnet_out, *p_si_nnet_out=NULL, *p_soft_nnet_out;
		Matrix<BaseFloat> nnet_out_h, nnet_diff_h;

		ModelMergeFunction *p_merge_func = model_sync->GetModelMergeFunction();

		//double t1, t2, t3, t4;
		int32 update_frames = 0, num_frames = 0, num_done = 0, num_dump = 0;
		kaldi::int64 total_frames = 0;

		int32 num_stream = opts->num_stream;
		int32 frame_limit = opts->max_frames;
		int32 targets_delay = opts->targets_delay;
		int32 batch_size = opts->batch_size;
        int32 skip_frames = opts->skip_frames;

	    std::vector< Matrix<BaseFloat> > feats_utt(num_stream);  // Feature matrix of every utterance
	    std::vector< std::vector<int> > labels_utt(num_stream);  // Label vector of every utterance
	    std::vector<int> num_utt_frame_in, num_utt_frame_out;
        std::vector<int> new_utt_flags;

	    Matrix<BaseFloat> feat_mat_host;
	    Vector<BaseFloat> frame_mask_host;
	    Posterior target;
	    std::vector<Posterior> targets_utt(num_stream);

	    CTCNnetExample *ctc_example = NULL;
	    DNNNnetExample *dnn_example = NULL;
	    NnetExample		*example = NULL;
	    Timer time;
	    double time_now = 0;

		int32 cur_stream_num = 0, num_skip, in_rows, out_rows;
		int32 feat_dim = nnet.InputDim();
	    num_skip = opts->skip_inner ? skip_frames : 1;
        frame_limit *= num_skip;

	    while (num_stream) {


			int32 s = 0, max_frame_num = 0, cur_frames = 0;
			cur_stream_num = 0; num_frames = 0;
			num_utt_frame_in.clear();
			num_utt_frame_out.clear();

			if (NULL == example)
				example = repository_->ProvideExample();

			if (NULL == example)
				break;

			while (s < num_stream && cur_frames < frame_limit && NULL != example)
			{
				std::string key = example->utt;
				Matrix<BaseFloat> &mat = example->input_frames;

				if (objective_function == "xent"){
					dnn_example = dynamic_cast<DNNNnetExample*>(example);
					targets_utt[s] = dnn_example->targets;
				}
				else if (objective_function == "ctc"){
					ctc_example = dynamic_cast<CTCNnetExample*>(example);
					labels_utt[s] = ctc_example->targets;
				}

				if ((s+1)*mat.NumRows() > frame_limit || (s+1)*max_frame_num > frame_limit) break;
				if (max_frame_num < mat.NumRows()) max_frame_num = mat.NumRows();

				// forward the features through a feature-transform,
				nnet_transf.Feedforward(CuMatrix<BaseFloat>(mat), &feats_transf);

				feats_utt[s].Resize(feats_transf.NumRows(), feats_transf.NumCols());
				feats_transf.CopyToMat(&feats_utt[s]);
		        	//feats_utt[s] = mat;
				in_rows = mat.NumRows();
				num_utt_frame_in.push_back(in_rows);
		        	num_frames += in_rows;

		        // inner skip frames
		        out_rows = in_rows/num_skip;
		        out_rows += in_rows%num_skip > 0 ? 1:0;
		        num_utt_frame_out.push_back(out_rows);

				s++;
				num_done++;
				cur_frames = max_frame_num * s;

				delete example;
				example = repository_->ProvideExample();
			}

			targets_delay *=  num_skip;
			cur_stream_num = s;
			new_utt_flags.resize(cur_stream_num, 1);

			// Create the final feature matrix. Every utterance is padded to the max length within this group of utterances
			feat_mat_host.Resize(cur_stream_num * max_frame_num, feat_dim, kSetZero);
			if (this->objective_function == "xent")
			{
				target.resize(cur_stream_num * (max_frame_num+num_skip-1)/num_skip);
				frame_mask_host.Resize(cur_stream_num * (max_frame_num+num_skip-1)/num_skip, kSetZero);
			}

			for (int s = 0; s < cur_stream_num; s++) {
			  //Matrix<BaseFloat> mat_tmp = feats_utt[s];
			  for (int r = 0; r < num_utt_frame_in[s]; r++) {
				  //feat_mat_host.Row(r*cur_stream_num + s).CopyFromVec(mat_tmp.Row(r));
				  if (r + targets_delay < num_utt_frame_in[s]) {
					  feat_mat_host.Row(r*cur_stream_num + s).CopyFromVec(feats_utt[s].Row(r+targets_delay));
				  }
				  else{
					  int last = (num_utt_frame_in[s]-1); // frame_num_utt[s]-1
					  feat_mat_host.Row(r*cur_stream_num + s).CopyFromVec(feats_utt[s].Row(last));
				  }
				  //ce label
				  if (this->objective_function == "xent" && r%num_skip == 0)
				  {
					  target[r*cur_stream_num/num_skip + s] = targets_utt[s][r];
					  frame_mask_host(r*cur_stream_num/num_skip + s) = 1;
				  }
			  }
			}
			      // Set the original lengths of utterances before padding
			//nnet.SetSeqLengths(frame_num_utt);
			//nnet.ResetLstmStreams(frame_num_utt);
			//for lstm
			nnet.ResetLstmStreams(new_utt_flags, batch_size);
			//for bilstm
			nnet.SetSeqLengths(num_utt_frame_in);

	        // report the speed
	        if (num_done % 5000 == 0) {
	          time_now = time.Elapsed();
	          KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
	                        << time_now/60 << " min; processed " << total_frames/time_now
	                        << " frames per second.";
	        }

	        // Propagation and CTC training
	        nnet.Propagate(CuMatrix<BaseFloat>(feat_mat_host), &nnet_out);

	        if (objective_function == "xent"){
	        	xent.Eval(frame_mask_host, nnet_out, target, &nnet_diff);
	        }
	        else if (objective_function == "ctc"){
	        	//ctc error
	        	ctc.EvalParallel(num_utt_frame_out, nnet_out, labels_utt, &nnet_diff);
	        	// Error rates
	        	ctc.ErrorRateMSeq(num_utt_frame_out, nnet_out, labels_utt);

                /*
                if (!KALDI_ISFINITE(nnet_out.Sum())) { // check there's no nan/inf,
                    KALDI_ERR << "NaN or inf found in transformed-features for " << example->utt << "\n" << 
                    labels_utt[0] << "\n" << labels_utt[1];
                }   
                */
	        }
	        else
	        	KALDI_ERR<< "Unknown objective function code : " << objective_function;


		        // backward pass
				if (!crossvalidate) {
					// backpropagate

                    if (model_sync->reset_gradient_[thread_idx] && parallel_opts->merge_func == "globalgradient")
                    {
                        nnet.ResetGradient();
                        model_sync->reset_gradient_[thread_idx] = false;
                        //KALDI_VLOG(1) << "Reset Gradient";
                    }

					if (parallel_opts->num_threads > 1 && update_frames >= opts->update_frames) {
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
					} else {
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
				}

				monitor(&nnet, total_frames, num_frames);

				// increase time counter
		        update_frames += num_frames;
		        total_frames += num_frames;

		        // track training process
			    if (!crossvalidate && this->thread_id_ == 0 && parallel_opts->myid == 0 && opts->dump_time > 0)
				{
                    int num_procs = parallel_opts->num_procs > 1 ? parallel_opts->num_procs : 1;
					if ((total_frames*parallel_opts->num_threads*num_procs)/(3600*100*opts->dump_time) > num_dump)
					{
						char name[512];
						num_dump++;
						sprintf(name, "%s_%d_%ld", model_filename.c_str(), num_dump, total_frames);
						nnet.Write(string(name), true);
					}
				}

		        fflush(stderr);
		        fsync(fileno(stderr));
		}

		model_sync->LockStates();

		stats_->total_frames += total_frames;
		stats_->num_done += num_done;
		if (objective_function == "xent")
			stats_->xent.Add(&xent);
		else if (objective_function == "ctc")
			dynamic_cast<NnetCtcStats*>(stats_)->ctc.Add(&ctc);
		else
			KALDI_ERR<< "Unknown objective function code : " << objective_function;

		model_sync->UnlockStates();

		//last merge
		if (!crossvalidate){
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
	}

};


void NnetCtcUpdateParallel(const NnetCtcUpdateOptions *opts,
		std::string	model_filename,
		std::string feature_rspecifier,
		std::string targets_rspecifier,
		Nnet *nnet,
		NnetCtcStats *stats)
{
		ExamplesRepository repository;
		NnetModelSync model_sync(nnet, opts->parallel_opts);

		TrainCtcParallelClass c(opts, &model_sync,
								model_filename, targets_rspecifier,
								&repository, nnet, stats);


	  {

		    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
		    RandomAccessInt32VectorReader targets_reader(targets_rspecifier);

	    // The initialization of the following class spawns the threads that
	    // process the examples.  They get re-joined in its destructor.
	    MultiThreader<TrainCtcParallelClass> mc(opts->parallel_opts->num_threads, c);

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

	    	example = new CTCNnetExample(&feature_reader, &targets_reader,
	    			&model_sync, stats, opts);
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

void NnetCEUpdateParallel(const NnetCtcUpdateOptions *opts,
		std::string	model_filename,
		std::string feature_rspecifier,
		std::string targets_rspecifier,
		Nnet *nnet,
		NnetStats *stats)
{
		ExamplesRepository repository;
		NnetModelSync model_sync(nnet, opts->parallel_opts);

		TrainCtcParallelClass c(opts, &model_sync,
								model_filename, targets_rspecifier,
								&repository, nnet, stats);


	  {

		    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
		    RandomAccessBaseFloatVectorReader weights_reader;
			RandomAccessPosteriorReader targets_reader(targets_rspecifier);

	    // The initialization of the following class spawns the threads that
	    // process the examples.  They get re-joined in its destructor.
	    MultiThreader<TrainCtcParallelClass> mc(opts->parallel_opts->num_threads, c);


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

	    	example = new DNNNnetExample(&feature_reader, &targets_reader,
	    			&weights_reader, &model_sync, stats, opts);
            example->SetSweepFrames(loop_frames);
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


