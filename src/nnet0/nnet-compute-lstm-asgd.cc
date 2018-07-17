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

#include "nnet0/nnet-compute-lstm-asgd.h"

namespace kaldi {
namespace nnet0 {

class TrainLstmAsgdClass: public MultiThreadable {

private:
    const NnetLstmUpdateOptions *opts;
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
    TrainLstmAsgdClass(const NnetLstmUpdateOptions *opts,
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
	    	si_nnet.Read(si_model_filename);

	    model_sync->Initialize(&nnet);

	    RandomizerMask randomizer_mask(*rnd_opts);
	    MatrixRandomizer feature_randomizer(*rnd_opts);
	    PosteriorRandomizer targets_randomizer(*rnd_opts);
	    VectorRandomizer weights_randomizer(*rnd_opts);

	    Xent xent;
	    Mse mse;
        MultiTaskLoss multitask;
        if (0 == objective_function.compare(0, 9, "multitask")) {
          // objective_function contains something like :
          // 'multitask,xent,2456,1.0,mse,440,0.001'
          //
          // the meaning is following:
          // 'multitask,<type1>,<dim1>,<weight1>,...,<typeN>,<dimN>,<weightN>'
          multitask.InitFromString(objective_function);
        }

		CuMatrix<BaseFloat> feats_transf, nnet_in, nnet_out, nnet_diff;
		CuMatrix<BaseFloat> si_nnet_out; // soft_nnet_out, *p_si_nnet_out=NULL, *p_soft_nnet_out;
		Matrix<BaseFloat> nnet_out_h, nnet_diff_h;

		ModelMergeFunction *p_merge_func = model_sync->GetModelMergeFunction();

		//double t1, t2, t3, t4;
		int32 update_frames = 0, num_frames = 0, num_done = 0, num_dump = 0, minibatch = 0;
		kaldi::int64 total_frames = 0;

		int32 num_stream = opts->num_stream;
		int32 batch_size = opts->batch_size;
		int32 targets_delay = opts->targets_delay;
		int32 skip_frames = opts->skip_frames;

	    //  book-keeping for multi-streams
	    std::vector<std::string> keys(num_stream);
	    std::vector<Matrix<BaseFloat> > feats(num_stream);
	    std::vector<Posterior> targets(num_stream);
	    std::vector<int> curt(num_stream, 0);
	    std::vector<int> lent(num_stream, 0);
	    std::vector<int> new_utt_flags(num_stream, 0);

	    // bptt batch buffer
	    //int32 feat_dim = nnet.InputDim();
	    Vector<BaseFloat> frame_mask(batch_size * num_stream, kSetZero);
	    Posterior target(batch_size * num_stream);
	    Matrix<BaseFloat> feat;

	    DNNNnetExample *example;
	    Timer time;
	    double time_now = 0;

	    while (num_stream) {
	        // loop over all streams, check if any stream reaches the end of its utterance,
	        // if any, feed the exhausted stream with a new utterance, update book-keeping infos
	        for (int s = 0; s < num_stream; s++) {
	            // this stream still has valid frames
	            if (curt[s] < lent[s] + targets_delay && curt[s] > 0) {
	                new_utt_flags[s] = 0;
	                continue;
	            }
			
	            // else, this stream exhausted, need new utterance
	            while ((example = dynamic_cast<DNNNnetExample*>(repository_->ProvideExample())) != NULL)
	            {
	                const std::string& key = example->utt;
	                // get the feature matrix,
	                const Matrix<BaseFloat> &mat = example->input_frames;
	                // forward the features through a feature-transform,
	                nnet_transf.Feedforward(CuMatrix<BaseFloat>(mat), &feats_transf);

	                // get the labels,
	                const Posterior& target = example->targets;

	                num_done++;

	                // checks ok, put the data in the buffers,
	                keys[s] = key;
	                feats[s].Resize(feats_transf.NumRows(), feats_transf.NumCols());
	                feats_transf.CopyToMat(&feats[s]);
	                //feats[s] = mat;
	                targets[s] = target;
	                curt[s] = 0;
	                lent[s] = target.size();
	                new_utt_flags[s] = 1;  // a new utterance feeded to this stream
	                delete example;
	                break;
	            }
	        }

	        // we are done if all streams are exhausted
	        int done = 1;
	        for (int s = 0; s < num_stream; s++) {
	            if (curt[s]  < lent[s] + targets_delay) done = 0;  // this stream still contains valid data, not exhausted
	        }

	        if (done) break;

	        if (feat.NumCols() != feats[0].NumCols()) {
	        	minibatch = batch_size * num_stream;
	        	if (opts->skip_inner) minibatch *= skip_frames;
	        	feat.Resize(minibatch, feats[0].NumCols(), kSetZero);
	        	nnet_in.Resize(minibatch, feats[0].NumCols(), kSetZero);
	        }

	        // fill a multi-stream bptt batch
	        // * frame_mask: 0 indicates padded frames, 1 indicates valid frames
	        // * target: padded to batch_size
	        // * feat: first shifted to achieve targets delay; then padded to batch_size
	        for (int t = 0; t < batch_size; t++) {
	            for (int s = 0; s < num_stream; s++) {
	                // frame_mask & targets padding
	                if (curt[s] < targets_delay) {
	                	frame_mask(t * num_stream + s) = 0;
	                	target[t * num_stream + s] = targets[s][0];
	                }
	                else if (curt[s] < lent[s] + targets_delay) {
	                    frame_mask(t * num_stream + s) = 1;
	                    target[t * num_stream + s] = targets[s][curt[s]-targets_delay];
	                } else {
	                    frame_mask(t * num_stream + s) = 0;
	                    target[t * num_stream + s] = targets[s][lent[s]-1];
	                }

	                // feat shifting & padding
	                int nframes = opts->skip_inner ? skip_frames : 1;
	                for (int i = 0; i < nframes; i++) {
	                	if (curt[s] < lent[s]) {
							feat.Row((t*nframes+i) * num_stream + s).CopyFromVec(feats[s].Row(curt[s]*nframes+i));
						} else {
							int last = feats[s].NumRows()-1;
							feat.Row((t*nframes+i) * num_stream + s).CopyFromVec(feats[s].Row(last));
						}
	                }
	                curt[s]++;
	            }
	        }

			num_frames = minibatch;
			// report the speed
			if (num_done % 5000 == 0) {
			  time_now = time.Elapsed();
			  KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
							<< time_now/60 << " min; processed " << total_frames/time_now
							<< " frames per second.";
			}

	        // apply optional feature transform
	        //nnet_transf.Feedforward(CuMatrix<BaseFloat>(feat), &feats_transf);
			nnet_in.CopyFromMat(feat);

	        // for streams with new utterance, history states need to be reset
	        nnet.ResetLstmStreams(new_utt_flags);

	        // forward pass
	        nnet.Propagate(nnet_in, &nnet_out);

            CuMatrix<BaseFloat> tgt_mat;
            if (this->kld_scale > 0) {
                si_nnet.ResetLstmStreams(new_utt_flags);
                si_nnet.Propagate(nnet_in, &si_nnet_out);
                // convert posterior to matrix,
                PosteriorToMatrix(target, nnet.OutputDim(), &tgt_mat);
                tgt_mat.Scale(1-this->kld_scale);
                tgt_mat.AddMat(this->kld_scale, si_nnet_out);
            }

            // evaluate objective function we've chosen
            if (objective_function == "xent") {
                // gradients re-scaled by weights in Eval,
                if (this->kld_scale > 0)
                    xent.Eval(frame_mask, nnet_out, tgt_mat, &nnet_diff);
                else
                    xent.Eval(frame_mask, nnet_out, target, &nnet_diff);
            }
            /*
            else if (objective_function == "mse") {
                // gradients re-scaled by weights in Eval,
                if (this->kld_scale > 0)
                    mse.Eval(frame_mask, nnet_out, tgt_mat, &nnet_diff);
                else
                    mse.Eval(frame_mask, nnet_out, target, &nnet_diff);
            }*/
            else if (0 == objective_function.compare(0, 9, "multitask")) {
                  // gradients re-scaled by weights in Eval,
                if (this->kld_scale > 0)
                    multitask.Eval(frame_mask, nnet_out, tgt_mat, &nnet_diff);
                else
                    multitask.Eval(frame_mask, nnet_out, target, &nnet_diff);
            } else {
                KALDI_ERR<< "Unknown objective function code : " << objective_function;
            }


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
							    KALDI_VLOG(1) << "Model " << parallel_opts->merge_func << " merge NO." 
										<< parallel_opts->num_merge - p_merge_func->leftMerge()
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

		if (objective_function == "xent"){
			//KALDI_LOG << xent.Report();
			stats_->xent.Add(&xent);
		 }else if (objective_function == "mse"){
			//KALDI_LOG << mse.Report();
			stats_->mse.Add(&mse);
		 }else if (0 == objective_function.compare(0, 9, "multitask")) {
             stats_->multitask.Add(&multitask);
         }else {
			 KALDI_ERR<< "Unknown objective function code : " << objective_function;
		 }

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


void NnetLstmUpdateAsgd(const NnetLstmUpdateOptions *opts,
		std::string	model_filename,
		std::string feature_rspecifier,
		std::string targets_rspecifier,
		Nnet *nnet,
		NnetStats *stats)
{
		ExamplesRepository repository;
		NnetModelSync model_sync(nnet, opts->parallel_opts);

		TrainLstmAsgdClass c(opts, &model_sync,
								model_filename, targets_rspecifier,
								&repository, nnet, stats);


	  {

		    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
		    RandomAccessPosteriorReader targets_reader(targets_rspecifier);
		    RandomAccessBaseFloatVectorReader weights_reader;
		    if (opts->frame_weights != "") {
		      weights_reader.Open(opts->frame_weights);
		    }

            if (opts->objective_function.compare(0, 9, "multitask") == 0)
                stats->multitask.InitFromString(opts->objective_function);

	    // The initialization of the following class spawns the threads that
	    // process the examples.  They get re-joined in its destructor.
	    MultiThreader<TrainLstmAsgdClass> mc(opts->parallel_opts->num_threads, c);

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


