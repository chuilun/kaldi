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
#include "nnet0/nnet-utils.h"

#include "nnet0/nnet-compute-parallel.h"

namespace kaldi {
namespace nnet0 {

class TrainParallelClass: public MultiThreadable {

private:
    const NnetUpdateOptions *opts;
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
	TrainParallelClass(const NnetUpdateOptions *opts,
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
	    MultiTaskLoss multitask;
	    if (0 == objective_function.compare(0, 9, "multitask")) {
		  // objective_function contains something like :
		  // 'multitask,xent,2456,1.0,mse,440,0.001'
		  //
		  // the meaning is following:
		  // 'multitask,<type1>,<dim1>,<weight1>,...,<typeN>,<dimN>,<weightN>'
		  multitask.InitFromString(objective_function);
          stats_->multitask.InitFromString(objective_function);
		}

	    Timer time, mpi_time;

		CuMatrix<BaseFloat> feats, feats_transf, nnet_out, nnet_diff;
		CuMatrix<BaseFloat> si_nnet_out; // *p_si_nnet_out = NULL;
		Matrix<BaseFloat> nnet_out_h, nnet_diff_h;

		DNNNnetExample *example;

		ModelMergeFunction *p_merge_func = model_sync->GetModelMergeFunction();

		//double t1, t2, t3, t4;
		int32 update_frames = 0, num_frames = 0, num_done = 0, num_dump = 0;
		kaldi::int64 total_frames = 0;

		while (1)
		{
			while (!feature_randomizer.IsFull() && (example = dynamic_cast<DNNNnetExample*>(repository_->ProvideExample())) != NULL)
			{
				//time.Reset();
				std::string utt = example->utt;
				const Matrix<BaseFloat> &mat = example->input_frames;
				Posterior &targets = example->targets;
				Vector<BaseFloat> &weights = example->frames_weights;
				//t1 = time.Elapsed();
				//time.Reset();

		        // apply optional feature transform
		        nnet_transf.Feedforward(CuMatrix<BaseFloat>(mat), &feats_transf);

		        // pass data to randomizers
		        KALDI_ASSERT(feats_transf.NumRows() == targets.size());
		        feature_randomizer.AddData(feats_transf);
		        targets_randomizer.AddData(targets);
		        weights_randomizer.AddData(weights);
		        num_done++;

		        // report the speed
		        if (num_done % 5000 == 0) {
		          double time_now = time.Elapsed();
		          KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
		                        << time_now/60 << " min; processed " << total_frames/time_now
		                        << " frames per second.";
		        }

		        // release the buffers we don't need anymore
		       	delete example;
			}

	        if (feature_randomizer.Done())
	        		break;

		      // randomize
      		if (!crossvalidate && opts->randomize) {
        		const std::vector<int32>& mask = randomizer_mask.Generate(feature_randomizer.NumFrames());
        		feature_randomizer.Randomize(mask);
        		targets_randomizer.Randomize(mask);
       			weights_randomizer.Randomize(mask);
      		}

	        // train with data from randomizers (using mini-batches)
			for (; !feature_randomizer.Done();
					feature_randomizer.Next(), targets_randomizer.Next(), weights_randomizer.Next())
			{
				// get block of feature/target pairs
				const CuMatrixBase<BaseFloat>& nnet_in = feature_randomizer.Value();
				const Posterior& nnet_tgt = targets_randomizer.Value();
				const Vector<BaseFloat>& frm_weights = weights_randomizer.Value();
				num_frames = nnet_in.NumRows();

				// forward pass
				nnet.Propagate(nnet_in, &nnet_out);

				CuMatrix<BaseFloat> tgt_mat;
			    if (this->kld_scale > 0)
			    {
			      	si_nnet.Propagate(nnet_in, &si_nnet_out);
			      	//p_si_nnet_out = &si_nnet_out;
						  // convert posterior to matrix,
					PosteriorToMatrix(nnet_tgt, nnet.OutputDim(), &tgt_mat);
					tgt_mat.Scale(1-this->kld_scale);
					tgt_mat.AddMat(this->kld_scale, si_nnet_out);
			    }


				// evaluate objective function we've chosen
				if (objective_function == "xent") {
					// gradients re-scaled by weights in Eval,
					if (this->kld_scale > 0)
						xent.Eval(frm_weights, nnet_out, tgt_mat, &nnet_diff);
					else
						xent.Eval(frm_weights, nnet_out, nnet_tgt, &nnet_diff);
				} else if (objective_function == "mse") {
					// gradients re-scaled by weights in Eval,
					if (this->kld_scale > 0)
						mse.Eval(frm_weights, nnet_out, tgt_mat, &nnet_diff);
					else
						mse.Eval(frm_weights, nnet_out, nnet_tgt, &nnet_diff);
				} else if (0 == objective_function.compare(0, 9, "multitask")) {
			          // gradients re-scaled by weights in Eval,
					if (this->kld_scale > 0)
						multitask.Eval(frm_weights, nnet_out, tgt_mat, &nnet_diff);
					else
						multitask.Eval(frm_weights, nnet_out, nnet_tgt, &nnet_diff);
			    } else {
					KALDI_ERR<< "Unknown objective function code : " << objective_function;
				}

		        // backward pass
				if (!crossvalidate) {
					// backpropagate

                    /*
                    if (model_sync->reset_gradient_[thread_idx] && parallel_opts->merge_func == "globalgradient")
                    {
                        nnet.ResetGradient();
                        model_sync->reset_gradient_[thread_idx] = false;
                        //KALDI_VLOG(1) << "Reset Gradient";
                    }
                    */

					if (parallel_opts->num_threads > 1 && update_frames >= opts->update_frames) {
					//if (update_frames >= opts->update_frames) {
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
                                //model_sync->ResetGradient();
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


void NnetUpdateParallel(const NnetUpdateOptions *opts,
		std::string	model_filename,
		std::string feature_rspecifier,
		std::string targets_rspecifier,
		Nnet *nnet,
		NnetStats *stats)
{
		ExamplesRepository repository;
		NnetModelSync model_sync(nnet, opts->parallel_opts);

		TrainParallelClass c(opts, &model_sync,
								model_filename, targets_rspecifier,
								&repository, nnet, stats);


	  {

		    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
		    RandomAccessPosteriorReader targets_reader(targets_rspecifier);
		    RandomAccessBaseFloatVectorReader weights_reader;
		    if (opts->frame_weights != "") 
		        weights_reader.Open(opts->frame_weights);

            if (opts->objective_function.compare(0, 9, "multitask") == 0)
                stats->multitask.InitFromString(opts->objective_function);

	    // The initialization of the following class spawns the threads that
	    // process the examples.  They get re-joined in its destructor.
	    MultiThreader<TrainParallelClass> m(opts->parallel_opts->num_threads, c);

	    NnetExample *example;
	    std::vector<NnetExample*> examples;
	    for (; !feature_reader.Done(); feature_reader.Next()) {
	    	example = new DNNNnetExample(&feature_reader, &targets_reader, &weights_reader, &model_sync, stats, opts);
	    	if (example->PrepareData(examples))
	    	{
	    		for (int i = 0; i < examples.size(); i++)
	    			repository.AcceptExample(examples[i]);
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


