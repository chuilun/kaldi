// nnet0/nnet-compute-lstm-lm-parallel.cc

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
#include <algorithm>
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
#include "nnet0/nnet-affine-transform.h"
#include "nnet0/nnet-class-affine-transform.h"
#include "nnet0/nnet-word-vector-transform.h"

#include "nnet0/nnet-compute-lstm-lm-parallel.h"

namespace kaldi {
namespace nnet0 {

class TrainLstmLmParallelClass: public MultiThreadable {

private:
    const NnetLstmLmUpdateOptions *opts;
    NnetModelSync *model_sync;

	std::string feature_transform,
				model_filename,
				classboundary_file,
				si_model_filename;

	ExamplesRepository *repository_;
    NnetLmStats *stats_;

    const NnetTrainOptions *trn_opts;
    const NnetDataRandomizerOptions *rnd_opts;
    const NnetParallelOptions *parallel_opts;

    BaseFloat 	kld_scale;

    std::string use_gpu;
    std::string objective_function;
    int32 num_threads;
    bool crossvalidate;

    std::vector<int32> class_boundary_, word2class_;

 public:
  // This constructor is only called for a temporary object
  // that we pass to the RunMultiThreaded function.
    TrainLstmLmParallelClass(const NnetLstmLmUpdateOptions *opts,
			NnetModelSync *model_sync,
			std::string	model_filename,
			ExamplesRepository *repository,
			Nnet *nnet,
			NnetLmStats *stats):
				opts(opts),
				model_sync(model_sync),
				model_filename(model_filename),
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
				classboundary_file = opts->class_boundary;
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

		NnetLmUtil util;
	    ClassAffineTransform *class_affine = NULL;
	    WordVectorTransform *word_transf = NULL;
	    CBSoftmax *cb_softmax = NULL;
	    for (int32 c = 0; c < nnet.NumComponents(); c++)
	    {
	    	if (nnet.GetComponent(c).GetType() == Component::kClassAffineTransform)
	    		class_affine = &(dynamic_cast<ClassAffineTransform&>(nnet.GetComponent(c)));
	    	else if (nnet.GetComponent(c).GetType() == Component::kWordVectorTransform)
	    		word_transf = &(dynamic_cast<WordVectorTransform&>(nnet.GetComponent(c)));
	    	else if (nnet.GetComponent(c).GetType() == Component::kCBSoftmax)
	    		cb_softmax = &(dynamic_cast<CBSoftmax&>(nnet.GetComponent(c)));
	    }

	    if (classboundary_file != "")
	    {
		    Input in;
		    Vector<BaseFloat> classinfo;
		    in.OpenTextMode(classboundary_file);
		    classinfo.Read(in.Stream(), false);
		    in.Close();
		    util.SetClassBoundary(classinfo, class_boundary_, word2class_);
	    }

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

	    CBXent cbxent;
        Xent xent;
	    Mse mse;

        if (NULL != class_affine)
        {
	        class_affine->SetClassBoundary(class_boundary_);
	        cb_softmax->SetClassBoundary(class_boundary_);
	        cbxent.SetClassBoundary(class_boundary_);
	        cbxent.SetVarPenalty(opts->var_penalty);
	        cbxent.SetZt(cb_softmax->GetZt(), cb_softmax->GetZtPatches());
        }

		CuMatrix<BaseFloat> feats_transf, nnet_out, nnet_diff;
		Matrix<BaseFloat> nnet_out_h, nnet_diff_h;

		ModelMergeFunction *p_merge_func = model_sync->GetModelMergeFunction();

		//double t1, t2, t3, t4;
		int32 update_frames = 0, num_frames = 0, num_done = 0;
		kaldi::int64 total_frames = 0;

		int32 num_stream = opts->num_stream;
		int32 batch_size = opts->batch_size;
		int32 targets_delay = opts->targets_delay;

	    //  book-keeping for multi-streams
	    std::vector<std::string> keys(num_stream);
	    std::vector<std::vector<int32> > feats(num_stream);
	    std::vector<Posterior> targets(num_stream);
	    std::vector<int> curt(num_stream, 0);
	    std::vector<int> lent(num_stream, 0);
	    std::vector<int> new_utt_flags(num_stream, 0);

	    // bptt batch buffer
	    //int32 feat_dim = nnet.InputDim();
	    Vector<BaseFloat> frame_mask(batch_size * num_stream, kSetZero);
	    Vector<BaseFloat> sorted_frame_mask(batch_size * num_stream, kSetZero);
	    Vector<BaseFloat> feat(batch_size * num_stream, kSetZero);
        Matrix<BaseFloat> featmat(batch_size * num_stream, 1, kSetZero);
        CuMatrix<BaseFloat> words(batch_size * num_stream, 1, kSetZero);
	    std::vector<int32> target(batch_size * num_stream, kSetZero);
	    std::vector<int32> sorted_target(batch_size * num_stream, kSetZero);
	    std::vector<int32> sortedclass_target(batch_size * num_stream, kSetZero);
	    std::vector<int32> sortedclass_target_index(batch_size * num_stream, kSetZero);
	    std::vector<int32> sortedclass_target_reindex(batch_size * num_stream, kSetZero);
	    std::vector<int32> sortedword_id(batch_size * num_stream, kSetZero);
	    std::vector<int32> sortedword_id_index(batch_size * num_stream, kSetZero);

	    LmNnetExample *example;
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
	            while ((example = dynamic_cast<LmNnetExample*>(repository_->ProvideExample())) != NULL)
	            {
	            	// checks ok, put the data in the buffers,
	            	keys[s] = example->utt;
	            	feats[s] = example->input_wordids;

	                num_done++;

	                curt[s] = 0;
	                lent[s] = feats[s].size() - 1;
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

	        // fill a multi-stream bptt batch
	        // * frame_mask: 0 indicates padded frames, 1 indicates valid frames
	        // * target: padded to batch_size
	        // * feat: first shifted to achieve targets delay; then padded to batch_size
	        for (int t = 0; t < batch_size; t++) {
	            for (int s = 0; s < num_stream; s++) {
	                // frame_mask & targets padding
	                if (curt[s] < targets_delay) {
	                	frame_mask(t * num_stream + s) = 0;
	                	target[t * num_stream + s] = feats[s][0];
	                }
	                else if (curt[s] < lent[s] + targets_delay) {
	                    frame_mask(t * num_stream + s) = 1;
	                    target[t * num_stream + s] = feats[s][curt[s]-targets_delay+1];
	                } else {
	                    frame_mask(t * num_stream + s) = 0;
	                    target[t * num_stream + s] = feats[s][lent[s]-1];
	                }
	                // feat shifting & padding
	                if (curt[s] < lent[s]) {
	                    feat(t * num_stream + s) = feats[s][curt[s]];
	                } else {
	                    feat(t * num_stream + s) = feats[s][lent[s]-1];

	                }

	                curt[s]++;
	            }
	        }

	                num_frames = feat.Dim();
	                // report the speed
	                if (num_done % 5000 == 0) {
	                  time_now = time.Elapsed();
	                  KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
	                                << time_now/60 << " min; processed " << total_frames/time_now
	                                << " frames per second.";
	                }

	        // apply optional feature transform
	        //nnet_transf.Feedforward(CuMatrix<BaseFloat>(feat), &feats_transf);

	        // for streams with new utterance, history states need to be reset
	        nnet.ResetLstmStreams(new_utt_flags);

        if (NULL != class_affine)
        {
	        // sort output class id
        	util.SortUpdateClass(target, sorted_target, sortedclass_target,
	        		sortedclass_target_index, sortedclass_target_reindex, frame_mask, sorted_frame_mask, word2class_);
	        class_affine->SetUpdateClassId(sortedclass_target, sortedclass_target_index, sortedclass_target_reindex);
	        cb_softmax->SetUpdateClassId(sortedclass_target);
        }

	        // sort input word id
	        util.SortUpdateWord(feat, sortedword_id, sortedword_id_index);
	        word_transf->SetUpdateWordId(sortedword_id, sortedword_id_index);

	        // forward pass
	        featmat.CopyColFromVec(feat, 0);
            words.CopyFromMat(featmat);

	        nnet.Propagate(words, &nnet_out);

	        // evaluate objective function we've chosen
	        if (objective_function == "xent") {
	        	xent.Eval(frame_mask, nnet_out, target, &nnet_diff);
	        } else if (objective_function == "cbxent") {
	        	cbxent.Eval(sorted_frame_mask, nnet_out, sorted_target, &nnet_diff);
	        } else {
	            KALDI_ERR << "Unknown objective function code : " << objective_function;
	        }


		        // backward pass
				if (!crossvalidate) {

                    if (model_sync->reset_gradient_[thread_idx] && parallel_opts->merge_func == "globalgradient")
                    {
                        nnet.ResetGradient();
                        model_sync->reset_gradient_[thread_idx] = false;
                        //KALDI_VLOG(1) << "Reset Gradient";
                    }

					// backpropagate
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
		        fflush(stderr);
		        fsync(fileno(stderr));
		}

		model_sync->LockStates();

		stats_->total_frames += total_frames;
		stats_->num_done += num_done;

		if (objective_function == "xent"){
			//KALDI_LOG << xent.Report();
			stats_->xent.Add(&xent);
		 }else if (objective_function == "cbxent"){
			//KALDI_LOG << xent.Report();
			stats_->cbxent.Add(&cbxent);
		 }else if (objective_function == "mse"){
			//KALDI_LOG << mse.Report();
			stats_->mse.Add(&mse);
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

void NnetLmUtil::SortUpdateClass(const std::vector<int32>& update_id, std::vector<int32>& sorted_id,
		std::vector<int32>& sortedclass_id, std::vector<int32>& sortedclass_id_index, std::vector<int32>& sortedclass_id_reindex,
			const Vector<BaseFloat>& frame_mask, Vector<BaseFloat>& sorted_frame_mask, const std::vector<int32> &word2class)
{
	int size = update_id.size();
	std::vector<Word> words(size);

	for (int i = 0; i < size; i++)
	{
		words[i].idx = i;
		words[i].wordid = update_id[i];
		words[i].classid = word2class[update_id[i]];
	}

	std::sort(words.begin(), words.end(), NnetLmUtil::compare_classid);

	sorted_id.resize(size);
	sortedclass_id.resize(size);
	sortedclass_id_index.resize(size);
	sortedclass_id_reindex.resize(size);

	for (int i = 0; i < size; i++)
	{
		sorted_id[i] = words[i].wordid;
		sortedclass_id[i] = words[i].classid;
		sortedclass_id_index[i] = words[i].idx;
		sortedclass_id_reindex[words[i].idx] = i;
		sorted_frame_mask(i) = frame_mask(words[i].idx);
	}
}

void NnetLmUtil::SortUpdateWord(const Vector<BaseFloat>& update_id,
		std::vector<int32>& sortedword_id, std::vector<int32>& sortedword_id_index)
{
	int size = update_id.Dim();
	std::vector<Word> words(size);

	for (int i = 0; i < size; i++)
	{
		words[i].idx = i;
		words[i].wordid = (int32)update_id(i);
		words[i].classid = (int32)update_id(i);
	}

	std::sort(words.begin(), words.end(), NnetLmUtil::compare_wordid);
	sortedword_id.resize(size);
	sortedword_id_index.resize(size);

	for (int i = 0; i < size; i++)
	{
		sortedword_id[i] = words[i].wordid;
		sortedword_id_index[i] = words[i].idx;
	}
}

void NnetLmUtil::SetClassBoundary(const Vector<BaseFloat>& classinfo,
		std::vector<int32> &class_boundary, std::vector<int32> &word2class)
{
	class_boundary.resize(classinfo.Dim());
	int32 num_class = class_boundary.size()-1;
    for (int i = 0; i < classinfo.Dim(); i++)
    	class_boundary[i] = classinfo(i);
	int i,j = 0;
	word2class.resize(class_boundary[num_class]);
	for (i = 0; i < class_boundary[num_class]; i++)
	{
		if (i>=class_boundary[j] && i<class_boundary[j+1])
			word2class[i] = j;
		else
			word2class[i] = ++j;
	}
}

void NnetLstmLmUpdateParallel(const NnetLstmLmUpdateOptions *opts,
		std::string	model_filename,
		std::string feature_rspecifier,
		Nnet *nnet,
		NnetLmStats *stats)
{
		ExamplesRepository repository;
		NnetModelSync model_sync(nnet, opts->parallel_opts);

		TrainLstmLmParallelClass c(opts, &model_sync,
								model_filename,
								&repository, nnet, stats);


	  {

		SequentialInt32VectorReader feature_reader(feature_rspecifier);

	    // The initialization of the following class spawns the threads that
	    // process the examples.  They get re-joined in its destructor.
	    MultiThreader<TrainLstmLmParallelClass> mc(opts->parallel_opts->num_threads, c);
	    NnetExample *example;
	    std::vector<NnetExample*> examples;
	    for (; !feature_reader.Done(); feature_reader.Next()) {
	    	example = new LmNnetExample(&feature_reader, opts);
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


