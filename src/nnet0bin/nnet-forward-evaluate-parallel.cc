// nnetbin/nnet-forward-evaluate.cc

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

#include <limits>

#include "nnet0/nnet-nnet.h"
#include "nnet0/nnet-loss.h"
#include "nnet0/nnet-pdf-prior.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"

#include "thread/kaldi-semaphore.h"
#include "thread/kaldi-mutex.h"
#include "thread/kaldi-thread.h"

namespace kaldi {
namespace nnet0 {
struct NnetForwardEvaluateOptions {
    std::string feature_transform;
    bool no_softmax;
    bool apply_log;
    bool copy_posterior;
    std::string use_gpu;
    int32 num_threads;

    int32 time_shift;
    int32 batch_size;
    int32 num_stream;
    int32 frame_dim;
    int32 num_iteration;

    const PdfPriorOptions *prior_opts;

    NnetForwardEvaluateOptions(const PdfPriorOptions *prior_opts)
    	:feature_transform(""),no_softmax(false),apply_log(false),copy_posterior(true),use_gpu("no"),num_threads(1),
		 	 	 	 	 	 	 time_shift(0),batch_size(20),num_stream(0),frame_dim(120),num_iteration(500),prior_opts(prior_opts)
    {

    }

    void Register(OptionsItf *po)
    {
    	po->Register("feature-transform", &feature_transform, "Feature transform in front of main network (in nnet format)");
    	po->Register("no-softmax", &no_softmax, "No softmax on MLP output (or remove it if found), the pre-softmax activations will be used as log-likelihoods, log-priors will be subtracted");
    	po->Register("apply-log", &apply_log, "Transform MLP output to logscale");
    	po->Register("copy-posterior", &copy_posterior, "Copy posterior for skip frames output");
    	po->Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");


    	po->Register("num-threads", &num_threads, "Number of threads(GPUs) to use");

        //<jiayu>
    	po->Register("time-shift", &time_shift, "LSTM : repeat last input frame N-times, discrad N initial output frames.");
        po->Register("batch-size", &batch_size, "---LSTM--- BPTT batch size");
        po->Register("num-stream", &num_stream, "---LSTM--- BPTT multi-stream training");
        //</jiayu>

        po->Register("num-iteration", &num_iteration, "times of nnet forward");
        po->Register("frame-dim", &frame_dim, "dim of input frame");

    }

};

struct NnetForwardEvaluateStats {

	int32 num_done;

	kaldi::int64 total_frames;

	NnetForwardEvaluateStats() { std::memset(this, 0, sizeof(*this)); }

	void Print(double time_now)
	{
	    // final message
	    KALDI_LOG << "Done " << num_done << " batches"
	              << " in " << time_now << " second,"
	              << " (fps " << total_frames/time_now << ")";
	}
};

class NnetForwardEvaluateParallelClass: public MultiThreadable {
private:
	const NnetForwardEvaluateOptions *opts;
	std::string model_filename;
	Mutex *examples_mutex;
	NnetForwardEvaluateStats *stats;

public:
	NnetForwardEvaluateParallelClass(const NnetForwardEvaluateOptions *opts,
							std::string model_filename,
							Mutex *examples_mutex,
							NnetForwardEvaluateStats *stats):
								opts(opts),model_filename(model_filename),
								examples_mutex(examples_mutex), stats(stats)
	{

	}

	  // This does the main function of the class.
	void operator ()()
	{

		examples_mutex->Lock();
		// Select the GPU
		#if HAVE_CUDA == 1
			if (opts->use_gpu == "yes")
		    	CuDevice::Instantiate().SelectGpu();
		    //CuDevice::Instantiate().DisableCaching();
		#endif

		examples_mutex->Unlock();

		bool no_softmax = opts->no_softmax;
		std::string feature_transform = opts->feature_transform;
		bool apply_log = opts->apply_log;
		int32 time_shift = opts->time_shift;
		const PdfPriorOptions *prior_opts = opts->prior_opts;
		//int32 num_stream = opts->num_stream;
		int32 batch_size = opts->batch_size;
		int32 frame_dim = opts->frame_dim;
		int32 num_iteration = opts->num_iteration;

		Nnet nnet_transf;

	    if (feature_transform != "") {
	      nnet_transf.Read(feature_transform);
          frame_dim = nnet_transf.InputDim();
	    }

	    Nnet nnet;
	    nnet.Read(model_filename);

	    // optionally remove softmax,
	    Component::ComponentType last_type = nnet.GetComponent(nnet.NumComponents()-1).GetType();
	    if (no_softmax) {
	      if (last_type == Component::kSoftmax || last_type == Component::kBlockSoftmax) {
	        KALDI_LOG << "Removing " << Component::TypeToMarker(last_type) << " from the nnet " << model_filename;
	        nnet.RemoveComponent(nnet.NumComponents()-1);
	      } else {
	        KALDI_WARN << "Cannot remove softmax using --no-softmax=true, as the last component is " << Component::TypeToMarker(last_type);
	      }
	    }

	    // avoid some bad option combinations,
	    if (apply_log && no_softmax) {
	      KALDI_ERR << "Cannot use both --apply-log=true --no-softmax=true, use only one of the two!";
	    }

	    // we will subtract log-priors later,
	    PdfPrior pdf_prior(*opts->prior_opts);

	    // disable dropout,
	    nnet_transf.SetDropoutRetention(1.0);
	    nnet.SetDropoutRetention(1.0);

	    kaldi::int64 total_frames = 0;


	    CuMatrix<BaseFloat> feats, feats_transf, nnet_out;
	    Matrix<BaseFloat> nnet_out_host, mat;
	    mat.Resize(batch_size, frame_dim);
	    mat.Set(1.0);

	    Timer time;
	    double time_now = 0;
	    int32 num_done = 0;
	    // iterate over all feature files
	    for (int i = 0; i < num_iteration; i++) {

	      // time-shift, copy the last frame of LSTM input N-times,
	      if (time_shift > 0) {
	        int32 last_row = mat.NumRows() - 1; // last row,
	        mat.Resize(mat.NumRows() + time_shift, mat.NumCols(), kCopyData);
	        for (int32 r = last_row+1; r<mat.NumRows(); r++) {
	          mat.CopyRowFromVec(mat.Row(last_row), r); // copy last row,
	        }
	      }

          // push it to gpu,
          feats = mat;

          // fwd-pass, feature transform,
          nnet_transf.Feedforward(feats, &feats_transf);

          // fwd-pass, nnet,
          nnet.Feedforward(feats_transf, &nnet_out);

	      // convert posteriors to log-posteriors,
	      if (apply_log) {
	        if (!(nnet_out.Min() >= 0.0 && nnet_out.Max() <= 1.0)) {
	          KALDI_WARN << num_done << " "
	                     << "Applying 'log' to data which don't seem to be probabilities "
	                     << "(is there a softmax somwhere?)";
	        }
	        nnet_out.Add(1e-20); // avoid log(0),
	        nnet_out.ApplyLog();
	      }

	      // subtract log-priors from log-posteriors or pre-softmax,
	      if (prior_opts->class_frame_counts != "") {
	        if (nnet_out.Min() >= 0.0 && nnet_out.Max() <= 1.0) {
	          KALDI_WARN << num_done << " "
	                     << "Subtracting log-prior on 'probability-like' data in range [0..1] "
	                     << "(Did you forget --no-softmax=true or --apply-log=true ?)";
	        }
	        pdf_prior.SubtractOnLogpost(&nnet_out);
	      }

	      // download from GPU,
	      nnet_out_host.Resize(nnet_out.NumRows(), nnet_out.NumCols());
	      nnet_out.CopyToMat(&nnet_out_host);

	      // time-shift, remove N first frames of LSTM output,
	      if (time_shift > 0) {
	        Matrix<BaseFloat> tmp(nnet_out_host);
	        nnet_out_host = tmp.RowRange(time_shift, tmp.NumRows() - time_shift);
	      }

	      // progress log
	      if (num_done % 100 == 0) {
	        time_now = time.Elapsed();
	        KALDI_VLOG(1) << "After " << num_done << " batches: time elapsed = "
	                      << time_now << " second; processed " << total_frames/time_now
	                      << " frames per second.";
	      }

	      num_done++;
	      total_frames += mat.NumRows();

	    }

	    examples_mutex->Lock();
	    stats->num_done += num_done;
	    stats->total_frames += total_frames;
	    examples_mutex->Unlock();

	}

};

void NnetForwardEvaluateParallel(const NnetForwardEvaluateOptions *opts,
						std::string	model_filename,
						NnetForwardEvaluateStats *stats)
{
    Mutex examples_mutex;

    NnetForwardEvaluateParallelClass c(opts, model_filename, &examples_mutex, stats);

    		// The initialization of the following class spawns the threads that
    	    // process the examples.  They get re-joined in its destructor.
    	    MultiThreader<NnetForwardEvaluateParallelClass> m(opts->num_threads, c);

}

} // namespace nnet0
} // namespace kaldi



int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet0;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Perform evaluate forward pass through Neural Network performance in parallel.\n"
        "\n"
        "Usage:  nnet-forward-evaluate-parallel [options] <model-in> \n"
        "e.g.: \n"
        " nnet-forward-evaluate-parallel --verbose=1 --num-threads=1 --batch-size=16 --num-iteration=500 --no-softmax=true --feature-transform=final.feature_transform final.nnet\n";

    ParseOptions po(usage);

    PdfPriorOptions prior_opts;
    prior_opts.Register(&po);

    NnetForwardEvaluateOptions opts(&prior_opts);
    opts.Register(&po);


    po.Read(argc, argv);

    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1);

    //Select the GPU
#if HAVE_CUDA==1
    if (opts.use_gpu == "yes")
    	CuDevice::Instantiate().Initialize();
    //CuDevice::Instantiate().DisableCaching();
#endif


    NnetForwardEvaluateStats stats;

    Timer time;
    double time_now = 0;

    KALDI_LOG << "Nnet Forward STARTED";

    NnetForwardEvaluateParallel(&opts, model_filename, &stats);

    KALDI_LOG << "Nnet Forward FINISHED; ";

    time_now = time.Elapsed();

    stats.Print(time_now);


#if HAVE_CUDA==1
    if (kaldi::g_kaldi_verbose_level >= 1) {
      CuDevice::Instantiate().PrintProfile();
    }
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
