// online0/online-nnet-ipc-forwarding.h

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

#ifndef ONLINE0_ONLINE_NNET_IPC_FORWARDING_H_
#define ONLINE0_ONLINE_NNET_IPC_FORWARDING_H_

#include "util/circular-queue.h"
#include "thread/kaldi-mutex.h"
#include "nnet0/nnet-trnopts.h"
#include "nnet0/nnet-pdf-prior.h"

#include "online0/online-ipc-message.h"
#include "online0/kaldi-unix-domain-socket.h"

namespace kaldi {

struct OnlineNnetIpcForwardingOptions {
    typedef nnet0::PdfPriorOptions PdfPriorOptions;
    std::string feature_transform;
    std::string network_model;
    std::string socket_filename;
    bool no_softmax;
    bool apply_log;
    bool copy_posterior;
    std::string use_gpu;
    int32 gpuid;
    int32 num_threads;

    int32 time_shift;
    int32 batch_size;
    int32 num_stream;
    int32 dump_interval;
    int32 skip_frames;
    int32 sweep_time;
    std::string sweep_frames_str;
    int32 left_splice;
    int32 right_splice;
    int32 splice;

    const PdfPriorOptions *prior_opts;

    OnlineNnetIpcForwardingOptions(const PdfPriorOptions *prior_opts)
    	:feature_transform(""),network_model(""),socket_filename(""),no_softmax(true),apply_log(false),copy_posterior(true),use_gpu("no"),gpuid(-1),num_threads(1),
		 	 	 	 	 	 	 time_shift(0),batch_size(8),num_stream(10),dump_interval(0),
								 skip_frames(1), sweep_time(1), sweep_frames_str("0"),
								 left_splice(0), right_splice(0), splice(0), prior_opts(prior_opts)
    {

    }

    void Register(OptionsItf *po)
    {
    	po->Register("feature-transform", &feature_transform, "Feature transform in front of main network (in nnet0 format)");
    	po->Register("network-model", &network_model, "Main neural network model (in nnet0 format)");
    	po->Register("socket-filename", &socket_filename, "Unix domain socket file name");
    	po->Register("no-softmax", &no_softmax, "No softmax on MLP output (or remove it if found), the pre-softmax activations will be used as log-likelihoods, log-priors will be subtracted");
    	po->Register("apply-log", &apply_log, "Transform MLP output to logscale");
    	po->Register("copy-posterior", &copy_posterior, "Copy posterior for skip frames output");
    	po->Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");
        po->Register("gpuid", &gpuid, "gpuid < 0 for automatic select gpu, gpuid >= 0 for select specified gpu, only has effect if compiled with CUDA");


    	po->Register("num-threads", &num_threads, "Number of threads(GPUs) to use");


        //<jiayu>
    	po->Register("time-shift", &time_shift, "LSTM : repeat last input frame N-times, discrad N initial output frames.");
        po->Register("batch-size", &batch_size, "---LSTM--- BPTT batch size");
        po->Register("num-stream", &num_stream, "---LSTM--- BPTT multi-stream training");
        po->Register("dump-interval", &dump_interval, "---LSTM--- num utts between model dumping [ 0 == disabled ]");
        po->Register("skip-frames", &skip_frames, "LSTM model skip frames for next input");
        //</jiayu>

        sweep_time = skip_frames;
        po->Register("sweep-time", &sweep_time, "Sweep times for each utterance in skip frames training(Deprecated, use --sweep-frames instead)");
        po->Register("sweep-frames", &sweep_frames_str, "Sweep frames index for each utterance in skip frames decoding, e.g. 0");
        // +/-splice feature
        po->Register("left-splice", &left_splice, "number of frames to concatenate with left of the central frame, e.g. 0");
        po->Register("right-splice", &right_splice, "number of frames to concatenate with right of the central frame, e.g. 0");
        po->Register("splice", &splice, "number of frames to concatenate with the central frame, e.g. 0");
    }

};

class IpcForwardSync {
public:
    IpcForwardSync() {}

    void LockGpu() {
        gpu_mutex_.Lock();
    }

    void UnlockGpu(){
        gpu_mutex_.Unlock();
    }

private:
    Mutex gpu_mutex_;
};

class OnlineNnetIpcForwardingClass : public MultiThreadable {
private:
	const static int MAX_BUFFER_SIZE = 10;
	const static int MATRIX_INC_STEP = 1024;
	const OnlineNnetIpcForwardingOptions &opts_;
	std::vector<UnixDomainSocket*> &client_socket_;
    IpcForwardSync &forward_sync_;
	std::string model_filename_;

    // check client data sample validity
    inline bool CheckSample(SocketSample &sample, int input_dim) {
        int size = sample.dim * sample.num_sample;
        if (size < 0) {
            KALDI_LOG << Timer::CurrentTime() <<" Invalid sample, dim = " << sample.dim << " num_sample = " << sample.num_sample;
            return false;
        }
        else if (size > MAX_SAMPLE_SIZE) {
            KALDI_LOG << Timer::CurrentTime() <<" Client sample size " << size << " exceed maximum socket sample size " << MAX_SAMPLE_SIZE;
            return false;
        }
        else if (input_dim != sample.dim) {
            KALDI_LOG << Timer::CurrentTime() <<" Client sample dim " << sample.dim << " is not consistent with model input dim " << input_dim;
            return false;
        }
        else if (sample.is_end == 0 && sample.num_sample != opts_.batch_size) {
            KALDI_LOG << Timer::CurrentTime() << " number of frame in client sample " << sample.num_sample << " is not consistent with forward batch size " << opts_.batch_size;
            return false;
        }
    
        return true;
    }
        

public:
	OnlineNnetIpcForwardingClass(const OnlineNnetIpcForwardingOptions &opts,
			std::vector<UnixDomainSocket*> &client_socket,
			IpcForwardSync &forward_sync, std::string model_filename):
				opts_(opts), client_socket_(client_socket), 
                forward_sync_(forward_sync), model_filename_(model_filename)

	{

	}

	~OnlineNnetIpcForwardingClass() {}

	void operator () ()
	{
        forward_sync_.LockGpu();
#if HAVE_CUDA==1
    if (opts_.use_gpu == "yes") {
        if (opts_.gpuid < 0)
            CuDevice::Instantiate().SelectGpu();
        else
            CuDevice::Instantiate().SelectPreferGpu(opts_.gpuid);
    }
#endif
        forward_sync_.UnlockGpu();

        using namespace kaldi::nnet0;

		bool no_softmax = opts_.no_softmax;
		std::string feature_transform = opts_.feature_transform;
		bool apply_log = opts_.apply_log;
		//bool copy_posterior = opts_.copy_posterior;
		int32 num_stream = opts_.num_stream;
		int32 batch_size = opts_.batch_size;
		int32 skip_frames = opts_.skip_frames;
		int32 left_splice = opts_.left_splice == 0 ? opts_.splice : opts_.left_splice;
		int32 right_splice = opts_.right_splice == 0 ? opts_.splice : opts_.right_splice;
		const PdfPriorOptions *prior_opts = opts_.prior_opts;

		Nnet nnet_transf;

	    if (feature_transform != "") {
	      nnet_transf.Read(feature_transform);
	    }

	    Nnet nnet;
	    nnet.Read(model_filename_);

	    // optionally remove softmax,
	    Component::ComponentType last_type = nnet.GetComponent(nnet.NumComponents()-1).GetType();
	    if (no_softmax) {
	      if (last_type == Component::kSoftmax || last_type == Component::kBlockSoftmax) {
	        KALDI_LOG << "Removing " << Component::TypeToMarker(last_type) << " from the nnet " << model_filename_;
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
	    PdfPrior pdf_prior(*prior_opts);

		std::vector<int32> sweep_frames;
		if (!kaldi::SplitStringToIntegers(opts_.sweep_frames_str, ":", false, &sweep_frames))
			KALDI_ERR << "Invalid sweep-frames string " << opts_.sweep_frames_str;

		if (sweep_frames[0] > skip_frames || sweep_frames.size() > 1)
			KALDI_ERR << "invalid sweep frame index";

		int in_frames = MAX_SAMPLE_SIZE/(nnet.InputDim()/(left_splice+right_splice+1));
		int out_frames = MAX_OUTPUT_SIZE/nnet.OutputDim();
        if (opts_.copy_posterior)
            out_frames /= skip_frames; 
		int min_frames = in_frames < out_frames ? in_frames : out_frames;
		if (batch_size != min_frames) {
			batch_size = min_frames;
			KALDI_LOG << "According to network input/output cache, adjust batch size to " << batch_size;
		}

	    CuMatrix<BaseFloat>  cufeat, feats_transf, nnet_out;

	    SocketSample socket_sample;
	    SocketDecodable socket_decodable;
	    std::vector<CircularQueue<SocketDecodable>> decodable_buffer(num_stream);

	    Matrix<BaseFloat> sample;
	    std::vector<Matrix<BaseFloat> > feats(num_stream);
	    std::vector<int> curt(num_stream, 0);
	    std::vector<int> lent(num_stream, 0);
	    std::vector<int> frame_num_utt(num_stream, 0);
	    std::vector<int> utt_curt(num_stream, 0);
	    std::vector<int> new_utt_flags(num_stream, 1);
	    std::vector<int> update_state_flags(num_stream, 1);
	    std::vector<int> recv_end(num_stream, 0);
	    std::vector<int> send_end(num_stream, 1);
	    std::vector<int> send_frames(num_stream, 0);
	    std::vector<MatrixIndexT> splice_idx(left_splice+right_splice+1);
	    Matrix<BaseFloat> feat, nnet_out_host;
	    //int tranf_dim = nnet_transf.InputDim();
	    int input_dim = feature_transform != "" ? nnet_transf.InputDim() : nnet.InputDim();
	    int out_dim = nnet.OutputDim();
	    int t, s , k, len;

        Timer time, gap_time;
        double time_now = 0, time_send = 0, time_received = 0, time_forward = 0;

	    while (true)
	    {
            // send network output data
	    	for (s = 0; s < num_stream; s++)
	    	{
	    		while (!decodable_buffer[s].Empty() && send_end[s] == 0 && client_socket_[s] != NULL)
	    		{
	    			SocketDecodable* decodable = decodable_buffer[s].Front();
                    gap_time.Reset();
	    			int ret = client_socket_[s]->Send((void*)decodable, sizeof(SocketDecodable), MSG_NOSIGNAL);
                    time_send += gap_time.Elapsed();

	    			if (ret > 0 && ret != sizeof(SocketDecodable)) 
                        KALDI_WARN << Timer::CurrentTime() <<" Send socket decodable: " << ret << " less than " << sizeof(SocketDecodable);
	    			if (ret <= 0) break;
	    			// send successful
	    			decodable_buffer[s].Pop();
                    send_frames[s] += decodable->num_sample;

	    			// send a utterance finished
	    			if(decodable->is_end)
	    				send_end[s] = 1;
	    		}
	    	}

	    	// loop over all streams, feed each stream with new data
	    	for (s = 0; s < num_stream; s++)
	    	{
	    		if (client_socket_[s] == NULL)
	    			continue;

	    		if (client_socket_[s]->isClosed())
	    		{
	    			delete client_socket_[s];
	    			client_socket_[s] = NULL;
	    			send_end[s] = 1;
    				lent[s] = 0;
                    decodable_buffer[s].Resize(MAX_BUFFER_SIZE);
                    KALDI_LOG << Timer::CurrentTime() << " Client decoder " << s << " disconnected.";
	    			continue;
	    		}

	    		if (send_end[s]) {
	    			recv_end[s] = 0;
    				lent[s] = 0;
    				curt[s] = 0;
    				utt_curt[s] = 0;
    				new_utt_flags[s] = 1;
                    send_frames[s] = 0;
                    decodable_buffer[s].Resize(MAX_BUFFER_SIZE);
	    		}

	    		while (recv_end[s] == 0)
	    		{
                    gap_time.Reset();
                    len = client_socket_[s]->Receive((void*)&socket_sample, sizeof(SocketSample));
                    time_received += gap_time.Elapsed();

                    if (len <= 0)
                        break;

                    // socket sample validity
                    if (len != sizeof(SocketSample) || !CheckSample(socket_sample, input_dim)) { 
                        send_end[s] = 1;
                        client_socket_[s]->Close();
                        break;
                    }

                    if (feats[s].NumRows() == 0)
                        feats[s].Resize(MATRIX_INC_STEP, socket_sample.dim, kUndefined, kStrideEqualNumCols);
					if (feats[s].NumRows() < lent[s]+socket_sample.num_sample)
					{
						Matrix<BaseFloat> tmp(feats[s].NumRows()+MATRIX_INC_STEP, socket_sample.dim, kUndefined, kStrideEqualNumCols);
						tmp.RowRange(0, lent[s]).CopyFromMat(feats[s].RowRange(0, lent[s]));
						feats[s].Swap(&tmp);
					}

                    int size = socket_sample.dim * socket_sample.num_sample * sizeof(float);
                    if (size > 0)
					    memcpy((char*)feats[s].RowData(lent[s]), (char*)socket_sample.sample, size);
					lent[s] += socket_sample.num_sample;
					recv_end[s] = socket_sample.is_end;

					frame_num_utt[s] = lent[s]/skip_frames;
					frame_num_utt[s] += lent[s]%skip_frames > sweep_frames[0] ? 1 : 0;
	    		}
	    	}

	        // we are done if all streams are exhausted
	        bool done = true;
	        for (s = 0; s < num_stream; s++) {
	            if (curt[s] < lent[s] || (recv_end[s] == 1 && send_end[s] == 0)) done = false;  // this stream still contains valid data, not exhausted
	        }

	        if (done) {
	        	usleep(0.02*1000000);
	        	continue;
	        }

	    	if (feat.NumCols() != input_dim) {
	    		feat.Resize(batch_size * num_stream, input_dim, kSetZero, kStrideEqualNumCols);
	            nnet_out_host.Resize(batch_size * num_stream, out_dim, kSetZero, kStrideEqualNumCols);
	    	}

	    	 // fill a multi-stream bptt batch
	    	for (t = 0; t < batch_size; t++) {
	    		for (s = 0; s < num_stream; s++) {
					if (curt[s] < lent[s]) {
						feat.Row(t * num_stream + s).CopyFromVec(feats[s].Row(curt[s]));
					    curt[s] += skip_frames;
					    update_state_flags[s] = 1;
					} else {
						update_state_flags[s] = 0;
					}
	    		}
	    	}

            gap_time.Reset();

	    	// apply optional feature transform
	    	cufeat.Resize(feat.NumRows(), feat.NumCols(), kUndefined);
            cufeat.CopyFromMat(feat);
	    	nnet_transf.Propagate(cufeat, &feats_transf); // Feedforward

			// for streams with new utterance, history states need to be reset
			nnet.ResetLstmStreams(new_utt_flags);
			nnet.SetSeqLengths(new_utt_flags);
	    	// for streams with new data, history states need to be update
	    	nnet.UpdateLstmStreamsState(update_state_flags);

			// forward pass
			nnet.Propagate(feats_transf, &nnet_out);

	    	// convert posteriors to log-posteriors,
	    	if (apply_log) {
	    	  nnet_out.Add(1e-20); // avoid log(0),
	    	  nnet_out.ApplyLog();
	    	}

	    	// subtract log-priors from log-posteriors or pre-softmax,
	    	if (prior_opts->class_frame_counts != "") {
	    	  pdf_prior.SubtractOnLogpost(&nnet_out);
	    	}

			nnet_out.CopyToMat(&nnet_out_host);

            time_forward += gap_time.Elapsed();

            double curt_time = time.Elapsed();
            if (curt_time - time_now >= 2) {
                KALDI_LOG << Timer::CurrentTime() << " Thread " << this->thread_id_ << ", time elapsed: " << curt_time << " s, socket send: " << time_send 
                            << " s, socket receive: " << time_received << " s, gpu forward: " << time_forward << " s.";
                time_now = time.Elapsed();
            }

            // rearrange output for each client
			for (s = 0; s < num_stream; s++) {
				if (opts_.copy_posterior) {
					if ((utt_curt[s] >= lent[s] && send_end[s] == 1) || 
                        (utt_curt[s] >= lent[s] && recv_end[s] == 0 && send_end[s] == 0))
						continue;
				}
				else if ((utt_curt[s] >= frame_num_utt[s] && send_end[s] == 1) ||
                        (utt_curt[s] >= frame_num_utt[s] && recv_end[s] == 0 && send_end[s] == 0))
					continue;

				// get new decodable buffer
				decodable_buffer[s].Push();
				SocketDecodable *decodable = decodable_buffer[s].Back();
                decodable->clear(); 

				out_dim = nnet_out_host.NumCols();
				float *dest = decodable->sample;

				for (t = 0; t < batch_size; t++) {
					// feat shifting & padding
					if (opts_.copy_posterior) {
					   for (k = 0; k < skip_frames; k++){
							if (utt_curt[s] < curt[s] && utt_curt[s] < lent[s]) {
								memcpy((char*)dest, (char*)nnet_out_host.RowData(t * num_stream + s), out_dim*sizeof(float));
								dest += out_dim;
								utt_curt[s]++;
								decodable->num_sample++;
							}
					   }
					}
					else {
					   if (utt_curt[s] < curt[s] && utt_curt[s] < lent[s]) {
						   memcpy((char*)dest, (char*)nnet_out_host.RowData(t * num_stream + s), out_dim*sizeof(float));
						   dest += out_dim;
						   utt_curt[s]+=skip_frames;
						   decodable->num_sample++;
					   }
					}
				}

				decodable->dim = out_dim;
				decodable->is_end = recv_end[s] == 1 && ((opts_.copy_posterior && utt_curt[s] == lent[s]) ||
										(!opts_.copy_posterior && utt_curt[s] == frame_num_utt[s]));
				new_utt_flags[s] = decodable->is_end ? 1 : 0;
				send_end[s] = 0;
			} // rearrangement

	    } // while loop
	}
};



}// namespace kaldi

#endif /* ONLINE0_ONLINE_NNET_IPC_FORWARDING_H_ */
