// online0/online-nnet-ipc-decoding.h

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

#ifndef ONLINE0_ONLINE_NNET_IPC_DECODING_H_
#define ONLINE0_ONLINE_NNET_IPC_DECODING_H_

#include "fstext/fstext-lib.h"
#include "decoder/decodable-matrix.h"
#include "thread/kaldi-semaphore.h"
#include "thread/kaldi-mutex.h"

#include "online0/online-nnet-faster-decoder.h"
#include "online0/kaldi-unix-domain-socket.h"
#include "online0/online-ipc-message.h"

namespace kaldi {

struct OnlineNnetIpcDecodingOptions {
	const OnlineNnetFasterDecoderOptions &decoder_opts;

	BaseFloat acoustic_scale;
	bool allow_partial;
	std::string word_syms_filename;
	int32 chunk_length_secs;
	std::string silence_phones_str;

	OnlineNnetIpcDecodingOptions(const OnlineNnetFasterDecoderOptions &decoder_opts):
                            decoder_opts(decoder_opts),
							acoustic_scale(0.1), allow_partial(true), word_syms_filename(""),
							silence_phones_str("")
    { }

	void Register(OptionsItf *po)
	{
		po->Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");
		po->Register("allow-partial", &allow_partial, "Produce output even when final state was not reached");
		po->Register("word-symbol-table", &word_syms_filename, "Symbol table for words [for debug output]");

		po->Register("silence-phones", &silence_phones_str,
		                     "Colon-separated list of integer ids of silence phones, e.g. 1:2:3");
	    po->Register("chunk-length", &chunk_length_secs,
	                "Length of chunk size in seconds, that we process.  Set to <= 0 "
	                "to use all input in one chunk.");
	}
};

class DecoderSync
{
public:
	DecoderSync() : sema_utt_(0), is_finished_(false) {}
	~DecoderSync() {}

	void UtteranceWait()
	{
		sema_utt_.Wait();
	}

	void UtteranceSignal()
	{
		sema_utt_.Signal();
	}

	std::string GetUtt()
	{
		return cur_utt_;
	}

	void SetUtt(std::string utt)
	{
		mutex_.Lock();
		cur_utt_ = utt;
		mutex_.Unlock();
	}

    bool IsFinsihed()
    {
        return is_finished_;
    }

    void Abort() 
    {
        is_finished_ = true;
    }

private:
	Semaphore sema_utt_;
	Mutex mutex_;
	std::string cur_utt_;
	bool is_finished_;
};

class OnlineNnetIpcDecodingClass : public MultiThreadable
{
public:
	OnlineNnetIpcDecodingClass(const OnlineNnetIpcDecodingOptions &opts,
			OnlineNnetFasterDecoder *decoder,
			UnixDomainSocket *socket,
			DecoderSync *decoder_sync,
			const TransitionModel &trans_model,
			fst::SymbolTable &word_syms,
			Int32VectorWriter &words_writer,
			Int32VectorWriter &alignment_writer):
				opts_(opts),
				decoder_(decoder), socket_(socket), decoder_sync_(decoder_sync),
				trans_model_(trans_model), word_syms_(word_syms),
				words_writer_(words_writer), alignment_writer_(alignment_writer)
	{
		decodable_ = new OnlineDecodableMatrixMapped(trans_model_, opts.acoustic_scale);
	}

	~OnlineNnetIpcDecodingClass() {}

	void operator () ()
	{
		fst::VectorFst<LatticeArc> out_fst;
		std::vector<int32> word_ids;
		std::vector<int32> tids;
		typedef OnlineNnetFasterDecoder::DecodeState DecodeState;
		int batch_size = opts_.decoder_opts.batch_size;
		SocketDecodable decodable;
		int num_pdfs = trans_model_.NumPdfs();
		std::string utt;

		while (!decoder_sync_->IsFinsihed())
		{
			decoder_->ResetDecoder(true);
			decoder_->InitDecoding();
			decodable_->Reset();

			while (true)
			{
				socket_->Receive((char*)&decodable, sizeof(SocketDecodable), MSG_WAITALL);
				Matrix<BaseFloat> loglikes(decodable.num_sample, num_pdfs, kUndefined, kStrideEqualNumCols);
				memcpy(loglikes.Data(), decodable.sample, loglikes.SizeInBytes());
				decodable_->AcceptLoglikes(&loglikes);

				while (decodable_->NumFramesReady() >= decoder_->NumFramesDecoded()+batch_size)
				{
					decoder_->Decode(decodable_);
					if (decoder_->PartialTraceback(&out_fst))
					{
						fst::GetLinearSymbolSequence(out_fst, static_cast<std::vector<int32> *>(0), 
                                                            &word_ids, static_cast<LatticeArc::Weight*>(0));
						PrintPartialResult(word_ids, &word_syms_, false);
					}
				}

				if (decodable.is_end)
				{
					utt = decoder_sync_->GetUtt();

					decodable_->InputIsFinished();
					decoder_->Decode(decodable_);

					decoder_->FinishTraceBack(&out_fst);
					fst::GetLinearSymbolSequence(out_fst, static_cast<std::vector<int32> *>(0), 
                                                        &word_ids, static_cast<LatticeArc::Weight*>(0));
					PrintPartialResult(word_ids, &word_syms_, true);

					// get best full path
					decoder_->GetBestPath(&out_fst);
					fst::GetLinearSymbolSequence(out_fst, &tids, &word_ids, static_cast<LatticeArc::Weight*>(0));

                    /*
					if (!word_ids.empty())
						words_writer_.Write(utt, word_ids);
					alignment_writer_.Write(utt, tids);
					*/

					decoder_sync_->UtteranceSignal();
					break;
				}

			} // message queue
		}
	}

private:

	const OnlineNnetIpcDecodingOptions &opts_;
	OnlineNnetFasterDecoder *decoder_;
	UnixDomainSocket *socket_;
	DecoderSync *decoder_sync_;
	OnlineDecodableMatrixMapped *decodable_;
	const TransitionModel &trans_model_; // needed for trans-id -> phone conversion
	fst::SymbolTable &word_syms_;
	Int32VectorWriter &words_writer_;
	Int32VectorWriter &alignment_writer_;
};


}// namespace kaldi

#endif /* ONLINE0_ONLINE_NNET_DECODING_MQUEUE_H_ */
