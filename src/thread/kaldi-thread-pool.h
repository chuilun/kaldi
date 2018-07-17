// thread/kaldi-thread-pool.h

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

#ifndef KALDI_THREAD_KALDI_THREAD_POOL_H_
#define KALDI_THREAD_KALDI_THREAD_POOL_H_

#include <pthread.h>
#include "itf/options-itf.h"
#include "thread/kaldi-semaphore.h"
#include "thread/kaldi-thread.h"


namespace kaldi {

template<class Threadable>
class ThreadPool
{
public:
	ThreadPool(int32 num_threads = 1):
		num_threads_(num_threads), done_(false)
	{
		if (num_threads_ < 1)
			num_threads_ = 1;

		threads_avail_(num_threads_);
		task_avail_(0);
		threads_ = new pthread_t[num_threads_];

		InitializeThreads();
	}

	bool RunTask(Threadable *task)
	{
		threads_avail_.Wait();
		mutex_.Lock();
		task_thread_.push_back(task);
		mutex_.Unlock();
		task_avail_.Signal();
		return true;
	}

	void Finsh()
	{
		for (int i = 0; i < num_threads_; i++)
			threads_avail_.Wait();

		mutex_.Lock();
		KALDI_ASSERT(task_thread_.empty());
		mutex_.Unlock();
		done_ = true;
		task_avail_.Signal();
	}

	~ThreadPool()
	{
		for (int i = 0; i < num_threads_; i++)
		{
			if (KALDI_PTHREAD_PTR(threads_[i]) != 0)
				if (pthread_join(threads_[i], NULL))
					KALDI_ERR << "Error rejoining thread.";
			delete [] threads_;
		}
	}

private:

	void InitializeThreads()
	{

		for (int i = 0; i < num_threads_; i++)
		{
		    int32 ret;
		    if ((ret=pthread_create(&(threads_[i]),
		                            NULL, // default attributes
									ThreadPool::ThreadExecute,
		                            static_cast<void*>(this)))) {
		      const char *c = strerror(ret);
		      KALDI_ERR << "Error creating thread, errno was: " << (c ? c : "[NULL]");
		    }
		}
	}

	Threadable* FetchTask()
	{
		task_avail_.Wait();

		if (done_)
		{
			KALDI_ASSERT(task_thread_.empty());
			// Increment the semaphore so the call by the next thread will not block.
			task_avail_.Signal();
			return NULL;
		}
		else
		{
			mutex_.Lock();
			KALDI_ASSERT(!task_thread_.empty());
			Threadable *task =	task_thread_.front();
			task_thread_.pop_front();
			mutex_.Unlock();
			threads_avail_.Signal();
			return task;
		}
	}

	static void ThreadExecute(void *param)
	{
		Threadable *task = NULL;

		while ((task = (static_cast<ThreadPool*>(param))->FetchTask()) != NULL)
		{
			task();
			delete	task;
			task = NULL;
		}
	}

	int32 num_threads_;
	Semaphore threads_avail_; // Initialized to the number of threads we are
	  // supposed to run with; the function Run() waits on this.
	Semaphore task_avail_;
	Mutex mutex_;
	bool done_;

	std::deque<Threadable*> task_thread_;
	pthread_t *threads_;

	KALDI_DISALLOW_COPY_AND_ASSIGN(ThreadPool);
};

} // namespace kaldi

#endif  // KALDI_THREAD_KALDI_THREAD_POOL_H_
