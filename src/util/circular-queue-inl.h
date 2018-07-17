// util/circular-queue-inl.h

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

#ifndef KALDI_UTIL_CIRCULAR_QUEUE_INL_H_
#define KALDI_UTIL_CIRCULAR_QUEUE_INL_H_


namespace kaldi {

	template<class T>
	CircularQueue<T>::CircularQueue(int size)
	{
		KALDI_ASSERT(size>0);

		buffer_.resize(size+1);
		rear_ = buffer_.begin();
		back_ = front_ = rear_;
		size_ = 0;
	}

	template<class T>
	void CircularQueue<T>::Push()
	{
		typename std::list<T>::iterator it = front_;
        it++;
        if (it == buffer_.end())
            it = buffer_.begin();

		if (it != rear_)
		{
			back_ = front_;
			front_ = it;
		}
		else // it == rear_ : queue full
		{
			T value;
			// insert from front of front_
			buffer_.insert(front_, value);
			back_++;
		}
		size_++;
	}

	template<class T>
	void CircularQueue<T>::Pop()
	{
		if (rear_ != front_)
		{
			rear_++;
            if (rear_ == buffer_.end())
                rear_ = buffer_.begin();
			size_--;
		}
	}

	template<class T>
	T* CircularQueue<T>::Front()
	{
		KALDI_ASSERT(size_ > 0);
		return &(*rear_);
	}

	template<class T>
	T* CircularQueue<T>::Back()
	{
		KALDI_ASSERT(size_ > 0);
		return &(*back_);
	}

	template<class T>
	bool CircularQueue<T>::Empty()
	{
		if (size_ == 0)
			KALDI_ASSERT(rear_ == front_);
		return rear_ == front_;
	}

	template<class T>
	void CircularQueue<T>::Clear()
	{
		buffer_.resize(2);
		rear_ = buffer_.begin();
		back_ = front_ = rear_;
		size_ = 0;
	}

	template<class T>
	int CircularQueue<T>::Capacity()
	{
		return buffer_.size()-1;
	}

	template<class T>
	int CircularQueue<T>::Size()
	{
		return size_;
	}

	template<class T>
	void CircularQueue<T>::Resize(int size)
	{
		KALDI_ASSERT(size>0);

		buffer_.resize(size+1);
		rear_ = buffer_.begin();
		back_ = front_ = rear_;
		size_ = 0;
	}
}

#endif /* UTIL_CIRCULAR_QUEUE_INL_H_ */
