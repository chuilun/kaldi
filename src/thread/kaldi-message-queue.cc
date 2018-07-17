// thread/kaldi-message-queue.cc

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


#include "base/kaldi-error.h"
#include "thread/kaldi-message-queue.h"


namespace kaldi {

	MessageQueue::MessageQueue(std::string pathname, int oflag, mode_t mode, struct mq_attr *attr):
		pathname_(pathname), oflag_(oflag), mode_(mode), attr_(attr)
	{
		if ( (mqd_ = mq_open(pathname.c_str(), oflag, mode, attr)) == (mqd_t)-1)
        {
            const char *c = strerror(errno);
            if (c == NULL) { c = "[NULL]"; }
            KALDI_ERR << "Error open message queue for " << pathname << " , errno was: " << c;
        }
	}

	MessageQueue::MessageQueue(): pathname_(""), mqd_(-1), oflag_(O_RDWR), mode_(FILE_MODE), attr_(NULL){}

	MessageQueue::~MessageQueue()
	{
        if (mqd_ != -1)
            mq_unlink(pathname_.c_str());
	}

	void MessageQueue::Open(std::string pathname, int oflag)
	{
		if ( (mqd_ = mq_open(pathname.c_str(), oflag, FILE_MODE, NULL)) == (mqd_t)-1)
        {
            const char *c = strerror(errno);
            if (c == NULL) { c = "[NULL]"; }
            KALDI_ERR << "Error open message queue for " << pathname << " , errno was: " << c;
        }
	}

	void MessageQueue::Create(std::string pathname, struct mq_attr *attr, int oflag)
	{

		while ( (mqd_ = mq_open(pathname.c_str(), oflag, FILE_MODE, attr)) == (mqd_t)-1)
		{
            const char *c = strerror(errno);
            if (c == NULL) { c = "[NULL]"; }
            KALDI_ERR << "Error creating message queue for " << pathname << " , errno was: " << c;
			//if (mq_unlink(pathname_.c_str()) == -1)
			//	KALDI_ERR << "Cannot create message queue for " << pathname;
		}
	}

	int MessageQueue::Send(char *ptr, size_t len, unsigned int prio)
	{
		int n;
		n = mq_send(mqd_, ptr, len, prio);
		return n;
	}

	ssize_t MessageQueue::Receive(char *ptr, size_t len, unsigned int *prio)
	{
		ssize_t	n = 0;

		n = mq_receive(mqd_, ptr, len, prio);
		return n;
	}

	void MessageQueue::Getattr(struct mq_attr *mqstat)
	{
		if (mq_getattr(mqd_, mqstat) == -1)
			KALDI_ERR << " Message queue getattr error";
	}

	void MessageQueue::Setattr(struct mq_attr *mqstat, struct mq_attr *omqstat)
	{
		if (mq_setattr(mqd_, mqstat, omqstat) == -1)
			KALDI_ERR << " Message queue setattr error";
	}
}

