// online0/kaldi-unix-domain-socket.h

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

#ifndef ONLINE0_KALDI_UNIX_DOMAIN_SOCKET_H_
#define ONLINE0_KALDI_UNIX_DOMAIN_SOCKET_H_

#include <sys/un.h>
#include <sys/socket.h>	/* basic socket definitions */
#include <sys/ioctl.h>
#include <fcntl.h>		/* for nonblocking */
#include <unistd.h>

#include "base/kaldi-error.h"

class UnixDomainSocket {
public:
	UnixDomainSocket(std::string unix_filepath,
			int type = SOCK_STREAM, bool block = true) : socket_(-1), block_(block)
	{
		if ((socket_ = socket(AF_LOCAL, type, 0)) < 0) {
			const char *c = strerror(errno);
			if (c == NULL) { c = "[NULL]"; }
				KALDI_ERR << "Error open socket , errno was: " << c;
		}

        // set socket buffer
        int buffer_size = 4194304;
        setsockopt(socket_, SOL_SOCKET, SO_SNDBUF, &buffer_size, sizeof(int));
        setsockopt(socket_, SOL_SOCKET, SO_RCVBUF, &buffer_size, sizeof(int));
        
        //socklen_t len = sizeof(buffer_size);
        //getsockopt(socket_, SOL_SOCKET, SO_SNDBUF, &buffer_size, &len);
        //KALDI_LOG << "getsockopt SO_SNDBUF: " << buffer_size; 
        
        /*
        // set send/recv timeout
        struct timeval timeout = {20,0};
        setsockopt(socket_, SOL_SOCKET,SO_SNDTIMEO, (char *)&timeout, sizeof(struct timeval));
        setsockopt(socket_, SOL_SOCKET,SO_RCVTIMEO, (char *)&timeout, sizeof(struct timeval));
        */

		// non block connect
		if (!block) {
			int flags = fcntl(socket_, F_GETFL, 0);
			fcntl(socket_, F_SETFL, flags | O_NONBLOCK);
		}

		bzero(&socket_addr_, sizeof(sockaddr_un));
		socket_addr_.sun_family = AF_LOCAL;
		strcpy(socket_addr_.sun_path, unix_filepath.c_str());

		// connect server socket with a absolute local file path
		int ret = connect(socket_, (struct sockaddr*)&socket_addr_, sizeof(sockaddr_un));
		if (ret < 0 && (!block_ && errno != EINPROGRESS)) {
			const char *c = strerror(errno);
			if (c == NULL) { c = "[NULL]"; }
			KALDI_ERR << "Error connect socket with path " << unix_filepath << " , errno was: " << c;
		}
	}

	UnixDomainSocket() : socket_(-1), block_(true) {}

	UnixDomainSocket(int socket, struct sockaddr_un	socket_addr, bool block = true)
	: socket_(socket), block_(block), socket_addr_(socket_addr) {}

	UnixDomainSocket(int type = SOCK_STREAM) : socket_(-1), block_(true)
	{
		if ((socket_ = socket(AF_LOCAL, type, 0)) < 0) {
			const char *c = strerror(errno);
			if (c == NULL) { c = "[NULL]"; }
				KALDI_ERR << "Error open socket , errno was: " << c;
		}
	}

	~UnixDomainSocket()
	{
		close(socket_);
	}

	int Connect(std::string unix_filepath, bool block = true)
	{
		// non block connect
		if (!block) {
			int flags = fcntl(socket_, F_GETFL, 0);
			fcntl(socket_, F_SETFL, flags | O_NONBLOCK);
			block_ = block;
		}

		bzero(&socket_addr_, sizeof(sockaddr_un));
		socket_addr_.sun_family = AF_LOCAL;
		strcpy(socket_addr_.sun_path, unix_filepath.c_str());

		// connect server socket with a absolute local file path
		int ret = connect(socket_, (struct sockaddr*)&socket_addr_, sizeof(sockaddr_un));
		if ((ret < 0 && (!block_ && errno != EINPROGRESS)) || (ret < 0 && block_)) {
			const char *c = strerror(errno);
			if (c == NULL) { c = "[NULL]"; }
			KALDI_ERR << "Error connect socket with path " << unix_filepath << " , errno was: " << c;
		}
		return ret;
	}

	int Connect(std::string unix_filepath, int nmsec, bool block = true) //millisecond
	{
		fd_set rset, wset;
		struct timeval	tval;
		int error = 0;
		socklen_t len;

		// non block connect
		int flags = fcntl(socket_, F_GETFL, 0);
		fcntl(socket_, F_SETFL, flags | O_NONBLOCK);

		bzero(&socket_addr_, sizeof(sockaddr_un));
		socket_addr_.sun_family = AF_LOCAL;
		strcpy(socket_addr_.sun_path, unix_filepath.c_str());

		// connect server socket with a absolute local file path
		int ret = connect(socket_, (struct sockaddr*)&socket_addr_, sizeof(sockaddr_un));

		if (ret == 0)
			goto done; /* connect completed immediately */

		if (errno != EINPROGRESS) {
			const char *c = strerror(errno);
			if (c == NULL) { c = "[NULL]"; }
			KALDI_ERR << "Error connect socket with path " << unix_filepath << " , errno was: " << c;
		}

		FD_ZERO(&rset);
		FD_SET(socket_, &rset);
		wset = rset;
		tval.tv_sec = 0;
		tval.tv_usec = nmsec*1000; // microseconds

		ret = select(socket_+1, &rset, &wset, NULL, nmsec ? &tval : NULL);
		if (ret == 0) {
			close(socket_);		/* timeout */
			errno = ETIMEDOUT;
			return -1;
		}

		if (FD_ISSET(socket_, &rset) || FD_ISSET(socket_, &wset))
		{
			len = sizeof(error);
			if (getsockopt(socket_, SOL_SOCKET, SO_ERROR, &error, &len) < 0)
				return -1;
		}

	done:
		if (block) {
			fcntl(socket_, F_SETFL, flags);	/* restore file status flags */
			block_ = block;
		}

		if (error) {
			close(socket_);		/* just in case */
			errno = error;
			return -1;
		}
		return 0;
	}

	ssize_t Send(void *buff, size_t nbytes, int flags = 0)
	{
        ssize_t nsend = 0, n = 0, req = nbytes;
        while (nsend < nbytes) {
            n = send(socket_, (char*)buff+nsend, req, flags);
            if (n > 0) {
                nsend += n;
                req -= n; 
            }
            else {
                if (block_ && !(flags & MSG_DONTWAIT)) {
                	//const char *c = strerror(errno);
                	//if (c == NULL) { c = "[NULL]"; }
                	//KALDI_ERR << "Error send block socket, errno was: " << c;
                    return n;
                }
                if ((!block_ || (flags & MSG_DONTWAIT)) && (nsend == 0 && n <= 0)) 
                    return n;
                switch (errno) {
                case EPIPE :
                case ECONNRESET:
                    return n;
                }
            }
        }
        return nsend;
	}

	ssize_t Receive(void *buff, size_t nbytes, int flags = 0)
	{
        ssize_t nrecv = 0, n = 0, req = nbytes;
        int avali = 0;
        
        if (!block_ || (flags & MSG_DONTWAIT)) {
            ioctl(socket_, FIONREAD, &avali);
            if (avali < nbytes)
                return -1;
        }
            
        while (nrecv < nbytes) {
            n = recv(socket_, (char*)buff+nrecv, req, flags);
            if (n <= 0) {
                const char *c = strerror(errno);
                if (c == 0) { c = "[NULL]"; }
                KALDI_WARN << "Error receive block socket, errno was: " << c;
            	return n;
            }
            nrecv += n;
            req -= n;
        }
        return nrecv;
	}

	void Close()
	{
		close(socket_);
	}

	bool isClosed()
	{
		int error = 0;
        // disconect error
		socklen_t len = sizeof(error);
		int ret = getsockopt(socket_, SOL_SOCKET, SO_ERROR, &error, &len);
		if (error != 0 || ret != 0)
			return true;

        // socket closed normally
        char c;
        ssize_t n = recv(socket_, &c, 1, MSG_PEEK);
        if (n == 0)
            return true;

		return false;
	}

private:
	int socket_; // listening socket
	bool block_;
	struct sockaddr_un	socket_addr_;

};




#endif /* ONLINE0_KALDI_UNIX_DOMAIN_SOCKET_SERVER_H_ */
