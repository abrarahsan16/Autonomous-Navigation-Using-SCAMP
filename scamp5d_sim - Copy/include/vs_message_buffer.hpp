
#ifndef VS_MESSAGE_RING_HPP
#define VS_MESSAGE_RING_HPP


#include <vs_mutex.hpp>
#include <vs_ringbuffer.hpp>
#include <vs_message.hpp>


#define VS_MSG_RING_CAP   16


class vs_message_ring:public vs_mutex,public vs_ringbuffer<int16_t,vs_message_t,VS_MSG_RING_CAP>{

protected:

public:

};


#endif
