/*!

\file

\ingroup VS_COMMON_MODULE

\author Jianing Chen

*/

#ifndef VS_IPC_HPP
#define VS_IPC_HPP


#include <vs_message_buffer.hpp>
#include <vs_protocol.hpp>

#include <vs_io_agent.hpp>
#include <vs_flash_agent.hpp>
#include <vs_public_ram.hpp>

#include <functional>

#ifdef __LPC43XX__
class vs_ipc_api{

public:

    volatile uint16_t signal;
    volatile uint16_t state;
    volatile vs_mct64* ptr_mclk;
    volatile uint32_t loopc;
    vs_message_ring* ptr_m4_msg_ring;
    vs_message_ring* ptr_m0_msg_ring;
    vs_io_agent_api* ptr_io_agent_uart;
    vs_io_agent_api* ptr_io_agent_usb;
    vs_io_agent_api* ptr_io_agent_spi;
    vs_public_ram_api* ptr_prm;
    vs_flash_agent_api* ptr_flash_agent_qflash;
    const char* m4_version;
    const char* m0_version;
    volatile uint32_t m0_bulk_addr;
};
#else
class vs_ipc_api{
public:
    uint32_t loopc;
    const char* m4_version;
    const char* m0_version;
    void* m0_bulk_addr;
};
#endif



/*

\brief take messages out of the message queue of the core
\param p 	pointer to the destination messages for transferring to
\param cap 	maximum number of message to obtain

*/
uint16_t vs_ipc_pop_msg(vs_message_t*p,size_t cap);


/*

\brief add a message to M0 core's message queue
\param code		a code defining the type of the message
\param id		an customized id to distinguish between messages
\param data		the data attached to the message

If the message is meant to be processed by existing callback functions,
a specific "code1" need to be used.

For example, the following message will activate the ::vs_on_user_value callback:
\code
vs_ipc_push_msg_to_m0(VS_MSG_USER_VALUE,100,0);
\endcode

*/
void vs_ipc_push_msg_to_m0(uint8_t code,uint8_t id,int32_t data=0);


/*

\brief add a message to M4 core's message queue
\param code		a code defining the type of the message
\param id		an customized id to distinguish between messages
\param data		the data attached to the message

*/
void vs_ipc_push_msg_to_m4(uint8_t code,uint8_t id,int32_t data=0);


/*

\brief queue a customized message to for a core determined by the ".route" element

*/
void vs_ipc_push_msg(vs_message_t const& msg);


/*

\brief get the loop counter value from the M0 core's program

*/
uint32_t vs_ipc_get_loop_counter();


/*!

\brief define the function to execute when a message carrying user value is received

\param f    the callback function

*/
void vs_on_user_value(std::function<void(uint8_t id,int32_t value)> f);


/*

\brief callback function when a message sent to M4 core is received

*/
void vs_on_ipc_message(std::function<void(const vs_message_t*)> cb);


/*

\brief callback function when a message sent to M4 core is received

*/
void vs_on_ipc_packet(std::function<void(uint8_t*packet_buffer,size_t packet_length)> cb);


/*!

\brief get the version of the library
\param part		0 for M0 version, 1 for M4 version

*/
const char* vs_get_version(int part);


/*
--------------------------------------------------------------------------------
*/


#ifdef VS_LIB_PRIVATE


extern volatile vs_ipc_api *vs_ipc;


void vs_init_ipc(void);
void vs_routine_ipc(void);
void vs_ipc_process_message(void);

#endif


#ifdef S5D_M4_PRIVATE


struct flash_stream_packet{
    vs_packet_header_t header;
    vs_data_header_t def;
    uint8_t data_block[4096];
};

#ifdef __LPC43XX__
#define VS_PACKET_BUFFER_SZIE	512
extern int16_t usb_state;
extern uint8_t *usb_packet_buffer;
//extern uint8_t usb_packet_buffer[VS_PACKET_BUFFER_SZIE];
extern vs_protocol_in usb_protocol;
//extern uint8_t spi_packet_buffer[VS_PACKET_BUFFER_SZIE];
//extern vs_protocol_in spi_protocol;
#endif

extern std::function<void(const vs_message_t*)> m4_message_callback;
extern std::function<void(uint8_t*packet_buffer,size_t packet_length)> m4_packet_callback;

void vs_routine_ipc_m4(void);
void vs_ipc_mct_polling(vs_time t_ms);
void vs_ipc_process_message_m4(void);
void vs_ipc_packet_handler(vs_protocol_in*p,uint8_t*packet_buffer,size_t packet_length);

int system_message_filter(vs_protocol_in*p,const vs_message_t*msg);
int system_packet_filter(vs_protocol_in*p,uint8_t*packet_buffer,size_t packet_length);

#endif



#ifdef S5D_M0_PRIVATE

extern std::function<void(void)> m0_shutdown_callback;
extern std::function<void(const vs_message_t*)> m0_message_callback;
extern std::function<void(uint8_t*packet_buffer,size_t packet_length)> m0_packet_callback;

#endif


#endif
