
#ifndef VS_IPC_IMPORT_HPP
#define VS_IPC_IMPORT_HPP


extern "C"{
#include <vs_common.h>
#include <vs_led.h>
#include <vs_gpio.h>
}


// software object
#include <vs_mutex.hpp>
#include <vs_message.hpp>
#include <vs_message_buffer.hpp>
#include <vs_post.hpp>
#include <vs_dotmat.hpp>
#include <vs_random.hpp>
#include <vs_stopwatch.hpp>


// interface object
#include <vs_io_agent.hpp>
#include <vs_flash_agent.hpp>
#include <vs_public_ram.hpp>
#include <vs_ipc.hpp>


/*
--------------------------------------------------------------------------------
*/


#ifdef S5D_M0_PRIVATE

void vs_init_import();

#endif


#endif
