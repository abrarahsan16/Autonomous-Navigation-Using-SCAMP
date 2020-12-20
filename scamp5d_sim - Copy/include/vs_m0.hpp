/*!

\file

\ingroup VS_M0_CORE_MODULE

\author Jianing Chen

*/

#ifndef VS_M0_HPP
#define VS_M0_HPP


#include <vs_import.hpp>
#include <vs_interaction.hpp>
#include <vs_frame_trigger.hpp>
#include <vs_vision_chip.hpp>


/*!

	@brief initialize the vision system on M0

*/
void vs_init();


/*!

	@brief process the interactions with the host

	The function process all messages received from a host or the M4 core.
	When a type of message is registered with a handler function (e.g. host connect, gui item value update),
	the callback function will be called within this function.
	Therefore, this function need to be called routinely. See also \ref scamp5_sec_2.

*/
void vs_process_message(void);


/*!
@brief process the loop counter, messages and frame trigger at once.

This function essentially combine the following three function in one:
- ::vs_loop_counter_inc
- ::vs_wait_frame_trigger
- ::vs_process_message
In addition, it will make VS_LED_1 (the middle LED, usually blue) blinking.
Thus if this function is used, it's no longer necessary to have those three functions in a frame loop.

*/
void vs_frame_loop_control(void);


/*!

	@brief polled time delay

	@param usec  time in microsecond

*/
void vs_wait(uint32_t usec);


/*!

	@brief polled wait for a io_agent to finish sending

	@param p  io_agent pointer

*/
void vs_wait_io_agent(vs_io_agent_api*p);


/*!

	@brief callback function when the core is about to shutdown

	@param cb  callback function

*/
void vs_on_shutdown(std::function<void(void)> cb);


/*!
@brief set loop counter to 0

*/

void vs_loop_counter_reset(void);


/*!
@brief get the value of loop counter

*/
uint32_t vs_loop_counter_get(void);


/*!
@brief increase loop counter by 1

*/
void vs_loop_counter_inc(void);


#define VS_LC	vs_loop_counter_get()


#endif
