/*!

\file 

\ingroup VS_M0_CORE_MODULE

\author Jianing Chen

*/

#ifndef VS_FRAME_TRIGGER_HPP
#define VS_FRAME_TRIGGER_HPP


#include <vs_import.hpp>


#define VS_FRAME_TRIGGER_CONST_LOOP		0
#define VS_FRAME_TRIGGER_CONST_PIX		1
#define VS_FRAME_TRIGGER_GPIO_HIGH		2
#define VS_FRAME_TRIGGER_GPIO_LOW		3
#define VS_FRAME_TRIGGER_GPIO_RISE		4
#define VS_FRAME_TRIGGER_GPIO_FALL		5
#define VS_FRAME_TRIGGER_SYNC			6


/*!
    \brief configure the frame trigger

    \param mode  the mode
    \param arg   an argument of the mode

    | mode                           | explanation                                            | arg             |
    |--------------------------------|--------------------------------------------------------|-----------------|
    | \c VS_FRAME_TRIGGER_SYNC       | always wait for incoming frame trigger signal          | rate (Hz)       |
    | \c VS_FRAME_TRIGGER_CONST_LOOP | wait only if the current frame trigger is not overdue  | rate (Hz)       |
    | \c VS_FRAME_TRIGGER_CONST_PIX  | maintain exposure time on PIX constant                 | rate (Hz)       |
    | \c VS_FRAME_TRIGGER_GPIO_HIGH  | as long as a gpio is high                              | GPIO #          |
    | \c VS_FRAME_TRIGGER_GPIO_LOW   | as long as a gpio is low                               | GPIO #          |
    | \c VS_FRAME_TRIGGER_GPIO_RISE  | a rising edge on a gpio                                | GPIO #          |
    | \c VS_FRAME_TRIGGER_GPIO_FALL  | a falling edge on a gpio                               | GPIO #          |

	For GPIO number, see \ref vs_gpio.h.

	The first three modes behave differently when the frame trigger for the next frame
	is overdue at the moment of a ::vs_wait_frame_trigger call, as following:
	- "SYNC" mode will wait until next frame trigger signal. It is similar to how V-Sync works in computer graphics.
	- "CONST_LOOP" mode will immediately return to allow the frame loop to proceed, so the overall frame loop rate
	is similar to the given frame rate.
	- "CONST_PIX" will reset PIX and wait for a incoming frame trigger, thus the exposure time for PIX
	is kept at (1/framerate) sec but the frame loop rate is slower than the given frame rate.

*/
void vs_configure_frame_trigger(uint16_t mode,int arg);


/*!
    \brief configure the frame trigger to be \c VS_FRAME_TRIGGER_PERIODIC and set the frequency in fraction

    \param f_num  the numerator of a fraction of a second
    \param f_den  the denominator of a fraction of a second

*/
void vs_frame_trigger_set(uint16_t f_num,uint16_t f_den);


/*!

    \brief wait till the frame trigger interval has elapsed from its previous return

	e.g. throttle the frame loop to iterate 60 times per seconds:

	\code
	vs_frametrigger_set(1,60);// a frame trigger interval of 1/60 sec.
    while(1){
        vs_process_message();

        // this function will no return until 1/60 sec has elapsed from its previous return
        vs_wait_frametrigger();

		// ...

        vs_loop_counter_inc();
    }
	\endcode

*/
void vs_wait_frame_trigger();


/*!

    \brief enable frame trigger

	Frame trigger is enabled by default.

*/
void vs_enable_frame_trigger();


/*!

    \brief disable frame trigger

	When frame trigger is disabled, ::scamp5_wait_frame_trigger will immediately return.
	Thus, the frame rate entirely depends on the execution time of the loop body.

*/
void vs_disable_frame_trigger();


/*
--------------------------------------------------------------------------------
*/


#ifdef S5D_M0_PRIVATE

extern bool frame_trigger_enabled;
extern bool frame_trigger;
extern vs_mct64 frame_trigger_target;
extern vs_mct64 frame_trigger_reload;

#endif

#endif
