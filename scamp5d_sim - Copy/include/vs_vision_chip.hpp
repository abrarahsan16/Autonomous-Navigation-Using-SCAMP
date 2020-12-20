/*!

\file 

\ingroup VS_M0_CORE_MODULE

\author Jianing Chen

*/

#ifndef VS_VISIONCHIP_HPP
#define VS_VISIONCHIP_HPP

#include <vs_import.hpp>


/*!
    @brief get various parameters of the vision chip

	Available values are:
	|            Macro Name            |                               Description                                |
	|----------------------------------|--------------------------------------------------------------------------|
	| SCAMP5_GLOBAL_SUM_16_AREG_MIN   | min. uniform AREG level required to obtain a min. 4x4 global sum output  |
	| SCAMP5_GLOBAL_SUM_16_AREG_MAX   | max. uniform AREG level required to obtain a max. 4x4 global sum output  |
	| SCAMP5_GLOBAL_SUM_16_FLOOR      | min. 4x4 global sum output                                               |
	| SCAMP5_GLOBAL_SUM_16_SPAN       | span (max - min) of the 4x4 global sum output                            |
	| SCAMP5_GLOBAL_SUM_64_AREG_MIN   | min. uniform AREG level required to obtain a min. 8x8 global sum output  |
	| SCAMP5_GLOBAL_SUM_64_AREG_MAX   | max. uniform AREG level required to obtain a max. 8x8 global sum output  |
	| SCAMP5_GLOBAL_SUM_64_FLOOR      | min. 8x8 global sum output                                               |
	| SCAMP5_GLOBAL_SUM_64_SPAN       | span (max - min) of the 8x8 global sum output                            |

*/
int32_t vs_vision_chip_get_parameter(int which);


/*!

    @brief reduce power consumption and register decay till next activity

	This function should also be called if a long MCU process is going to execute next.

	For example:
	- waiting for a io_agent that is likely taking a long time;
	- waiting a GPIO;
	- computing large amount of data.

	This is not necessary to for ::vs_wait_frame_trigger and ::vs_wait_io_agent.
	These two already have it inside.

*/
void vs_vision_chip_rest();


/*!
    @brief reset the light exposure currently accumulated on the sensor

*/
void vs_vision_chip_reset_exposure();


/*
--------------------------------------------------------------------------------
*/


#ifdef VS_M0_PRIVATE


void vs_init_vision_chip();
void vs_vision_chip_stop_clock();


#endif

#endif
