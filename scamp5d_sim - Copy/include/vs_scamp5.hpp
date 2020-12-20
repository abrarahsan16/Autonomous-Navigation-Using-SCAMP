/*!

\file

\ingroup VS_M0_CORE_MODULE

\author Jianing Chen

*/

#ifndef VS_SCAMP5_HPP
#define VS_SCAMP5_HPP

#include <vs_m0.hpp>


#define SCAMP5_GLOBAL_SUM_16_AREG_MIN	4
#define SCAMP5_GLOBAL_SUM_16_AREG_MAX	5
#define SCAMP5_GLOBAL_SUM_16_FLOOR		6
#define SCAMP5_GLOBAL_SUM_16_SPAN		7
#define SCAMP5_GLOBAL_SUM_64_AREG_MIN	8
#define SCAMP5_GLOBAL_SUM_64_AREG_MAX	9
#define SCAMP5_GLOBAL_SUM_64_FLOOR		10
#define SCAMP5_GLOBAL_SUM_64_SPAN		11


#ifdef __LPC43XX__
#include <scamp5_icw.hpp>
#else
#include <sim_instruction.hpp>
#endif


/*!

	@brief set the 12-bit DAC

*/
void vs_scamp5_set_dac(uint16_t dac12);


/*!
	@brief calibrate DAC voltages

*/
void vs_scamp5_calibrate_dac(int print_result = 0);


/*!
	@brief calibrate mid-point voltages

*/
void vs_scamp5_calibrate_mid(int print_result = 0);


/*!
	@brief calibrate global sum parameters

*/
void vs_scamp5_calibrate_global_sum();


/*!
	@brief configure a voltage source supplying the vision chip

*/
void vs_scamp5_configure_voltage(int e, uint16_t d);


/*!

	@brief configure the clocks of the vision chip. (use NULL to reset to default clocks.)

	Ask Steve to tailor a set of clock for your box.

*/
void vs_scamp5_configure_clock(const uint8_t(*clock_setting)[8]);


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
void vs_scamp5_rest();


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
int32_t vs_scamp5_get_parameter(int which);


/*!

	@brief execute a sequence of SCAMP-5 ICW

*/
void vs_scamp5_execute_icw(const void*icw_addr, uint32_t icw_count, bool highspeed = false);


/*
--------------------------------------------------------------------------------
*/


#ifdef S5D_M0_PRIVATE

struct scamp5_dac_calibration_data_t {
	uint16_t adc_sum[4096];// sum of the adc value of the 64 positions
	uint32_t dac_sample_sum[256];
	uint16_t dac_sample_count[256];
	uint8_t frame_samples[64];
};

extern uint16_t scamp5_dac_map[256];
extern int32_t scamp5_parameter[32];
extern int32_t scamp5_mean_8x8_ca[2];

void vs_scamp5_calibrate_mean_8x8();

#endif

#endif
