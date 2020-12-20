/*!

\file

\ingroup VS_SCAMP5_MODULE

\author Jianing Chen

*/

#ifndef SCAMP5_API_INPUT_HPP
#define SCAMP5_API_INPUT_HPP

#include <vs_scamp5.hpp>
#include <scamp5_output.hpp>


/*!

	@brief put the exposure result in PIX to AREGs and reset PIX

	@param 	yf	full range [-128,127]
	@param	yh	half range [0,127]
	@param	gain (optional) gain [1,5]

*/
void scamp5_get_image(const areg_t& yf,const areg_t& yh,int gain=1);


/*!

    @brief load a analog value to the AREG with error&noise correction

    @param areg   target AREG
    @param value  analogue value to be loaded (in the range of [-128,127])
    @param temp   (optional) temporary kernel register to be used in the function

*/
void scamp5_in(const areg_t& areg,int8_t value,const areg_t& temp = SCAMP5_MACRO_T_REG);


/*!

    @brief load a analog value to the AREG plane without error&noise correction

    @param areg   target AREG
    @param value  analogue value to be loaded (in the range of [-128,127])
    @param temp   (optional) temporary kernel register to be used in the function

*/
void scamp5_load_in(const areg_t& areg,int8_t value,const areg_t& temp = SCAMP5_MACRO_T_REG);


/*!

	@brief load a analog value to IN without error&noise correction

    @param value  analogue value to be loaded (in the range of [-128,127])

	This function will load the value to IN, which can be used in kernels later.

    Example Usage:

    \code
    // R5 = where (C > 30)
    scamp5_load_in(30);
    scamp5_kernel_begin();
        sub(A,C,IN);
        where(A);
            MOV(R5,FLAG);
        all();
    scamp5_kernel_end();
    \endcode

*/
void scamp5_load_in(int8_t value);


/*!

    @brief load an analog value to the AREG plane using a raw DAC value

    @param areg   target AREG
    @param dac_db a 12-bit DAC value to use (in the range of [0,4095])
    @param temp   (optional) temporary kernel register to be used in the function

	This function will configure the DAC and then load whatever analogue value is
	obtained to the target register.

*/
void scamp5_load_dac(const areg_t& areg,uint16_t dac_db,const areg_t& temp = SCAMP5_MACRO_T_REG);


/*!

    @brief load an analog value to IN using a raw DAC value

    @param dac_db a 12-bit DAC value to use (in the range of [0,4095])

*/
void scamp5_load_dac(uint16_t dac_db);


/*!

    @brief shift an AREG image

    @param areg   	target AREG
    @param h      	amount of horizontal shift (positive: east, negative west)
    @param v   		amount of vertical shift (positive: north, negative south)

	Compared to use kernel operations to shift an AREG image, the method used inside
	this function is optimized.

    Example Usage:

    \code
    scamp5_shift(B,-3,2);
    \endcode

*/
void scamp5_shift(const areg_t& areg,int h,int v);


/*!

    @brief diffuse an AREG image

    @param target   	target AREG
    @param iterations   number of times to repeat the diffuse process
    @param vertical   	(optional) enables diffusion along vertical direction
    @param horizontal   (optional) enables diffusion along horizontal direction

    Example Usage:

    \code
    scamp5_diffuse(A,5);// diffuse along both direction and repeat 5 times
    \endcode

*/
void scamp5_diffuse(const areg_t& target,int iterations,bool vertical = true,bool horizontal = true,const areg_t& t0 = SCAMP5_MACRO_T_REG);


/*!

    @brief get the AREG sum level using a set of 4x4 sparse grid

    @param areg 		target AREG
    @param result16v 	pointer to an array of 16 uint8_t for result

    @return sum of the result when \p result16v is NULL

	When \p result16v is a NULL pointer, the function will return sum of the data.

    The result is not equal to the actual sum of all pixels of the AREG in the area,
    but it is proportional to the actual sum.
    This function takes roughly 14 usec to execute.

	\sa ::vs_scamp5_get_parameter for the range of the result summation.

*/
uint32_t scamp5_global_sum_16(const areg_t& areg,uint8_t*result16v=NULL);


/*!

    @brief get the AREG sum level using a set of 8x8 sparse grid

    @param areg 		target AREG
    @param result64v 	pointer to an array of 64 uint8_t for result

    @return sum of the result when \p result64v is NULL

	When \p result64v is a NULL pointer, the function will return sum of the data.

    This function achieves similar functionally as the 4x4 version, but the grid used is 8x8.
    As a consequence, the result has higher resolution but it takes longer to execute (roughly 28 usec).

*/
uint32_t scamp5_global_sum_64(const areg_t& areg,uint8_t*result64v=NULL);


/*!

    @brief 	get approximate sum level of the whole AREG plane
    @param 	areg 		target AREG
    @return the result

*/
uint8_t scamp5_global_sum_fast(const areg_t& areg);


/*!

    @brief get sum level of the pixels selected using a pattern
    @param areg 		target AREG
    @param r 			row address match
    @param c 			column address match
    @param rx 			row address mask
    @param cx 			column address mask
    @return the result

    This result is less probable to saturate because it only counts a quarter of the pixels in the AREG plane.

*/
uint8_t scamp5_global_sum_sparse(AREG areg,uint8_t r=2,uint8_t c=2,uint8_t rx=254,uint8_t cx=254);


/*
--------------------------------------------------------------------------------
*/


#ifdef S5D_M0_PRIVATE

#endif


#endif
