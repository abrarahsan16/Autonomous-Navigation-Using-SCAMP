/*!

\file

\ingroup VS_SCAMP5_MODULE

\author Jianing Chen

*/

#ifndef SCAMP5_API_HPP
#define SCAMP5_API_HPP

#include <vs_scamp5.hpp>
#include <scamp5_output.hpp>


/*!

    @brief shift an DREG image

    @param dreg   	target DREG
    @param h      	amount of horizontal shift (positive: east, negative west)
    @param v   		amount of vertical shift (positive: north, negative south)
    @param boundary whether boundaries are '0' or '1'

    Same as the ::DNEWS kernel function, this function will modify R1, R2, R3, R4

    Example Usage:

    \code
    scamp5_shift(R5,-3,2);
    \endcode

*/
void scamp5_shift(DREG dreg,int h,int v,const int boundary=0);


/*!

    @brief get OR result of all pixel in a DREG plane
    @param r    (optional) mask pattern row index to match
    @param c    (optional) mask pattern pixel column index to match
    @param rx   (optional) mask pattern row index bits to ignore
    @param cx   (optional) mask pattern column index bits to ignore

    @return 0 if all pixel is 0, non-zero otherwise.

    The default mask pattern parameters gives the whole image.
    ::scamp5_load_pattern can be used to work out the correct parameter for a desired mask pattern,

*/
int scamp5_global_or(DSRC dreg,uint8_t r=0,uint8_t c=0,uint8_t rx=255,uint8_t cx=255);


/*!

	@brief get a estimation of the number of '1's in a DREG plane

	@param dreg  	the DREG to be checked
	@param temp     an AREG to use as temporary storage
	@param options  0: use 4x4 global sum, 1: use 8x8 global sum

	@return ranged in [0,4095] proportional to the number of '1's

	Internally this function convert the DREG image to AREG image using suitable analog levels
	to represent '0's and '1's. Then a AREG global sum is done and the result is uniformed into
	[0,4095].

*/
int scamp5_global_count(DREG_SRC dreg, AREG_DST temp, int options = 0);


/*!

	@brief flooding a DREG image with '1's from the '1's in the DREG image

	@param dreg_target  the DREG to be flooded
	@param dreg_mask    the control DREG to inhibit the flood
	@param boundary     boundary value
	@param iterations   (optional) number of iterations
	@param use_local    (optional) use pre-exist R1-R4 to control the flood direction

	The target DREG will be flooded by '1's except those areas that are closed by '0's in the control register.
	Since SCAMP-5 vision chip is able to perform this process in parallel and asynchronously,
	so the time cost to do the flooding is order of magnitude less than serial processing or
	step-by-step parallel propagation.

	\image html images\module_async_propagation2.png

	To increase the travelling distance of the flood, the number of iteration should be increased.

	The following image shows the result of a flood starting from a single pixel at (128,128)
	after 1 iteration:
	\image html images\module_async_point_flood.png

	A flooding with boundary value = 1:
	\image html images\module_async_propagation.png


	Example Usage:

	Flooding R5 from a pixel at (100,50) using R6 as mask:
	\code
	scamp5_load_point(R5,100,50);
	scamp5_flood(R5,R6,0);
	\endcode

*/
void scamp5_flood(DREG_DST dreg_target, DREG_SRC dreg_mask, int boundary, int iterations = 5, bool use_local = false);


/*!
	@brief set a single pixel on a DREG image to 1, the rest to 0

	@param dr    target DREG
	@param r     pixel row index
	@param c   	 pixel column index

*/
void scamp5_load_point(DREG_DST dr, uint8_t r, uint8_t c);


/*!
	@brief set a rectangle area on a DREG image to 1, the rest to 0

	@param dr   target DREG
	@param r0   pixel row index of the top right corner
	@param c0   pixel column index of the top right corner
	@param r1   pixel row index of the bottom left corner
	@param c1   pixel column index of the bottom left corner

*/
void scamp5_load_rect(DREG_DST dr, uint8_t r0, uint8_t c0, uint8_t r1, uint8_t c1);


/*!
	@brief set those pixels with matching address to 1, the rest to 0

	@param dr   target DREG
	@param r    pixel row index to match
	@param c    pixel column index to match
	@param rx   bits of the pixel row index to mask out
	@param cx   bits of the pixel column index to mask out

	To mask out certain bits in the column/row address allow a match to occur periodically.
	For example, set mask to 192 (11000000b) makes the four following address
	match the value 3 (00000011b): 3(00000011b), 67(01000011b), 131(10000011b) and 195(11000011b).

*/
void scamp5_load_pattern(DREG_DST dr, uint8_t r, uint8_t c, uint8_t rx, uint8_t cx);


/*!

	@brief same as ::scamp5_load_point, but put the point in SELECT

	Example Usage:

	\code
	// clr R5(100,100)
	scamp5_select_point(100,100);
	scamp5_kernel_begin();
		CLR_IF(R5,SELECT);
	scamp5_kernel_end();
	\endcode

*/
void scamp5_select_point(uint8_t r, uint8_t c);


/*!

	@brief same as ::scamp5_load_rect, but put the rectangle in RECT (also modify SELECT)

*/
void scamp5_select_rect(uint8_t r0, uint8_t c0, uint8_t r1, uint8_t c1);


/*!

	@brief same as ::scamp5_load_pattern, but put the pattern in SELECT

*/
void scamp5_select_pattern(uint8_t r, uint8_t c, uint8_t rx, uint8_t cx);


/*!

	@brief select column

*/
void scamp5_select_col(uint8_t c);


/*!

	@brief select row

*/
void scamp5_select_row(uint8_t r);


/*!

	@brief select column mask

*/
void scamp5_select_colx(uint8_t cx);


/*!

	@brief select row mask

*/
void scamp5_select_rowx(uint8_t rx);


/*!

	@brief targeting a DREG image to a series of drawing operations

	@param dr   target DREG

	This allow the 'scamp5_draw_' functions to be used to draw 1's on to a DREG image.

*/
void scamp5_draw_begin(DREG_DST dr);


/*!

	@brief end the drawing operations

*/
void scamp5_draw_end();


/*!

	@brief draw a point, wrap around if it's outside the border

	@param r 	row coordinate
	@param c 	column coordinate

*/
void scamp5_draw_pixel(uint8_t r, uint8_t c);


/*!

	@brief draw a point when its position is within the image

	@param r 	row coordinate
	@param c 	column coordinate

	\return whether the point is inside the image and drawn

*/
bool scamp5_draw_point(int r, int c);


/*!
	@brief draw a rectangle

	@param r0   pixel row index of the top right corner
	@param c0   pixel column index of the top right corner
	@param r1   pixel row index of the bottom left corner
	@param c1   pixel column index of the bottom left corner

*/
void scamp5_draw_rect(uint8_t r0, uint8_t c0, uint8_t r1, uint8_t c1);


/*!

	@brief draw a line

	@param r0 	starting point row coordinate
	@param c0 	starting point column coordinate
	@param r1 	finishing point row coordinate
	@param c1 	finishing point column coordinate
	@param repeat whether to wrap around when point goes outside the image

*/
void scamp5_draw_line(int r0, int c0, int r1, int c1, bool repeat = false);


/*!

	@brief draw a circle

	@param r0 	center point row coordinate
	@param c0 	center point column coordinate
	@param radius 	radius of the circle

*/
void scamp5_draw_circle(int x0, int y0, int radius, bool repeat = false);


/*!

	@brief do a binary inversion of the DREG image.

*/
void scamp5_draw_negate();


/*
--------------------------------------------------------------------------------
*/


#ifdef S5D_M0_PRIVATE

#endif

#endif
