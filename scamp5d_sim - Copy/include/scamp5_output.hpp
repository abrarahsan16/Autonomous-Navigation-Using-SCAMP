/*!

\file

\ingroup VS_SCAMP5_MODULE

\author Jianing Chen

*/

#ifndef SCAMP5_OUTPUT_HPP
#define SCAMP5_OUTPUT_HPP

#include <vs_scamp5.hpp>


/*!
	\brief set the output port the scamp5 output functions

	\param p 		pointer of a 'io_agent'

*/
void scamp5_bind_io_agent(vs_io_agent_api*p);


/*!
	\brief set the output port the scamp5 output functions

	@return		pointer to the io agent

*/
vs_io_agent_api*scamp5_get_io_agent();


/*!
	\brief read a single pixel

	\param areg 		AREG
	\param r0		row address
	\param c0		column address
	@return			ADC result, ranged from [0,255]

*/
uint8_t scamp5_read_areg(AREG_SRC areg, uint8_t r, uint8_t c);


/*!
	\brief scan a customized grid of pixels in an AREG image

	\param areg 	AREG
	\param buffer 	pointer to a buffer to store the results
	\param r0		row start
	\param c0		column start
	\param r1		row stop
	\param c1		column stop
	\param rs		row step
	\param cs		column step

	Note, the result image is stored in column-major format, starting from top right.
	i.e. "buffer[3]" is the pixel the on 1st column right, 4th row down.
	This applies to all "scamp5_scan_areg_*" series functions.
*/
void scamp5_scan_areg(AREG_SRC areg, uint8_t*buffer, uint8_t r0, uint8_t c0, uint8_t r1, uint8_t c1, uint8_t rs, uint8_t cs);


/*!
	\brief scan a 8x8 grid of pixels in an AREG image

	This function is slightly faster and more accurate than ::scamp5_scan_areg.
*/
void scamp5_scan_areg_8x8(AREG_SRC areg, uint8_t*result8x8);


/*!
	\brief divide the AREG image into 8x8 square blocks, and get the average of each block

	\param areg 		AREG
	\param result8x8 	pointer to a buffer to store the results

*/
void scamp5_scan_areg_mean_8x8(AREG_SRC areg, uint8_t*result8x8);


/*!
	\brief scan DREG image, store the result in a buffer

	\param dreg 		DREG
	\param mem 			pointer to a buffer
	\param r0 			first row index
	\param r1 			last row index

	The size of the buffer need to be a least 32 times the number of rows to scan.
	Thus, a full DREG image requires a buffer of 8192 bytes.
*/
void scamp5_scan_dreg(DREG_SRC dreg, uint8_t*mem, uint8_t r0 = 0, uint8_t r1 = 255);


/*!
	\brief scan the coordinates of all '1's in a DREG image and store the result in a buffer

	\param dreg 		DREG
	\param mem 			pointer to a buffer of (max_num*2) bytes
	\param max_num		maximum number of events
	\param h_dir		horizontal order of scanning, '0' means from column 0 to column 255
	\param v_dir		vertical order of scanning, '0' means from row 0 to row 255

	Up-to 4000 events scan be scanned.
	Note, the execution time of this function is proportional to \p max_num.
	In the result, the first occurrence of (0,0) indicates the end of effective events.

*/
void scamp5_scan_events(DREG_SRC dreg, uint8_t*mem, uint16_t max_num = 1000, uint8_t h_dir = 0, uint8_t v_dir = 0);


/*!
	\brief scan the bounding box coordinates of '1's in a DREG image

	\param dreg 		DREG
	\param vec4v 		pointer to a buffer of 4 byte

	The coordinates are two points: the top-right and the bottom-left corners of
	the axis-aligned bounding box (AABB) of '1's in the DREG.

*/
void scamp5_scan_boundingbox(DREG_SRC dr, uint8_t*vec4v);


/*!
	\brief output a full sized image of an AREG through the active io agent

	\param areg 		AREG
	\param display 		handle of a gui display

*/
void scamp5_output_image(AREG_SRC areg, vs_handle display);


/*!
	\brief output a full sized image of a DREG through the active io agent

	\param dreg 		DREG
	\param display 		handle of a gui display

*/
void scamp5_output_image(DREG_SRC dreg, vs_handle display);


/*!
	\brief output a sub sampled AREG image through the active io agent

	\param dreg 		AREG
	\param display 		handle of a gui display
	\param s 			sub-sample divisor
	\param r0 			pixel phase when sub-sampling
	\param c0 			pixel phase when sub-sampling

	E.g. when \p den is 2, the image size will be 128x128.

*/
void scamp5_output_areg(AREG_SRC areg, vs_handle display, int s = 1, int r0 = 0, int c0 = 0);


/*!
	\brief output the bounding box coordinates of the '1's in a DREG through the active io agent

	\param dreg 		DREG
	\param display 		handle of a gui display
	\param max_num		upper limit of the number of events
	\param h_dir		horizontal order of scanning, '0' means from column 0 to column 255
	\param v_dir		vertical order of scanning, '0' means from row 0 to row 255

	Time cost of this function is proportional to the \p max_num, which is the maximum number of events to scan.
	The limit of \p max_num is 4000.

*/
void scamp5_output_events(DREG_SRC dreg, vs_handle display, uint16_t max_num = 1000, uint8_t h_dir = 0, uint8_t v_dir = 0);


/*!
	\brief output the bounding box coordinates of the '1's in a DREG to current output port

	\param dreg 		DREG
	\param display 		handle of a gui display
	\param vec4v 		result will also be stored in this array (optional)

*/
void scamp5_output_boundingbox(DREG_SRC dr, vs_handle display, uint8_t*result = NULL);


/*!
	\brief display the bounding box coordinates in a GUI display

	\param display 		handle of a gui display
	\param vec4v 		result will also be stored in this array
	\param count 		number of bounding box

	This function is useful when used in combination with ::scamp5_scan_boundingbox so that
	any process can be applied on the result before output.
*/
void scamp5_display_boundingbox(vs_handle display, const uint8_t*vec4v, size_t count);


/*!
	\brief display a image

	\param display 		handle of a gui display
	\param buffer 		pointer to the column-major pixel buffer
	\param w 			width of the image
	\param h 			height of the image

	This function is useful when used in combination with scanning functions such as ::scamp5_scan_areg
	to display the result. Note, the image is stored in column major format, starting from top right.
	i.e. "buffer[3]" is the pixel the on 1st column right, 4th row down.
*/
void scamp5_display_image(vs_handle display, const uint8_t*buffer, uint8_t w, uint8_t h);


/*!
	\brief display a image as a patch over existing display

	\param display 		handle of a gui display
	\param buffer 		pointer to the column-major pixel buffer
	\param w 			width of the image
	\param h 			height of the image
	\param r0 			row position of the patch
	\param c0 			column position of the patch

	This function will apply the image over the previous image in the display as a patch.
*/
void scamp5_display_image_patch(vs_handle display, const uint8_t*buffer, uint8_t w, uint8_t h, uint8_t r0, uint8_t c0);


/*!
	\brief begin to output a bit-stack.

	\param display 		handle of a gui display
	\param n_bits 		number of bits in each pixel of the grey-scale image

	A bit-stack is a grey-scale image composed of several DREG images
	where each DREG	image provide one bit in the grey level	in an order of
	most significant bit to least significant bit.

*/
void scamp5_output_bitstack_begin(vs_handle display, int n_bits);


/*!
	\brief output a DREG image as a bit in a bit-stack

	\param dreg 		DREG

*/
void scamp5_output_bitstack_bit(DREG_SRC dreg);


/*!
	\brief finish outputting a bit-stack

*/
void scamp5_output_bitstack_end();


/*!

	@brief make the host gui save all incoming data for one loop iteration

*/
void scamp5_save_snapshots();


/*--------------------------------------------------------------------------------*/


#ifdef S5D_M0_PRIVATE


struct scamp5_data_overhead_t {
	vs_packet_header_t header;
	vs_data_header_t def;
};


extern vs_protocol_out scamp5_protocol;
extern vs_io_agent_api*scamp5_io;
extern uint8_t scamp5_adc_delay_0;
extern uint8_t scamp5_adc_delay_1;
extern uint8_t scamp5_adc_delay_2;
extern uint8_t scamp5_adc_delay_scan_0;
extern uint8_t scamp5_adc_delay_scan_1;
extern uint8_t scamp5_adc_delay_scan_2;
extern int32_t scamp5_mean_8x8_ca[2];

void scamp5_send_aout_overhead(void*buffer, int n_rows, int n_cols, uint32_t channel);
void scamp5_send_dout_overhead(void*buffer, int n_rows, int n_cols, uint32_t channel);

#define scamp5_io_buffer	scamp5_io->get_io_buffer()
#define scamp5_io_agent	    scamp5_io
#define scamp5_packet_header_encode(u,v) 	scamp5_protocol.encode_header(u,VS_PACKET_TYPE_DATA,v)


#endif

#endif
