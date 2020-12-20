/*!

\file

\ingroup VS_COMMON_MODULE

\author Jianing Chen

*/

#ifndef VS_POST_HPP
#define VS_POST_HPP


#include <vs_io_agent.hpp>
#include <vs_dotmat.hpp>
#include <vs_protocol.hpp>
#include <vs_message.hpp>


/*!
\brief assign which io agent to use when packet need to be sent

\param u 		pointer to an io agent

Example:

Sending "Hello World!\n" to through both SPI and USB:
\code
...
vs_post_bind_io_agent(vs_spi);
vs_post_text("Hello World!\n");

vs_post_bind_io_agent(vs_usb);
vs_post_text("Hello World!\n");
...
\endcode

*/
void vs_post_bind_io_agent(vs_io_agent_api*u);


/*!
\brief get the current io agent assignment

*/
vs_io_agent_api* vs_post_get_io_agent(void);


/*!
\brief set the channel number for the content

This number can be read at the receiving end to help distinguish packet.
*/
void vs_post_set_channel(uint32_t ch);


/*!
\brief send a message

Details about messages can be found in vs_message.hpp.
*/
void vs_post_message(uint8_t code0,uint8_t code1 = 0,uint32_t value = 0);


/*!
\brief similar to \c printf in stdio

There are a few commands which can be interpreted by the Host App to achieve some extra functionalities.

To clear the text stored in the console:
\code
vs_post_text("!clear");
\endcode

To save the text currently stored:
\code
vs_post_text("!save filename");
\endcode

Another example: prefix the filename with loop counter:
\code
vs_post_text("!save test_1_%d",vs_loop_counter_get());
\endcode

*/
void vs_post_text(const char*format,...) PRINTF_ARG;


/*!
\brief send an array of \c int8_t data

\param data 	pointer to the data array
\param rows 	number of rows
\param cols 	number of cols

If an 1-D array needs to be sent, \c rows should be set to 1.
*/
void vs_post_int8(const int8_t*data,size_t rows,size_t cols);


/*!
\brief send an array of \c int16_t data

\param data 	pointer to the data array
\param rows 	number of rows
\param cols 	number of cols

If an 1-D array needs to be sent, \c rows should be set to 1.
*/
void vs_post_int16(const int16_t*data,size_t rows,size_t cols);


/*!
\brief send an array of \c int32_t data

\param data 	pointer to the data array
\param rows 	number of rows
\param cols 	number of cols

If an 1-D array needs to be sent, \c rows should be set to 1.
*/
void vs_post_int32(const int32_t*data,size_t rows,size_t cols);


/*!
\brief send an array of \c float data

\param data 	pointer to the data array
\param rows 	number of rows
\param cols 	number of cols

If an 1-D array needs to be sent, \c rows should be set to 1.
*/
void vs_post_float(const float*data,size_t rows,size_t cols);


/*!
\brief send a dotmat image

\param u    the dotmat image object

In this case, the channel number will be used as display number to
show the image in Scamp5d Host.
*/
void vs_post_dotmat(vs_dotmat const& u);


/*----------------------------------------------------------------------------*/


#ifdef VS_LIB_PRIVATE

struct vs_data_packet_t{
    vs_packet_header_t header;
    vs_data_header_t def;
    uint8_t data[4096];// placeholder size
};

extern vs_protocol_out post_protocol;
extern uint32_t post_channel;

void vs_init_post();

#endif


#endif
