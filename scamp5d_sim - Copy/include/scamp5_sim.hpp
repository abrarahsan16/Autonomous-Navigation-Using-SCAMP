/*!

\file

\ingroup VS_SCAMP5_MODULE

\author Jianing Chen

*/

#ifndef VS_SIM_HPP
#define VS_SIM_HPP

#include <vs_scamp5.hpp>


#define scamp5_sim	vs_sim

/*!
	all functions under this namespace code are only effective in simulation
*/
namespace vs_sim {


/*!
	\brief check if the program is a simulation

*/
bool is_simulation();


/*!
	\brief (only in simulation) configure the simulation client

	\param field 		field
	\param value 		options

	|     Field     |       Description       |
	|---------------|-------------------------|
	| "server_ip"   | simulation server IP    |
	| "server_port" | simulation server Port  |

	In simulation, this function must be called before ::vs_init to take effect.

*/
void config(const char*field, const char*value);


/*!
	\brief (only in simulation) enable some keyboard control of the simulation client

*/
void enable_keyboard_control();


/*!
	\brief (only in simulation) reset the simulation model on the server and modify error model level

*/
void reset_model(uint32_t level);


/*!
	\brief (only in simulation) manually modify the light integration level on PIX

*/
void pix_integrate(uint16_t increment_num, uint16_t increment_den);


/*!
	\brief (only in simulation) save the image of a AREG as a BMP file

*/
void save_image(AREG_SRC areg, const char*filepath);


/*!
	\brief (only in simulation) save the image of a DREG as a BMP file

*/
void save_image(DREG_SRC dreg, const char*filepath);

};


#endif
