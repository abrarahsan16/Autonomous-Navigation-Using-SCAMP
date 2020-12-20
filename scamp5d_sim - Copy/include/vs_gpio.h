/*!

\file

\ingroup VS_COMMON_MODULE

\author Jianing Chen

*/

#ifndef VS_GPIO_H
#define VS_GPIO_H

#include <vs_common.h>


#define VS_GPIO_X0	23
#define VS_GPIO_X1	27
#define VS_GPIO_X2	28
#define VS_GPIO_X3	29
#define VS_GPIO_X4	30


/*!

	\brief configure a pin in GPIO6 as an input pin

    \param pin		pin number
    \param pu		1 for pull-up, 0 for pull-down

*/
void vs_gpio_as_input(uint8_t pin,int pu);


/*!

	\brief configure a pin in GPIO6 as an output pin

    \param pin		pin number

*/
void vs_gpio_as_output(uint8_t pin);


/*!

	\brief get the state an input GPIO pin

    \param pin		pin number

*/
int vs_gpio_input(uint8_t pin);


/*!

	\brief set the state of an output GPIO pin

    \param pin		pin number

*/
void vs_gpio_output(uint8_t pin,int output);


#endif
