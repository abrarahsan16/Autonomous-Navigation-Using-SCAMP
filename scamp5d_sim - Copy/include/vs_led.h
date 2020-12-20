/*!

\file

\ingroup VS_COMMON_MODULE

\author Jianing Chen

*/

#ifndef VS_LED_H
#define VS_LED_H

#include <vs_common.h>


#define VS_LED_0   (1<<8)
#define VS_LED_1   (1<<9)
#define VS_LED_2   (1<<11)


/*!
    \brief turn on an LED
*/
void vs_led_on(uint32_t bits);


/*!
    \brief turn off an LED
*/
void vs_led_off(uint32_t bits);


/*!
    \brief toggle on/off an LED
*/
void vs_led_toggle(uint32_t bits);


#endif
