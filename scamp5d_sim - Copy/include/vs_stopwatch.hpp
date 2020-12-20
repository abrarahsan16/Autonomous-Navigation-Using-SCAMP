/*!

\file

\ingroup VS_OBJECT

\author Jianing Chen

*/

#ifndef VS_STOPWATCH_HPP
#define VS_STOPWATCH_HPP


#include <cstdint>
#include <vs_common.h>


/*!
A class used to measure the time interval between two points in a program.
*/
class vs_stopwatch{

protected:

#ifdef __LPC43XX__
	uint32_t rct_start;
#else
    static uint64_t pc_freq;
    uint64_t pc_start;
#endif


public:

	vs_stopwatch();
    ~vs_stopwatch();

/*!
    \brief reset the stopwatch start point
*/
    void reset();

/*!
    \brief get microsecond elapsed since last reset
*/
    uint32_t get_usec() const;

};

#endif
