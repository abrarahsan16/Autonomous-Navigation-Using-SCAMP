/*!

\file

\ingroup VS_OBJECT

\author Jianing Chen

*/

#ifndef VS_RANDOM_HPP
#define VS_RANDOM_HPP


#include <cstdint>
#include <cstdlib>


/*!
A random number generator with full integer implementation.
*/
class vs_random{

protected:

    uint32_t m_seed;
    uint32_t m_w;// Marsaglia's MWC Algorithm
    uint32_t m_z;

    inline void roll(){
        m_z = 36969*(m_z&65535) + (m_z>>16);
        m_w = 18000*(m_w&65535) + (m_w>>16);
    }


public:

    vs_random(){
        m_seed = 0;
        m_w = 521288629UL;
        m_z = 362436069UL;
    }

	/*!
		@brief constructor with a given seed
	*/
    vs_random(uint32_t seed){
        m_seed = seed;
        m_w = 521288629 + m_seed;
        m_z = 362436069 - m_seed;
    }

    /*!
    	@brief reset the random sequence using the current seed
    */
    void reset(){
        m_w = 521288629UL + m_seed;
        m_z = 362436069UL - m_seed;
    }

    /*!
    	@brief assign a new seed and reset the random sequence
    */
    void reseed(uint32_t seed){
        m_seed = seed;
        m_w = 521288629UL + m_seed;
        m_z = 362436069UL - m_seed;
    }

    /*!
    	@brief get the current seed
    */
    inline uint32_t get_seed(){
        return m_seed;
    }

    /*!
    	@brief get a random \c uint32_t
    */
    inline uint32_t get_uint32(){
        roll();
        return (m_z<<16) + (m_w&0xFFFF);
    }

    /*!
    	@brief get a random \c uint16_t
    */
    inline uint16_t get_uint16(){
        roll();
        return (m_z<<8) + (m_w&0xFF);
    }

    /*!
    	@brief get a random \c int in the range of [ \c r_min , \c r_max ]
    	@param r_min minimal value possible
    	@param r_max maximal value possible
		@return the result
    */
    inline int get_int(int r_min,int r_max){
        uint32_t s = r_max - r_min + 1;
        uint32_t r = get_uint16();
        return r*s/65536UL + r_min;
    }

    /*!
    	@brief return \c TRUE with a chance of \c num/den
    */
    inline bool chance(uint16_t num,uint16_t den){
        uint32_t r = this->get_uint16();
        return (num*r)<(den*65536UL);
    }

    /*!
    	@brief same as ::vs_random::get_int but using \c operator()
    */
    inline int operator()(int r_min,int r_max){
        return get_int(r_min,r_max);
    }

};

#endif
