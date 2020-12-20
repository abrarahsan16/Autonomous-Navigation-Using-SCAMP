/*!

\file

\ingroup VS_OBJECT

\author Jianing Chen

*/

#ifndef VS_MUTEX_HPP
#define VS_MUTEX_HPP

#include <vs_common.h>


#ifdef __LPC43XX__
class vs_mutex {


protected:

	volatile uint8_t m4_flag;
	volatile uint8_t m0_flag;
	volatile uint16_t turn;


public:

	vs_mutex() {
		m4_flag = 0;
		m0_flag = 0;
		turn = 0;
	}

#ifdef CORE_M4
	inline bool is_locked() {
		return m4_flag == 1;
	}

	inline void lock() {
		m4_flag = 1;
		turn = 1;
		while (m0_flag && turn == 1) {
			__NOP();
		}
		__ISB();
	}

	inline void unlock() {
		m4_flag = 0;
		__ISB();
	}
#endif

#ifdef CORE_M0
	inline bool is_locked() {
		return m0_flag == 1;
	}

	inline void lock() {
		m0_flag = 1;
		turn = 0;
		while (m4_flag && turn == 0) {
			__NOP();
		}
		__ISB();
	}

	inline void unlock() {
		m0_flag = 0;
		__ISB();
	}
#endif

};

#else

#include <atomic>
#include <mutex>

class vs_mutex {

protected:
	std::mutex m;

public:

	vs_mutex() {
	}

	inline void lock() {
		m.lock();
	}

	inline void unlock() {
		m.unlock();
	}

};

#endif

#endif
