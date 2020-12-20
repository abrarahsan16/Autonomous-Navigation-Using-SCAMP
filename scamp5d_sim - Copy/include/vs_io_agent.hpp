/*!

\file

\ingroup VS_SHARED_MODULE

\author Jianing Chen

*/

#ifndef VS_IO_AGENT_HPP
#define VS_IO_AGENT_HPP


#include <functional>
#include "vs_mutex.hpp"

/*!
An API class used to access IOs of various buses using the same set of
functions and shared by both cores.

Note: dont create instances of this class, only use the existing global pointers for
each of the IO buses: \ref vs_uart,  \ref vs_usb, \ref vs_spi.
*/
class vs_io_agent_api :protected vs_mutex {

public:

	/*!
	@brief request to send a data buffer
	@param data 	pointer to the data buffer
	@param bytes 	number of bytes to send in the buffer

	The data buffer must retain valid till the send is finished.
	*/
	void send(const void*data, size_t bytes);

	/*!
	@brief check if last issued send request is done
	*/
	bool send_done();

	/*!
	@brief request receive data
	@param buffer 		pointer to a buffer
	@param buffer_size 	maximum number of bytes to receive
	@param min_size 	least number of bytes required for the 'receive' to be considered 'done'
	*/
	void receive(void*buffer, size_t buffer_size, size_t min_size = 1);

	/*!
	@brief check if last issued receive request is finished
	@param length 		stores the actual number of bytes received

	*/
	bool receive_done(size_t*length);

	/*!
	@brief bind a buffer for potential use

	The functions of IO Agent don't directly use this buffer. It is mainly designed to
	write down a free-to-use buffer for any potential usage in various place.
	*/
	inline void bind_io_buffer(void*mem, size_t s) {
		io_buffer = mem;
		io_buffer_size = s;
	}


	/*!
	@brief get the pointer to a buffer provided by the IO Agent
	*/
	inline void* get_io_buffer() {
		return io_buffer;
	}

	/*!
	@brief get the size of the buffer provided by the IO Agent
	*/
	inline size_t get_io_buffer_size() {
		return io_buffer_size;
	}

	/*!
	@brief get any existing error
	*/
	inline int32_t get_error() {
		return error_code;
	}

	/*!
	@brief try begin a session that consists multiple send request
	@return 1 when a session is successfully acquired; 0 otherwise
	*/
	inline int acquire_session() {
		int r;
		lock();
		if (session_lock) {
			r = 0;
		}
		else {
			session_lock = 1;
			r = 1;
		}
		unlock();
		return r;
	}

	/*!
	@brief end a session
	*/
	inline void free_session() {
		session_lock = 0;
	}

	/*!
	@brief check if the io agent is already occupied in a session
	*/
	inline bool is_session_free() {
		lock();
		bool b = session_lock == 0;
		unlock();
		return b;
	}


protected:

	vs_io_agent_api() {

	}
	~vs_io_agent_api() {

	}

	int16_t error_code;
	std::atomic<int16_t> session_lock;
	volatile uint32_t tx_addr;
	volatile uint32_t rx_addr;
	volatile uint16_t tx_size;
	volatile uint16_t rx_size;
	volatile uint16_t rx_min;
	volatile uint16_t tx_state;
	volatile uint16_t rx_state;
	void *io_buffer;
	size_t io_buffer_size;

};


extern vs_io_agent_api* vs_uart;//!< UART IO Agent functions should be called through this pointer
extern vs_io_agent_api* vs_usb;//!< USB IO Agent functions should be called through this pointer
extern vs_io_agent_api* vs_spi;//!< SPI IO Agent functions should be called through this pointer


void vs_on_send_done(vs_io_agent_api*a, std::function<void(vs_io_agent_api*)> cb);


/*
--------------------------------------------------------------------------------
*/


#ifdef VS_LIB_PRIVATE

extern std::function<void(vs_io_agent_api*)> vs_uart_tx_callback;
extern std::function<void(vs_io_agent_api*)> vs_usb_tx_callback;
extern std::function<void(vs_io_agent_api*)> vs_spi_tx_callback;

#endif


#ifdef S5D_M4_PRIVATE


class vs_io_agent_m4_base :public vs_io_agent_api {

public:
	vs_io_agent_m4_base();
	~vs_io_agent_m4_base();

};


void vs_init_io_agent(void);
void vs_routine_io_agent(void);

#endif


#endif
