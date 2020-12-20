
#ifndef SCAMP5_KERNEL_API_HPP
#define SCAMP5_KERNEL_API_HPP


#include <functional>
#include <vs_scamp5.hpp>

class scamp5_kernel{

protected:
	static uint32_t total_sn;
	static scamp5_kernel* active_one;

    std::function<void(void)> f;

public:

	uint32_t serial_number;
	uint32_t icw_count;
	bool highspeed;
	bool move_to_heap;
	
    static void print_debug_info();

	scamp5_kernel();
	scamp5_kernel(std::function<void(void)> kernel_func);
	void operator()(std::function<void(void)> kernel_func);

	void begin();
	void end();

    friend void scamp5_launch_kernel(scamp5_kernel&o){
        o.f();
    };

};

#define scamp5_kernel_begin() { using namespace SCAMP5_PE; using namespace scamp5_kernel_api; static scamp5_kernel kn; kn.begin(); 
#define scamp5_kernel_end()   kn.end(); }

#define scamp5_dynamic_kernel_begin() { using namespace SCAMP5_PE; using namespace scamp5_kernel_api; scamp5_kernel kn; kn.begin(); 
#define scamp5_dynamic_kernel_end()   kn.end(); }

void scamp5_kernel_init();
void scamp5_kernel_all();
void scamp5_kernel_refresh_dreg();


namespace scamp5_kernel_api{

	void allow_highspeed(bool k);

	void move_to_heap(bool k);

    /* Single-instruction Operations */

    void nop();

    void rpix();

    void bus( const std::initializer_list<const areg_t>& dst, const std::initializer_list<const aref_t>& src );

    void bus( const areg_t& a );

    void bus( const areg_t& a, const aref_t& a0 );

    void bus( const areg_t& a, const aref_t& a0, const aref_t& a1 );

    void bus( const areg_t& a, const aref_t& a0, const aref_t& a1, const aref_t& a2 );

    void bus( const areg_t& a, const aref_t& a0, const aref_t& a1, const aref_t& a2, const aref_t& a3 );

    void bus2( const areg_t& a, const areg_t& b );

    void bus2( const areg_t& a, const areg_t& b, const aref_t& a0 );

    void bus2( const areg_t& a, const areg_t& b, const aref_t& a0, const aref_t& a1 );

    void bus3( const areg_t& a, const areg_t& b, const areg_t& c, const aref_t& a0 );

	void blurset();

    void blur( const areg_t& a, const aref_t& a0 );

    void blurh( const areg_t& a, const aref_t& a0 );

    void blurv( const areg_t& a, const aref_t& a0 );

    void sq( const aref_t& a0 );

    void sq( const aref_t& a0, const aref_t& a1 );

    void where( const aref_t& a );

    void where( const aref_t& a0, const aref_t& a1 );

    void where( const aref_t& a0, const aref_t& a1, const aref_t& a2 );

    void all();

    void WHERE( const dref_t& d );

    void WHERE( const dref_t& d0, const dref_t& d1 );

    void WHERE( const dref_t& d0, const dref_t& d1, const dref_t& d2 );

    void ALL();

    void SET( const dreg_t& d0 );

    void SET( const dreg_t& d0, const dreg_t& d1 );

    void SET( const dreg_t& d0, const dreg_t& d1, const dreg_t& d2 );

    void SET( const dreg_t& d0, const dreg_t& d1, const dreg_t& d2, const dreg_t& d3 );

    void CLR( const dreg_t& d0 );

    void CLR( const dreg_t& d0, const dreg_t& d1 );

    void CLR( const dreg_t& d0, const dreg_t& d1, const dreg_t& d2 );

    void CLR( const dreg_t& d0, const dreg_t& d1, const dreg_t& d2, const dreg_t& d3 );

    void MOV( const dreg_t& d, const dref_t& d0 );

    void OR( const dreg_t& d, const dref_t& d0, const dref_t& d1 );

    void OR( const dreg_t& d, const dref_t& d0, const dref_t& d1, const dref_t& d2 );

    void OR( const dreg_t& d, const dref_t& d0, const dref_t& d1, const dref_t& d2, const dref_t& d3 );

    void NOT( const dreg_t& d, const dref_t& d0 );

    void NOR( const dreg_t& d, const dref_t& d0, const dref_t& d1 );

    void NOR( const dreg_t& d, const dref_t& d0, const dref_t& d1, const dref_t& d2 );

    void NOR( const dreg_t& d, const dref_t& d0, const dref_t& d1, const dref_t& d2, const dref_t& d3 );

    void DNEWS0( const dreg_t& d, const dref_t& d0 );

    void DNEWS1( const dreg_t& d, const dref_t& d0 );

};

#endif
