/*!

\file

\ingroup VS_SHARED_MODULE

\author Jianing Chen

*/

#ifndef VS_FLASH_AGENT_HPP
#define VS_FLASH_AGENT_HPP


#include <functional>
#include "vs_mutex.hpp"


class vs_flash_agent_api:protected vs_mutex{

public:

    int erase_all();
    int erase_block(uint32_t addr);
    int write(uint32_t addr,const void*data,size_t bytes);
    int read(uint32_t addr,void*buffer,size_t bytes);

    inline bool is_busy(){
    	return state!=10;
    }

    inline uint32_t get_page_size(uint32_t addr){
    	return page_size;
    }

    inline uint32_t get_block_size(uint32_t addr){
    	return block_size[addr>=block_division];
    }

    inline uint32_t get_plane_size(uint32_t addr){
    	return plane_size;
    }

    inline uint32_t get_error(){
        return error_code;
    }


protected:

	vs_flash_agent_api(){}
    ~vs_flash_agent_api(){}

    volatile int32_t error_code;
    volatile int32_t state;
    volatile uint32_t arg_flash_addr;
    volatile uint32_t arg_local_addr;
    volatile uint32_t arg_size;


    uint32_t page_size;
    uint32_t block_division;
    uint32_t block_size[2];
    uint32_t plane_size;

};


extern vs_flash_agent_api* vs_qflash;


/*
--------------------------------------------------------------------------------
*/


#ifdef VS_LIB_PRIVATE

// nothing

#endif


#ifdef S5D_M4_PRIVATE

void vs_init_flash_agent(void);
void vs_routine_flash_agent(void);

#endif


#endif
