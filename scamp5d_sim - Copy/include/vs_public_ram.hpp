/*!

\file

\ingroup VS_SHARED_MODULE

\author Jianing Chen

*/

#ifndef VS_PUBLIC_RAM_HPP
#define VS_PUBLIC_RAM_HPP


#include "vs_mutex.hpp"


#define S5D_PUBLIC_RAM_ADDRESS      0x20000000
#define S5D_PUBLIC_RAM_SIZE     	65536
#define S5D_PUBLIC_RAM_CELL_SIZE    512
#define S5D_PUBLIC_RAM_CELL_NUM     (S5D_PUBLIC_RAM_SIZE/S5D_PUBLIC_RAM_CELL_SIZE)


/*!
An API class used to allocate a stand-alone RAM Block shared by M4/M0 core.

Note: dont create instances of this class, only use the existing global pointer \ref vs_prm.
*/
class vs_public_ram_api:protected vs_mutex{

public:

	/*!
	@brief allocate memory from public RAM

	The allocation has a minimum size of 512 bytes, so it is not recommended
	to allocate small memory blocks using public RAM.
	*/
	void*malloc(size_t bytes);

	/*!
	@brief free a allocated memory from public RAM
	*/
	void free(void*ptr);


protected:

	vs_public_ram_api(){}
	~vs_public_ram_api(){}

    int32_t error_code;
    uint16_t cell_owner[S5D_PUBLIC_RAM_CELL_NUM];
    uint16_t cell_now;

};


extern vs_public_ram_api* vs_prm;//!< all membership functions of ::vs_public_ram_api should be called through this pointer


/*
--------------------------------------------------------------------------------
*/

#ifdef S5D_M4_PRIVATE

void vs_init_public_ram();

#endif

#endif
