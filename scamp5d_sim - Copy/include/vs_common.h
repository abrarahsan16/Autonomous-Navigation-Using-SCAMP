
#ifndef VS_COMMON_H
#define VS_COMMON_H

#ifdef __cplusplus
extern "C"{
#endif

#include <stdlib.h>
#include <stdint.h>

#ifdef __GNUC__
#define PRINTF_ARG __attribute__ ((format(printf,1,2)))
#define SNPRINTF_ARG __attribute__ ((format(printf,3,4)))
#else
#define PRINTF_ARG  
#define SNPRINTF_ARG  
#endif

#ifdef __LPC43XX__
#define __ASM            __asm
#define __INLINE         inline
#define __STATIC_INLINE  static inline
#include <core_cmInstr.h>
#define S5D_INLINE 		INLINE
#define S5D_IPC_IRQ		__SEV
#endif

typedef uint8_t vs_bool;
typedef uint32_t vs_handle;

/** general purpose time type, unit = 1/1000 second */
typedef uint32_t vs_time;


#define VS_TIME_Q   		1000
#define VS_MASTERCLOCK_Q   144000UL
/** master clock time type, unit = 1/144000 second */
typedef uint64_t vs_mct64;
typedef uint32_t vs_mct32;

#define VS_TIME_TO_MCT32(t) ((vs_mct32)(VS_MASTERCLOCK_Q/VS_TIME_Q)*(t))
#define VS_TIME_TO_MCT64(t) ((vs_mct64)(VS_MASTERCLOCK_Q/VS_TIME_Q)*(t))


/*----------------------------------------------------------------------------*/


#ifdef VS_LIB_PRIVATE

#ifdef __LPC43XX__
#include <chip.h>
#endif

#define CLOCK_M4        	204000000UL
#define M0APP_MEM_BASE  	0x10080000
#define AHB_MEM_BASE  		0x20000000

#define VS_HANDLE_TO_POINTER(h) ((void*)h)
#define VS_POINTER_TO_HANDLE(p) ((vs_handle)p)

#endif


#ifdef __cplusplus
}
#endif

#endif
