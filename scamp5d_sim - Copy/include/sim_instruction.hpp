
#ifndef SCAMP5_ICW_HPP
#define SCAMP5_ICW_HPP

#include <cstdint>
#include <initializer_list>
#undef IN

struct aref_t{
    const uint16_t index;
    const char*name;
//    operator const uint16_t() const{
//        return index;
//    }
};


struct areg_t{
	const uint16_t index;
    const char*name;
    operator const aref_t&() const{
        return *((const aref_t*)this);
    }
//    operator const uint16_t() const{
//        return index;
//    }
};


struct dref_t{
	const uint16_t index;
    const char*name;
//    operator const uint16_t() const{
//        return index;
//    }
};


struct dreg_t{
	const uint16_t index;
    const char*name;
    operator const dref_t&() const{
        return *((const dref_t*)this);
    }
//    operator const uint16_t() const{
//        return index;
//    }
};


enum news_t { north = 1, east = 2, south = 4, west = 8, alldir = 15 };


namespace SCAMP5_PE{

    const aref_t Z =    { 0, ".Z" };
    const aref_t PIX =  { 1, "PIX" };
    const aref_t IN =   { 2, "IN" };
    const aref_t RES =  { 3, ".RES" };

    const aref_t NORTH = { 4, "NORTH" };
    const aref_t EAST =  { 5, "EAST" };
    const aref_t SOUTH = { 6, "SOUTH" };
    const aref_t WEST =  { 7, "WEST" };

	const areg_t XN = { 4, "XN" };
	const areg_t XE = { 5, "XE" };
	const areg_t XS = { 6, "XS" };
	const areg_t XW = { 7, "XW" };

    const areg_t SQ =  { 8, ".SQ" };

    const areg_t NEWS = { 9, "NEWS" };
    const areg_t A =    { 10, "A" };
    const areg_t B =    { 11, "B" };
    const areg_t C =    { 12, "C" };
    const areg_t D =    { 13, "D" };
    const areg_t E =    { 14, "E" };
    const areg_t F =    { 15, "F" };

    const areg_t Q[32] = {
        { 16, ".Q[0]" },
        { 17, ".Q[1]" },
        { 18, ".Q[2]" },
        { 19, ".Q[3]" },
        { 20, ".Q[4]" },
        { 21, ".Q[5]" },
        { 22, ".Q[6]" },
        { 23, ".Q[7]" },
        { 24, ".Q[8]" },
        { 25, ".Q[9]" },
        { 26, ".Q[10]" },
        { 27, ".Q[11]" },
        { 28, ".Q[12]" },
        { 29, ".Q[13]" },
        { 30, ".Q[14]" },
        { 31, ".Q[15]" },
        { 32, ".Q[16]" },
        { 33, ".Q[17]" },
        { 34, ".Q[18]" },
        { 35, ".Q[19]" },
        { 36, ".Q[20]" },
        { 37, ".Q[21]" },
        { 38, ".Q[22]" },
        { 39, ".Q[23]" },
        { 40, ".Q[24]" },
        { 41, ".Q[25]" },
        { 42, ".Q[26]" },
        { 43, ".Q[27]" },
        { 44, ".Q[28]" },
        { 45, ".Q[29]" },
        { 46, ".Q[30]" },
        { 47, ".Q[31]" }
    };


    const dreg_t RB = { 0, ".RB" };

    const dref_t FLAG = { 1, "FLAG" };
    const dref_t SELECT = { 2, "SELECT" };

    const dreg_t R0 =   { 3, "R0" };
    const dreg_t R1 =   { 4, "R1" };
    const dreg_t R2 =   { 5, "R2" };
    const dreg_t R3 =   { 6, "R3" };
    const dreg_t R4 =   { 7, "R4" };
    const dreg_t R5 =   { 8, "R5" };
    const dreg_t R6 =   { 9, "R6" };
    const dreg_t R7 =   { 10, "R7" };
    const dreg_t R8 =   { 11, "R8" };
    const dreg_t R9 =   { 12, "R9" };
    const dreg_t R10 =  { 13, "R10" };
    const dreg_t R11 =  { 14, "R11" };
    const dreg_t R12 =  { 15, "R12" };

    const dreg_t S[128] = {
        { 16, ".S[0]" },
        { 17, ".S[1]" },
        { 18, ".S[2]" },
        { 19, ".S[3]" },
        { 20, ".S[4]" },
        { 21, ".S[5]" },
        { 22, ".S[6]" },
        { 23, ".S[7]" },
        { 24, ".S[8]" },
        { 25, ".S[9]" },
        { 26, ".S[10]" },
        { 27, ".S[11]" },
        { 28, ".S[12]" },
        { 29, ".S[13]" },
        { 30, ".S[14]" },
        { 31, ".S[15]" },
        { 32, ".S[16]" },
        { 33, ".S[17]" },
        { 34, ".S[18]" },
        { 35, ".S[19]" },
        { 36, ".S[20]" },
        { 37, ".S[21]" },
        { 38, ".S[22]" },
        { 39, ".S[23]" },
        { 40, ".S[24]" },
        { 41, ".S[25]" },
        { 42, ".S[26]" },
        { 43, ".S[27]" },
        { 44, ".S[28]" },
        { 45, ".S[29]" },
        { 46, ".S[30]" },
        { 47, ".S[31]" },
        { 48, ".S[32]" },
        { 49, ".S[33]" },
        { 50, ".S[34]" },
        { 51, ".S[35]" },
        { 52, ".S[36]" },
        { 53, ".S[37]" },
        { 54, ".S[38]" },
        { 55, ".S[39]" },
        { 56, ".S[40]" },
        { 57, ".S[41]" },
        { 58, ".S[42]" },
        { 59, ".S[43]" },
        { 60, ".S[44]" },
        { 61, ".S[45]" },
        { 62, ".S[46]" },
        { 63, ".S[47]" },
        { 64, ".S[48]" },
        { 65, ".S[49]" },
        { 66, ".S[50]" },
        { 67, ".S[51]" },
        { 68, ".S[52]" },
        { 69, ".S[53]" },
        { 70, ".S[54]" },
        { 71, ".S[55]" },
        { 72, ".S[56]" },
        { 73, ".S[57]" },
        { 74, ".S[58]" },
        { 75, ".S[59]" },
        { 76, ".S[60]" },
        { 77, ".S[61]" },
        { 78, ".S[62]" },
        { 79, ".S[63]" },
        { 80, ".S[64]" },
        { 81, ".S[65]" },
        { 82, ".S[66]" },
        { 83, ".S[67]" },
        { 84, ".S[68]" },
        { 85, ".S[69]" },
        { 86, ".S[70]" },
        { 87, ".S[71]" },
        { 88, ".S[72]" },
        { 89, ".S[73]" },
        { 90, ".S[74]" },
        { 91, ".S[75]" },
        { 92, ".S[76]" },
        { 93, ".S[77]" },
        { 94, ".S[78]" },
        { 95, ".S[79]" },
        { 96, ".S[80]" },
        { 97, ".S[81]" },
        { 98, ".S[82]" },
        { 99, ".S[83]" },
        { 100, ".S[84]" },
        { 101, ".S[85]" },
        { 102, ".S[86]" },
        { 103, ".S[87]" },
        { 104, ".S[88]" },
        { 105, ".S[89]" },
        { 106, ".S[90]" },
        { 107, ".S[91]" },
        { 108, ".S[92]" },
        { 109, ".S[93]" },
        { 110, ".S[94]" },
        { 111, ".S[95]" },
        { 112, ".S[96]" },
        { 113, ".S[97]" },
        { 114, ".S[98]" },
        { 115, ".S[99]" },
        { 116, ".S[100]" },
        { 117, ".S[101]" },
        { 118, ".S[102]" },
        { 119, ".S[103]" },
        { 120, ".S[104]" },
        { 121, ".S[105]" },
        { 122, ".S[106]" },
        { 123, ".S[107]" },
        { 124, ".S[108]" },
        { 125, ".S[109]" },
        { 126, ".S[110]" },
        { 127, ".S[111]" },
        { 128, ".S[112]" },
        { 129, ".S[113]" },
        { 130, ".S[114]" },
        { 131, ".S[115]" },
        { 132, ".S[116]" },
        { 133, ".S[117]" },
        { 134, ".S[118]" },
        { 135, ".S[119]" },
        { 136, ".S[120]" },
        { 137, ".S[121]" },
        { 138, ".S[122]" },
        { 139, ".S[123]" },
        { 140, ".S[124]" },
        { 141, ".S[125]" },
        { 142, ".S[126]" },
        { 143, ".S[127]" }
    };

	const dref_t RECT = { 144, "RECT" };
};

#define RS 	R1
#define RW 	R2
#define RN 	R3
#define RE 	R4
#define S0	R5
#define S1	R6
#define S2	R7
#define S3	R8
#define S4	R9
#define S5	R10
#define S6	R11
#define RP 	R12
#define RF 	R0

#define AREG_DST areg_t const&
#define AREG_SRC aref_t const&
#define DREG_DST dreg_t const&
#define DREG_SRC dref_t const&

#define AREG areg_t const&
#define ASRC aref_t const&
#define DREG dreg_t const&
#define DSRC dref_t const&

#define SCAMP5_MACRO_T_REG  SCAMP5_PE::NEWS
#define SCAMP5_MACRO_T_DREG  SCAMP5_PE::R0


class sim_instruction{

public:
    int16_t opcode;
    int16_t flag;
    int16_t arg[8];

    inline uint32_t& arg_uint32(size_t i){
        return i[(uint32_t*)arg];
    }

};


class scamp5_icw:public sim_instruction{

protected:

    void check_operand(int length);

public:

    void rpix();

    void nop();

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

    void where( const aref_t& a );

    void where( const aref_t& a0, const aref_t& a1 );

    void where( const aref_t& a0, const aref_t& a1, const aref_t& a2 );

    void all();

    void WHERE( const dref_t& d );

    void WHERE( const dref_t& d0, const dref_t& d1 );

    void WHERE( const dref_t& d0, const dref_t& d1, const dref_t& d2 );

    void SET( const dreg_t& d );

    void SET( const dreg_t& d0, const dreg_t& d1 );

    void SET( const dreg_t& d0, const dreg_t& d1, const dreg_t& d2 );

    void SET( const dreg_t& d0, const dreg_t& d1, const dreg_t& d2, const dreg_t& d3 );

    void CLR( const dreg_t& d );

    void CLR( const dreg_t& d0, const dreg_t& d1 );

    void CLR( const dreg_t& d0, const dreg_t& d1, const dreg_t& d2 );

    void CLR( const dreg_t& d0, const dreg_t& d1, const dreg_t& d2, const dreg_t& d3 );

    void MOV( const dreg_t& d, const bool b );

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

    void PROP_R();

    void PROP_0();

    void PROP_1();

};


#endif
