/*!

\file

\ingroup VS_OBJECT

\author Jianing Chen

*/

#ifndef VS_DOTMAT_HPP
#define VS_DOTMAT_HPP


#include <vs_io_agent.hpp>
#include <cstring>
#include <algorithm>


/*!
A dot-matrix image class.
*/
class vs_dotmat{

public:

/*!
    \brief calculate the buffer size required for a dot-matrix image of the given dimension

    This can be used to get the memory size required for a dotmat of 'w' x 'h' size.
*/
	static size_t get_buffer_size(uint16_t w,uint16_t h){
		return w*h/8;
	}


	vs_dotmat(){
		dotmap = NULL;
		size_x = 0;
		size_y = 0;
	}

/*!
    \brief bind a buffer for drawing
*/
	inline void bind_buffer(void*mem,uint16_t w,uint16_t h){
		dotmap = (uint8_t*)mem;
		size_x = w;
		size_y = h;
	}

/*!
    \brief get the buffer currently binded and it's size
*/
	inline void* get_buffer(size_t*size=NULL)const{
		if(size){
			*size = size_x*size_y/8;
		}
		return dotmap;
	}

/*!
	\brief get height of the dot-matrix image
*/
	inline uint16_t get_height(void)const{
		return size_y;
	}

/*!
	\brief get width of the dot-matrix image
*/
	inline uint16_t get_width(void)const{
		return size_x;
	}

/*!
    \brief clear the image with either black or white
    \param w    1 for white, 0 for black
*/
	inline void clear(bool w=0){
		memset(dotmap,w?0xFF:0x00,size_x*size_y/8);
	}

/*!
    \brief draw a point, discard if the coordinates is outside the image
    \param x    x position
    \param y    y position
    \param w    1 for white, 0 for black
*/
	inline bool draw_point(int x,int y,bool w=1){
		if(x < 0){
			return false;
		}
		if(y < 0){
			return false;
		}
		if(x >= size_x){
			return false;
		}
		if(y >= size_y){
			return false;
		}
		if(w){
			dotmap[y*size_x/8 + x/8] |= (1<<(x%8));
		}else{
			dotmap[y*size_x/8 + x/8] &= ~(1<<(x%8));
		}
		return true;
	}

/*!
    \brief draw a point, wrap around if the coordinates is outside the image
    \param x    x position
    \param y    y position
    \param w    1 for white, 0 for black
*/
	inline void draw_pixel(uint16_t x,uint16_t y,bool w=1){
		x %= size_x;
		y %= size_y;
		if(w){
			dotmap[y*size_x/8 + x/8] |= (1<<(x%8));
		}else{
			dotmap[y*size_x/8 + x/8] &= ~(1<<(x%8));
		}
	}

#define draw_point_repeat	draw_pixel

/*!
    \brief draw a white line 
    \param r0    y position of the start point
    \param c0    x position of the start point
    \param r1    y position of the end point
    \param c1    x position of the end point
    \param repeat    whether to wrap around
    \param w     1 for white, 0 for black
*/
	void draw_line(int r0,int c0,int r1,int c1,const bool repeat=false,bool w=1){
		int x0 = c0;
		int y0 = r0;
		int x1 = c1;
		int y1 = r1;
		int dx = std::abs(x1 - x0);
		int dy = std::abs(y1 - y0);
		int	sx = (x0<x1) ? 1 : -1;
		int sy = (y0<y1) ? 1 : -1;
		int err = ((dx>dy) ? dx : -dy)/2;
		int e2;

		while(1){

			if(repeat){
				draw_pixel(x0,y0,w);
			}else{
				draw_point(x0,y0,w);
			}

			if(x0==x1&&y0==y1){
				break;
			}

			e2 = err;

			if(e2 > -dx){
				err -= dy;
				x0 += sx;
			}

			if(e2 < dy){
				err += dx;
				y0 += sy;
			}

		}

	}/* Reference: https://rosettacode.org/wiki/Bitmap/Bresenham%27s_line_algorithm#C */


/*!
    \brief draw a white circle
    \param x0    x position of the centeroid
    \param y0    y position of the centeroid
    \param radius    radius of the circle
    \param repeat    whether to wrap around
    \param w     1 for white, 0 for black
*/
	void draw_circle(int x0,int y0,int radius,const bool repeat=false,bool w=1){
	    int x = radius;
	    int y = 0;
	    int err = 0;

	    while (x >= y){

	    	if(repeat){
	    		draw_point_repeat(x0 + x, y0 + y,w);
	    		draw_point_repeat(x0 + y, y0 + x,w);
	    		draw_point_repeat(x0 - y, y0 + x,w);
	    		draw_point_repeat(x0 - x, y0 + y,w);
	    		draw_point_repeat(x0 - x, y0 - y,w);
	    		draw_point_repeat(x0 - y, y0 - x,w);
	    		draw_point_repeat(x0 + y, y0 - x,w);
	    		draw_point_repeat(x0 + x, y0 - y,w);
	    	}else{
	    		draw_point(x0 + x, y0 + y,w);
	    		draw_point(x0 + y, y0 + x,w);
	    		draw_point(x0 - y, y0 + x,w);
	    		draw_point(x0 - x, y0 + y,w);
	    		draw_point(x0 - x, y0 - y,w);
	    		draw_point(x0 - y, y0 - x,w);
	    		draw_point(x0 + y, y0 - x,w);
	    		draw_point(x0 + x, y0 - y,w);
	    	}

	        if(err <= 0){
	            y += 1;
	            err += 2*y + 1;
	        }else{
	            x -= 1;
	            err -= 2*x + 1;
	        }
	    }

	}/* Reference: https://en.wikipedia.org/wiki/Midpoint_circle_algorithm */


protected:

	uint8_t *dotmap;
	uint16_t size_x;
	uint16_t size_y;

};


#endif
