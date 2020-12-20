/*!

\file

\ingroup VS_OBJECT

\author Jianing Chen

*/

#ifndef VS_RINGBUFFER_HPP
#define VS_RINGBUFFER_HPP

#include <cstdint>

/*!
A container class template. It stores elements in a native array of size \c N.
When the capacity is full, new data will overwrite old data.
It supports auto 'for' loop, where the loop always starts from the oldest data.
*/
template <typename _idx_t_,typename _dat_t_,size_t N>
class vs_ringbuffer{

protected:

	_idx_t_ front_idx;
	_idx_t_ back_idx;
    _idx_t_ length;
	_dat_t_ buffer[N];


public:

    vs_ringbuffer(){
        front_idx = 0;
        back_idx = 0;
        length = 0;
    }

    inline void clear(){
    	front_idx = 0;
    	back_idx = 0;
        length = 0;
    }

    inline bool is_empty(){
        return length==0;
    }

    inline bool is_full(){
        return length==N;
    }

    inline size_t get_length(){
    	return length;
    }

    inline size_t get_capacity(){
    	return N;
    }

    inline _dat_t_& front(){
        return buffer[front_idx];
    }

    inline void push_front(){
        if(length){
            front_idx += 1;
            front_idx %= N;
            if(front_idx == back_idx){
                back_idx += 1;
                back_idx %= N;
            }else{
                length += 1;
            }
        }else{
            length = 1;
        }
    }

    inline _dat_t_& back(){
        return buffer[back_idx];
    }

    inline void pop_back(){
        if(length){
            length -= 1;
            if(front_idx!=back_idx){
                back_idx += 1;
                back_idx %= N;
            }
        }
    }

    class iterator{

    protected:
        _dat_t_ *buffer;
        _idx_t_ i;
        _idx_t_ n;

    public:
        iterator(_idx_t_ init,_dat_t_*p,_idx_t_ length){
            buffer = p;
            i = init;
            n = length;
        }

        bool operator!=(iterator const& arg) const {
            return n!=0;
        }

        _dat_t_& operator++(){
            i += 1;
            i %= N;
            n -= 1;
            return buffer[i];
        }

        const _dat_t_& operator*() const {
            return buffer[i];
        }
    };

    inline iterator begin(){
        return iterator(back_idx,buffer,get_length());
    }

    inline iterator end(){
        return iterator(back_idx,buffer,0);
    }

    inline iterator begin() const{
        return iterator(back_idx,buffer,get_length());
    }

    inline iterator end() const{
        return iterator(back_idx,buffer,0);
    }

};

#endif
