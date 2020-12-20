/*!

\file 

\ingroup VS_M0_CORE_MODULE

\author Jianing Chen

*/

#ifndef VS_INTERACTION_HPP
#define VS_INTERACTION_HPP

#include <array>
#include <functional>
#include <vs_import.hpp>
#include <vs_protocol_gui.hpp>


class vs_gui_script{

private:
    static uint16_t count;
    static std::array<const char*,20> name_ptr;
    static std::array<const char*,20> text_ptr;

protected:
    uint16_t index;

public:
    vs_gui_script(const char*name,const char*content);

    static void push_all();

    operator const char*() const {
        return name_ptr[index];
    }

};

#define VS_GUI_DISPLAY_STYLE(var_name,content)	static vs_gui_script var_name(#var_name,content);


/*!

@brief check whether the host application is connected

*/
bool vs_gui_is_on();


/*!

@brief set a description string for the program that can be seen in the recording folder

*/
void vs_gui_set_info(const char*str);


/*!

@brief clear GUI items

This will remove all items that has been added

*/
void vs_gui_clear_items();


/*!

@brief send the gui configuration to a connected host

When the host application is connected, this will be done once automatically.

*/
void vs_gui_push_items();


/*!

@brief add a display to the GUI configuration

@param text  name of the display
@param row   row position of this display
@param col   column position of this display
@param size  row/column span of the window
@param style style description

@return handle of the display

The 'style' argument can be either the file name of a style file or the file name of a palette file.
In the later case, the palette will be used as the palette for image.

More details about costmizing display style can be found in \ref PAGE_GUIDE_GUI.

*/
vs_handle vs_gui_add_display(const char*text,uint8_t row,uint8_t col,int size=1,const char*style=NULL);


/*!

@brief set the display style for a display

@param display 	handle of a created display
@param style 	style name

\p style can be either a display style's name or the filename of a palette.
*/
void vs_gui_set_display_style(vs_handle display,const char*style);


/*!
@brief setup a bar plot for a existing display to plot data array

@param h 	handle of the display
@param min	minimum value (plot floor)
@param max	maximum value (plot ceiling)
@param width width of each bar
@param opt 	other options

@return handle of the display

*/
vs_handle vs_gui_set_barplot(vs_handle h,int32_t min,int32_t max,uint16_t width,uint16_t opt=0);


/*!
@brief setup a scope for a existing display to plot scalar/vector data

@param h 	handle of the display
@param min	minimum value (plot floor)
@param max	maximum value (plot ceiling)
@param time length of the time window
@param opt 	other options

@return handle of the display

*/
vs_handle vs_gui_set_scope(vs_handle h,int32_t min,int32_t max,uint16_t time,uint16_t opt=0);


/*!

@brief add a button to the GUI configuration

@param text       text appeared on the button

@return handle of the button

*/
vs_handle vs_gui_add_button(const char*text);


/*!

@brief add a switch to the GUI configuration

@param text       text appeared on the switch

@return handle of the switch

*/
vs_handle vs_gui_add_switch(const char*text,bool init,volatile int*writeback = NULL);


/*!

@brief add a slider to the GUI configuration

@param text    text appeared on the slider
@param min     minimal value allowed
@param max     maximum value allowed
@param init    initial value
@param writeback (optional) a write-back variable pointer

@return handle of the slider

The write-back variable will be automatically updated when a slider value update is received.

More details about customizing display style can be found in \ref PAGE_GUIDE_GUI.

*/
vs_handle vs_gui_add_slider(const char*text,int32_t min,int32_t max,int32_t init,volatile int*writeback = NULL);


/*!

@brief add a latched slider to the GUI configuration

@param text    text appeared on the slider
@param min     minimal value allowed
@param max     maximum value allowed
@param init    initial value
@param writeback (optional) a write-back variable pointer

@return handle of the slider

A latched slider wont send its updated value until the sliding operation is finished.

*/
vs_handle vs_gui_add_slider_latched(const char*text,int32_t min,int32_t max,int32_t init,volatile int*writeback = NULL);


/*!

@brief get the cached value of a slider

@param h    handle of the slider

@return value of the slider

*/
int32_t vs_gui_read_slider(vs_handle h);


/*!

@brief change the value of a slider

@param h    	handle of the slider
@param value 	the value
@param feedback whether to execute any existing callback

This will change both the cached value on the device, and the slider on the host GUI.

*/
void vs_gui_move_slider(vs_handle h,int32_t value,bool feedback=true);


/*!

@brief change the value of a slider locally

@param h    	handle of the slider
@param value 	the value

This function will not change the slider value on the host.
It can be used in situation when no host is connected but parameters in the algorithm are all read from sliders.

*/
void vs_gui_set_slider(vs_handle h,int32_t value);


/*!

@brief request a file from the host computer

*/
int vs_gui_request_file(const char*filepath,std::function<void(const uint8_t*chunk,size_t bytes,size_t bytes_remain)> cb,uint32_t timeout_msec=1000);


/*!

@brief request a binary image from the host computer

@param filepath    	filepath relative to the host GUI
@param bits 		bits per pixel (highest bit first)
@param cb 			callback function
@return 			none zero if there is already a request undergoing

The callback function will be called when the image is received or when a timeout occurs.
It is necessary to keep frame loop running (or execute ::vs_process_message) to keep the polling system working on the M0 core.
If there is nothing else to do until the image arrive, a waiting loop could be used. For example:
\code
vs_gui_request_image("test.bmp",1,[&](vs_dotmat const& dotmat,int s){
if(s<0){
vs_post_text("image not received in time!\n");
return;
}
// ...
});
do{
vs_process_message();
}while(!vs_gui_request_done());
\endcode
*/
int vs_gui_request_image(const char*filepath,int bits,std::function<void(vs_dotmat const&,int)> cb,uint32_t timeout_msec=1000);


/*!
@brief get whether there is a ongoing request
*/
bool vs_gui_request_done();


/*!

@brief make the host gui save all incoming data for one loop iteration

*/
void vs_gui_save_snapshots();


/*!
@brief save the image in the display using a specific filepath
@param h 			display handle
@param filepath    	filepath relative to the host GUI

If the path contains folders, please make sure the folders exist. Otherwise the file won't get saved.
*/
void vs_gui_save_image(vs_handle h,const char*filepath);


/*!

@brief define the function to execute when the value of a GUI item is changed

@param gui_item   handle of the GUI item (slider/button)
@param f    the callback function

To get the update of the values from host gui items, use the following macro as	handle:

|      Macro Name       |         Description           |
|-----------------------|-------------------------------|
| VS_GUI_HOST_STATUS    | whether the host is connected	|
| VS_GUI_FRAME_RATE     | frame rate slider             |
| VS_GUI_FRAME_GAIN     | frame gain slider             |
| VS_GUI_IMAGE_LENGTH   | image data length slider      |
| VS_GUI_VECTOR_LENGTH  | vector data length slider     |
| VS_GUI_IMAGE_BLEND    | image display blend slider    |

*/
void vs_on_gui_update(vs_handle gui_item,std::function<void(int32_t)> f);


/*!

@brief define the function to execute when the host starts recording

@param f    the callback function

*/
void vs_on_gui_start_recording(std::function<void(void)> f);


/*!

@brief define the function to execute when the host stops recording

@param f    the callback function

*/
void vs_on_gui_stop_recording(std::function<void(void)> f);


/*!

@brief define the function to execute when the device is connected to a host

@param f    the callback function

*/
void vs_on_host_connect(std::function<void(void)> f);


/*!

@brief define the function to execute when the device is disconnected from a host

@param f    the callback function

*/
void vs_on_host_disconnect(std::function<void(void)> f);


/*!

@brief define the function to execute when a message carrying user value is received

@param f    the callback function

*/
void vs_on_user_value(std::function<void(uint8_t id,int32_t value)> f);


/*
--------------------------------------------------------------------------------
*/


#ifdef S5D_M0_PRIVATE

extern const char*vs_gui_program_info_str;
extern bool host_gui_on;


void vs_init_interaction();
void vs_routine_interaction();
int vs_gui_message_filter(const vs_message_t*);

inline vs_handle vs_gui_index_to_handle(uint32_t i){
    return (vs_handle)(i);
}

inline uint32_t vs_gui_handle_to_index(vs_handle h){
    return (uint32_t)(h);
}

inline vs_handle vs_gui_display_index_to_handle(uint32_t i){
    return (vs_handle)(i + 100);
}

inline uint32_t vs_gui_display_handle_to_index(vs_handle h){
    return (uint32_t)h - 100;
}

#endif

#endif
