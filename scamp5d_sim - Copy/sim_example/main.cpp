/*
* Scamp5d M0 Example - Simulation
*
* Note: this source code works on both of the hardware and the simulation
*       but with an emphasis on demonstrating simulation-related functions.
*/




#include <scamp5.hpp>
#include "conv_instructions.hpp"

using namespace SCAMP5_PE;

volatile int threshold;

int main(){

    vs_sim::config("server_ip","127.0.0.1");
    vs_sim::config("server_port","27715");

    // Initialization
    vs_init();

    vs_sim::enable_keyboard_control();// this allow a few shortcuts to be used. e.g. 'Q' to quit.
	vs_sim::reset_model(3);// reset model is also used to configure the error model


    // Setup Host GUI
	vs_gui_set_info(VS_M0_PROJECT_INFO_STRING);


	vs_on_host_connect([&]() {
		vs_post_text("test TEST!\n");
	});

	vs_on_host_disconnect([&]() {
		vs_led_off(VS_LED_2);
	});


    auto display_a = vs_gui_add_display("Register A",0,0);
    auto display_b = vs_gui_add_display("Register B",0,1);

			//auto original_input = vs_gui_add_display("Original Input", 2, 2);

    auto slider_threshold = vs_gui_add_slider("threshold: ",-100,100,60,&threshold);



    // Frame Loop
    while(1){
		vs_frame_loop_control();

	//	scamp5_in(E, threshold);		// let E = threshold value
       scamp5_kernel_begin();
        //    get_image(C,D);
			//sub(A, C, E);
			//where(A);
			//MOV(R3,FLAG);
			//all();
	   
	    get_image(A);                     // get image and reset PIX

		div(D, E, A);
		diva(D, E, C);
		diva(D, E, C);
		mov2x(E, D, west, north);
		mov2x(C, D, west, north);
		movx(F, D, north);
		neg(A, D);
		sub(B, A, D);
		add(C, E, C);
		add2x(A, A, B, north, east);
		subx(D, D, east, F);
		addx(B, E, C, south);
		add2x(E, E, F, south, south);
		add(C, C, E, D);
		add(A, C, A, B);

/////////////////////////////////////
		//mov(B, A);
		//mov(C, A);
/*
	    mov(A, A, west);
		diva(A, E, F);
		diva(A, E, F);
		mov(D, A, south);
		mov(D, D, east);
		mov(E, A, north);
		mov(F, E);
		add(E, E, F);
		sub(A, A, E);
		mov(E, D, west);
		sub(A, A, E);
		mov(E, D, north);
		mov(E, E, north);
		sub(D, D, E);
		add(A, A, D);
		mov(D, D, east);
		mov(F, D);
		add(D, D, F);
		add(A, A, D);
*/

		// lower case: Ainstruction Upper case£ºDinstruction
		
		// API and functions
		// https://personalpages.manchester.ac.uk/staff/jianing.chen/scamp5d_lib_doc_html/scamp5__kernel__api__macro__analog_8hpp.html#a098bb4096dec2f54c4d1643b6a5873a2 
		// https://personalpages.manchester.ac.uk/staff/jianing.chen/scamp5d_lib_doc_html/scamp5__kernel__api_8hpp.html#ac04c85ac270c664a58ff526f129c3df5

		//   last try on the RELU
		// have to use where+MOV 
		mov(E, A);
		where(E);    //FLAG := ASRC > 0.
		MOV(R8, FLAG);
		if (R8 == 0) {
			bus(E);
		}

		
		MOV(R9, FLAG);
		MOV(R10, FLAG);

		all();      //FLAG := 1
        scamp5_kernel_end();


		if (vs_gui_is_on()) {
			scamp5_output_image(A, display_a);
			scamp5_output_image(R8, display_b);
		}

		if(vs_sim::is_simulation()){
			// these are not possible for the device, but can be used in simulation
			//vs_sim::save_image(C, "test_save_areg.BMP");
			//vs_sim::save_image(R5, "test_save_dreg.BMP");
		}
    }

    return 0;
}


