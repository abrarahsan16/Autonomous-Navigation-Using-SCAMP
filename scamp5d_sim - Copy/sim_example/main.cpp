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

        // First Conv layer
        neg(B, A);
        subx(C, A, south, B);
        subx(D, A, north, B);
        subx(B, B, west, A);
        movx(A, B, east);
        addx(C, C, D, east);
        addx(A, B, A, north);
        sub2x(B, A, south, south, A);
        sub2x(A, C, west, west, C);

        scamp5_in(C, threshold);
        scamp5_in(D, threshold);
        //
        // Second Conv layer Reg A
        div(E, C, A);
        diva(E, C, A);
        div(C, A, D, E);
        sub2x(A, C, east, south, C);
        movx(D, E, east);
        addx(C, C, E, east);
        mov2x(E, E, west, north);
        add(A, E, A);
        sub2x(C, C, west, north, D);
        mov2x(D, C, south, west);
        add(A, A, C, D);

        scamp5_in(C, threshold);
        scamp5_in(D, threshold);
        scamp5_in(E, threshold);

        //
        // Second Conv layer Reg B
        div(E, D, B);
        diva(E, D, C);
        div(D, C, B, E);
        neg(C, D);
        sub2x(B, C, east, south, D);
        sub(D, C, E);
        mov2x(E, E, north, north);
        add2x(E, C, E, east, east);
        movx(C, C, west);
        addx(B, C, B, north);
        addx(C, C, B, south);
        add2x(D, D, E, south, west);
        add(B, D, B, C);

        scamp5_in(C, threshold);
        scamp5_in(D, threshold);
        scamp5_in(E, threshold);

        // Addtion of two comvoluted images
        // NOTE!!: Here the sum of A and B is stored in D, to reduce num of "error"
        bus(C,A,B); //C := -(A + B) + error
        bus(D,C); //D := -C + error

        scamp5_in(A, threshold);
        scamp5_in(B, threshold);
        //

        // Third Conv layer
        div(C, A, D);
        div(A, B, E, C);
        diva(A, B, E);
        neg(B, A);
        sub(E, B, C);
        neg(C, A);
        add2x(C, B, C, south, west);
        addx(E, B, E, south);
        addx(B, C, B, west);
        subx(C, E, north, A);
        add(E, B, E, A);
        mov2x(A, A, west, north);
        add2x(E, E, B, east, east);
        add2x(B, B, A, east, east);
        addx(E, E, B, west);
        add(B, A, B);
        addx(E, E, A, north);
        addx(A, E, A, south);
        mov2x(E, C, west, south);
        addx(C, C, E, east);
        mov2x(E, C, west, north);
        add(B, B, C, E);


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


