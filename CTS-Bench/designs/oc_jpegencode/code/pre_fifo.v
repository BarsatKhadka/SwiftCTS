/////////////////////////////////////////////////////////////////////
////                                                             ////
////  JPEG Encoder Core - Verilog                                ////
////                                                             ////
////  Author: David Lundgren                                     ////
////          davidklun@gmail.com                                ////
////                                                             ////
/////////////////////////////////////////////////////////////////////
////                                                             ////
//// Copyright (C) 2009 David Lundgren                           ////
////                  davidklun@gmail.com                        ////
////                                                             ////
//// This source file may be used and distributed without        ////
//// restriction provided that this copyright statement is not   ////
//// removed from the file and that any derivative work contains ////
//// the original copyright notice and the associated disclaimer.////
////                                                             ////
////     THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY     ////
//// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED   ////
//// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS   ////
//// FOR A PARTICULAR PURPOSE. IN NO EVENT SHALL THE AUTHOR      ////
//// OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,         ////
//// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES    ////
//// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE   ////
//// GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR        ////
//// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF  ////
//// LIABILITY, WHETHER IN  CONTRACT, STRICT LIABILITY, OR TORT  ////
//// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT  ////
//// OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         ////
//// POSSIBILITY OF SUCH DAMAGE.                                 ////
////                                                             ////
/////////////////////////////////////////////////////////////////////


/* Grayscale (Y-only) version: RGB2YCBCR, Cb and Cr pipelines removed.
   Accepts 8-bit grayscale input directly. */

`timescale 1ns / 100ps

module pre_fifo(clk, rst, enable, data_in, y_JPEG_bitstream,
y_data_ready, y_orc, y_eob_output, y_eob_empty);
input		clk, rst, enable;
input	[7:0]	data_in;
output  [31:0]  y_JPEG_bitstream;
output		y_data_ready;
output  [4:0] y_orc;
output		y_eob_output;
output		y_eob_empty;

	yd_q_h u13(.clk(clk), .rst(rst), .enable(enable), .data_in(data_in),
	.JPEG_bitstream(y_JPEG_bitstream),
	 .data_ready(y_data_ready), .y_orc(y_orc),
	 .end_of_block_output(y_eob_output), .end_of_block_empty(y_eob_empty));

	endmodule
