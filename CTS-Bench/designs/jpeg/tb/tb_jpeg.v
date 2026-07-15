`timescale 1ns/1ps
module tb_jpeg;

reg        clk, rst;
reg        end_of_file_signal;
reg        enable;
reg [23:0] data_in;
wire [31:0] JPEG_bitstream;
wire        data_ready;
wire [4:0]  end_of_file_bitstream_count;
wire        eof_data_partial_ready;

jpeg_top dut (
    .clk(clk), .rst(rst),
    .end_of_file_signal(end_of_file_signal),
    .enable(enable), .data_in(data_in),
    .JPEG_bitstream(JPEG_bitstream), .data_ready(data_ready),
    .end_of_file_bitstream_count(end_of_file_bitstream_count),
    .eof_data_partial_ready(eof_data_partial_ready)
);

initial clk = 0;
always #5 clk = ~clk;

integer i;
initial begin
    $dumpfile("tb_jpeg.vcd");
    $dumpvars(0, tb_jpeg);
    rst = 1; enable = 0; end_of_file_signal = 0; data_in = 0;
    repeat(4) @(posedge clk);
    rst = 0;
    enable = 1;
    for (i = 0; i < 2000; i = i + 1) begin
        @(posedge clk);
        data_in <= {$random} & 24'hffffff;
    end
    $finish;
end

endmodule
