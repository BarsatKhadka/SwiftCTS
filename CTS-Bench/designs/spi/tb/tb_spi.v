`timescale 1ns/1ps
module tb_spi;

reg        clk_i;
reg        rst_i;
reg        cyc_i;
reg        stb_i;
reg  [2:0] adr_i;
reg        we_i;
reg  [7:0] dat_i;
wire [7:0] dat_o;
wire       ack_o;
wire       inta_o;
wire       sck_o;
reg        miso_i;
wire       mosi_o;
wire       ss_o;

simple_spi dut (
    .clk_i(clk_i), .rst_i(rst_i),
    .cyc_i(cyc_i), .stb_i(stb_i), .adr_i(adr_i),
    .we_i(we_i), .dat_i(dat_i), .dat_o(dat_o),
    .ack_o(ack_o), .inta_o(inta_o),
    .sck_o(sck_o), .miso_i(miso_i), .mosi_o(mosi_o), .ss_o(ss_o)
);

initial clk_i = 0;
always #5 clk_i = ~clk_i;

initial begin
    $dumpfile("tb_spi.vcd");
    $dumpvars(0, tb_spi);
    rst_i = 1; cyc_i = 0; stb_i = 0; we_i = 0;
    adr_i = 0; dat_i = 0; miso_i = 0;
    repeat(4) @(posedge clk_i);
    rst_i = 0;
    repeat(2000) @(posedge clk_i);
    $finish;
end

endmodule
