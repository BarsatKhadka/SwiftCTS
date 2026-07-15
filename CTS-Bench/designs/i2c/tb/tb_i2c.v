`timescale 1ns/1ps
module tb_i2c;

reg        wb_clk_i;
reg        wb_rst_i;
reg        arst_i;
reg  [2:0] wb_adr_i;
reg  [7:0] wb_dat_i;
wire [7:0] wb_dat_o;
reg        wb_we_i;
reg        wb_stb_i;
reg        wb_cyc_i;
wire       wb_ack_o;
wire       wb_inta_o;
reg        scl_pad_i;
wire       scl_pad_o;
wire       scl_padoen_o;
reg        sda_pad_i;
wire       sda_pad_o;
wire       sda_padoen_o;

i2c_master_top dut (
    .wb_clk_i(wb_clk_i), .wb_rst_i(wb_rst_i), .arst_i(arst_i),
    .wb_adr_i(wb_adr_i), .wb_dat_i(wb_dat_i), .wb_dat_o(wb_dat_o),
    .wb_we_i(wb_we_i), .wb_stb_i(wb_stb_i), .wb_cyc_i(wb_cyc_i),
    .wb_ack_o(wb_ack_o), .wb_inta_o(wb_inta_o),
    .scl_pad_i(scl_pad_i), .scl_pad_o(scl_pad_o), .scl_padoen_o(scl_padoen_o),
    .sda_pad_i(sda_pad_i), .sda_pad_o(sda_pad_o), .sda_padoen_o(sda_padoen_o)
);

initial wb_clk_i = 0;
always #5 wb_clk_i = ~wb_clk_i;

initial begin
    $dumpfile("tb_i2c.vcd");
    $dumpvars(0, tb_i2c);
    wb_rst_i = 1; arst_i = 1;
    wb_adr_i = 0; wb_dat_i = 0; wb_we_i = 0; wb_stb_i = 0; wb_cyc_i = 0;
    scl_pad_i = 1; sda_pad_i = 1;
    repeat(4) @(posedge wb_clk_i);
    wb_rst_i = 0; arst_i = 0;
    repeat(2000) @(posedge wb_clk_i);
    $finish;
end

endmodule
