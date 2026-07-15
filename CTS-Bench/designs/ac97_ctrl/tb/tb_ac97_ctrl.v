`timescale 1ns/1ps
module tb_ac97_ctrl;

reg         clk_i, rst_i;
reg  [31:0] wb_data_i;
wire [31:0] wb_data_o;
reg  [31:0] wb_addr_i;
reg  [3:0]  wb_sel_i;
reg         wb_we_i, wb_cyc_i, wb_stb_i;
wire        wb_ack_o, wb_err_o;
wire        int_o;
wire [8:0]  dma_req_o;
reg  [8:0]  dma_ack_i;
wire        suspended_o;
reg         bit_clk_pad_i;
wire        sync_pad_o, sdata_pad_o;
reg         sdata_pad_i;
wire        ac97_reset_pad_o_;

ac97_top dut (
    .clk_i(clk_i), .rst_i(rst_i),
    .wb_data_i(wb_data_i), .wb_data_o(wb_data_o),
    .wb_addr_i(wb_addr_i), .wb_sel_i(wb_sel_i),
    .wb_we_i(wb_we_i), .wb_cyc_i(wb_cyc_i), .wb_stb_i(wb_stb_i),
    .wb_ack_o(wb_ack_o), .wb_err_o(wb_err_o),
    .int_o(int_o), .dma_req_o(dma_req_o), .dma_ack_i(dma_ack_i),
    .suspended_o(suspended_o),
    .bit_clk_pad_i(bit_clk_pad_i),
    .sync_pad_o(sync_pad_o), .sdata_pad_o(sdata_pad_o),
    .sdata_pad_i(sdata_pad_i), .ac97_reset_pad_o_(ac97_reset_pad_o_)
);

initial clk_i = 0;
always #5 clk_i = ~clk_i;

// AC97 bit clock: typically 48kHz * 256 = 12.288 MHz (~81ns period)
initial bit_clk_pad_i = 0;
always #41 bit_clk_pad_i = ~bit_clk_pad_i;

initial begin
    $dumpfile("tb_ac97_ctrl.vcd");
    $dumpvars(0, tb_ac97_ctrl);
    rst_i = 1;
    wb_data_i = 0; wb_addr_i = 0; wb_sel_i = 4'hf;
    wb_we_i = 0; wb_cyc_i = 0; wb_stb_i = 0;
    dma_ack_i = 0; sdata_pad_i = 0;
    repeat(8) @(posedge clk_i);
    rst_i = 0;
    repeat(2000) @(posedge clk_i);
    $finish;
end

endmodule
