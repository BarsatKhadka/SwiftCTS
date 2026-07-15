`timescale 1ns/1ps
module tb_wb_dma;

reg         clk_i, rst_i;
// WB0 slave
reg  [31:0] wb0s_data_i;
wire [31:0] wb0s_data_o;
reg  [31:0] wb0_addr_i;
reg  [3:0]  wb0_sel_i;
reg         wb0_we_i, wb0_cyc_i, wb0_stb_i;
wire        wb0_ack_o, wb0_err_o, wb0_rty_o;
// WB0 master
reg  [31:0] wb0m_data_i;
wire [31:0] wb0m_data_o;
wire [31:0] wb0_addr_o;
wire [3:0]  wb0_sel_o;
wire        wb0_we_o, wb0_cyc_o, wb0_stb_o;
reg         wb0_ack_i, wb0_err_i, wb0_rty_i;
// WB1 slave
reg  [31:0] wb1s_data_i;
wire [31:0] wb1s_data_o;
reg  [31:0] wb1_addr_i;
reg  [3:0]  wb1_sel_i;
reg         wb1_we_i, wb1_cyc_i, wb1_stb_i;
wire        wb1_ack_o, wb1_err_o, wb1_rty_o;
// WB1 master
reg  [31:0] wb1m_data_i;
wire [31:0] wb1m_data_o;
wire [31:0] wb1_addr_o;
wire [3:0]  wb1_sel_o;
wire        wb1_we_o, wb1_cyc_o, wb1_stb_o;
reg         wb1_ack_i, wb1_err_i, wb1_rty_i;
// DMA
reg         dma_req_i, dma_nd_i, dma_rest_i;
wire        dma_ack_o;
wire        inta_o, intb_o;

wb_dma_top dut (
    .clk_i(clk_i), .rst_i(rst_i),
    .wb0s_data_i(wb0s_data_i), .wb0s_data_o(wb0s_data_o),
    .wb0_addr_i(wb0_addr_i), .wb0_sel_i(wb0_sel_i),
    .wb0_we_i(wb0_we_i), .wb0_cyc_i(wb0_cyc_i), .wb0_stb_i(wb0_stb_i),
    .wb0_ack_o(wb0_ack_o), .wb0_err_o(wb0_err_o), .wb0_rty_o(wb0_rty_o),
    .wb0m_data_i(wb0m_data_i), .wb0m_data_o(wb0m_data_o),
    .wb0_addr_o(wb0_addr_o), .wb0_sel_o(wb0_sel_o),
    .wb0_we_o(wb0_we_o), .wb0_cyc_o(wb0_cyc_o), .wb0_stb_o(wb0_stb_o),
    .wb0_ack_i(wb0_ack_i), .wb0_err_i(wb0_err_i), .wb0_rty_i(wb0_rty_i),
    .wb1s_data_i(wb1s_data_i), .wb1s_data_o(wb1s_data_o),
    .wb1_addr_i(wb1_addr_i), .wb1_sel_i(wb1_sel_i),
    .wb1_we_i(wb1_we_i), .wb1_cyc_i(wb1_cyc_i), .wb1_stb_i(wb1_stb_i),
    .wb1_ack_o(wb1_ack_o), .wb1_err_o(wb1_err_o), .wb1_rty_o(wb1_rty_o),
    .wb1m_data_i(wb1m_data_i), .wb1m_data_o(wb1m_data_o),
    .wb1_addr_o(wb1_addr_o), .wb1_sel_o(wb1_sel_o),
    .wb1_we_o(wb1_we_o), .wb1_cyc_o(wb1_cyc_o), .wb1_stb_o(wb1_stb_o),
    .wb1_ack_i(wb1_ack_i), .wb1_err_i(wb1_err_i), .wb1_rty_i(wb1_rty_i),
    .dma_req_i(dma_req_i), .dma_ack_o(dma_ack_o),
    .dma_nd_i(dma_nd_i), .dma_rest_i(dma_rest_i),
    .inta_o(inta_o), .intb_o(intb_o)
);

initial clk_i = 0;
always #5 clk_i = ~clk_i;

initial begin
    $dumpfile("tb_wb_dma.vcd");
    $dumpvars(0, tb_wb_dma);
    rst_i = 1;
    wb0s_data_i = 0; wb0_addr_i = 0; wb0_sel_i = 4'hf;
    wb0_we_i = 0; wb0_cyc_i = 0; wb0_stb_i = 0;
    wb0m_data_i = 0; wb0_ack_i = 0; wb0_err_i = 0; wb0_rty_i = 0;
    wb1s_data_i = 0; wb1_addr_i = 0; wb1_sel_i = 4'hf;
    wb1_we_i = 0; wb1_cyc_i = 0; wb1_stb_i = 0;
    wb1m_data_i = 0; wb1_ack_i = 0; wb1_err_i = 0; wb1_rty_i = 0;
    dma_req_i = 0; dma_nd_i = 1; dma_rest_i = 0;
    repeat(8) @(posedge clk_i);
    rst_i = 0;
    repeat(2000) @(posedge clk_i);
    $finish;
end

endmodule
