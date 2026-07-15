`timescale 1ns/1ps
module tb_mem_ctrl;

reg         clk_i, rst_i;
reg  [31:0] wb_data_i;
wire [31:0] wb_data_o;
reg  [31:0] wb_addr_i;
reg  [3:0]  wb_sel_i;
reg         wb_we_i, wb_cyc_i, wb_stb_i;
wire        wb_ack_o, wb_err_o;
reg         susp_req_i, resume_req_i;
wire        suspended_o;
wire [31:0] poc_o;
reg         mc_clk_i;
reg         mc_br_pad_i;
wire        mc_bg_pad_o;
reg         mc_ack_pad_i;
wire [23:0] mc_addr_pad_o;
reg  [31:0] mc_data_pad_i;
wire [31:0] mc_data_pad_o;
reg  [3:0]  mc_dp_pad_i;
wire [3:0]  mc_dp_pad_o;
wire        mc_doe_pad_doe_o;
wire [3:0]  mc_dqm_pad_o;
wire        mc_oe_pad_o_, mc_we_pad_o_, mc_cas_pad_o_, mc_ras_pad_o_, mc_cke_pad_o_;
wire [7:0]  mc_cs_pad_o_;
reg         mc_sts_pad_i;
wire        mc_rp_pad_o_, mc_vpen_pad_o, mc_adsc_pad_o_, mc_adv_pad_o_, mc_zz_pad_o;
wire        mc_coe_pad_coe_o;

mc_top dut (
    .clk_i(clk_i), .rst_i(rst_i),
    .wb_data_i(wb_data_i), .wb_data_o(wb_data_o), .wb_addr_i(wb_addr_i),
    .wb_sel_i(wb_sel_i), .wb_we_i(wb_we_i), .wb_cyc_i(wb_cyc_i),
    .wb_stb_i(wb_stb_i), .wb_ack_o(wb_ack_o), .wb_err_o(wb_err_o),
    .susp_req_i(susp_req_i), .resume_req_i(resume_req_i), .suspended_o(suspended_o),
    .poc_o(poc_o),
    .mc_clk_i(mc_clk_i), .mc_br_pad_i(mc_br_pad_i), .mc_bg_pad_o(mc_bg_pad_o),
    .mc_ack_pad_i(mc_ack_pad_i), .mc_addr_pad_o(mc_addr_pad_o),
    .mc_data_pad_i(mc_data_pad_i), .mc_data_pad_o(mc_data_pad_o),
    .mc_dp_pad_i(mc_dp_pad_i), .mc_dp_pad_o(mc_dp_pad_o),
    .mc_doe_pad_doe_o(mc_doe_pad_doe_o), .mc_dqm_pad_o(mc_dqm_pad_o),
    .mc_oe_pad_o_(mc_oe_pad_o_), .mc_we_pad_o_(mc_we_pad_o_),
    .mc_cas_pad_o_(mc_cas_pad_o_), .mc_ras_pad_o_(mc_ras_pad_o_),
    .mc_cke_pad_o_(mc_cke_pad_o_), .mc_cs_pad_o_(mc_cs_pad_o_),
    .mc_sts_pad_i(mc_sts_pad_i), .mc_rp_pad_o_(mc_rp_pad_o_),
    .mc_vpen_pad_o(mc_vpen_pad_o), .mc_adsc_pad_o_(mc_adsc_pad_o_),
    .mc_adv_pad_o_(mc_adv_pad_o_), .mc_zz_pad_o(mc_zz_pad_o),
    .mc_coe_pad_coe_o(mc_coe_pad_coe_o)
);

initial clk_i = 0;
always #5 clk_i = ~clk_i;
always #5 mc_clk_i = clk_i;

initial begin
    $dumpfile("tb_mem_ctrl.vcd");
    $dumpvars(0, tb_mem_ctrl);
    rst_i = 1;
    wb_data_i = 0; wb_addr_i = 0; wb_sel_i = 4'hf;
    wb_we_i = 0; wb_cyc_i = 0; wb_stb_i = 0;
    susp_req_i = 0; resume_req_i = 0;
    mc_br_pad_i = 0; mc_ack_pad_i = 0;
    mc_data_pad_i = 0; mc_dp_pad_i = 0; mc_sts_pad_i = 0;
    repeat(8) @(posedge clk_i);
    rst_i = 0;
    repeat(2000) @(posedge clk_i);
    $finish;
end

endmodule
