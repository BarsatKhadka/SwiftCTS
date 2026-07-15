`timescale 1ns/1ps
module tb_pci;

// WB system
reg         wb_clk_i, wb_rst_i;
wire        wb_rst_o, wb_int_o;
reg         wb_int_i;
// WB slave
reg  [31:0] wbs_adr_i, wbs_dat_i;
wire [31:0] wbs_dat_o;
reg  [3:0]  wbs_sel_i;
reg         wbs_cyc_i, wbs_stb_i, wbs_we_i;
reg  [2:0]  wbs_cti_i;
reg  [1:0]  wbs_bte_i;
wire        wbs_ack_o, wbs_rty_o, wbs_err_o;
// WB master
wire [31:0] wbm_adr_o, wbm_dat_o;
reg  [31:0] wbm_dat_i;
wire [3:0]  wbm_sel_o;
wire        wbm_cyc_o, wbm_stb_o, wbm_we_o;
wire [2:0]  wbm_cti_o;
wire [1:0]  wbm_bte_o;
reg         wbm_ack_i, wbm_rty_i, wbm_err_i;
// PCI system
reg         pci_clk_i, pci_rst_i;
wire        pci_rst_o, pci_rst_oe_o;
reg         pci_inta_i;
wire        pci_inta_o, pci_inta_oe_o;
wire        pci_req_o, pci_req_oe_o;
reg         pci_gnt_i;
reg         pci_frame_i;
wire        pci_frame_o, pci_frame_oe_o, pci_irdy_oe_o;
wire        pci_devsel_oe_o, pci_trdy_oe_o, pci_stop_oe_o;
wire [31:0] pci_ad_oe_o;
wire [3:0]  pci_cbe_oe_o;
reg         pci_irdy_i;
wire        pci_irdy_o;
reg         pci_idsel_i;
reg         pci_devsel_i;
wire        pci_devsel_o;
reg         pci_trdy_i;
wire        pci_trdy_o;
reg         pci_stop_i;
wire        pci_stop_o;
reg  [31:0] pci_ad_i;
wire [31:0] pci_ad_o;
reg  [3:0]  pci_cbe_i;
wire [3:0]  pci_cbe_o;
reg         pci_par_i;
wire        pci_par_o, pci_par_oe_o;
reg         pci_perr_i;
wire        pci_perr_o, pci_perr_oe_o;
wire        pci_serr_o, pci_serr_oe_o;

pci_bridge32 dut (
    .wb_clk_i(wb_clk_i), .wb_rst_i(wb_rst_i),
    .wb_rst_o(wb_rst_o), .wb_int_i(wb_int_i), .wb_int_o(wb_int_o),
    .wbs_adr_i(wbs_adr_i), .wbs_dat_i(wbs_dat_i), .wbs_dat_o(wbs_dat_o),
    .wbs_sel_i(wbs_sel_i), .wbs_cyc_i(wbs_cyc_i), .wbs_stb_i(wbs_stb_i),
    .wbs_we_i(wbs_we_i), .wbs_cti_i(wbs_cti_i), .wbs_bte_i(wbs_bte_i),
    .wbs_ack_o(wbs_ack_o), .wbs_rty_o(wbs_rty_o), .wbs_err_o(wbs_err_o),
    .wbm_adr_o(wbm_adr_o), .wbm_dat_i(wbm_dat_i), .wbm_dat_o(wbm_dat_o),
    .wbm_sel_o(wbm_sel_o), .wbm_cyc_o(wbm_cyc_o), .wbm_stb_o(wbm_stb_o),
    .wbm_we_o(wbm_we_o), .wbm_cti_o(wbm_cti_o), .wbm_bte_o(wbm_bte_o),
    .wbm_ack_i(wbm_ack_i), .wbm_rty_i(wbm_rty_i), .wbm_err_i(wbm_err_i),
    .pci_clk_i(pci_clk_i), .pci_rst_i(pci_rst_i),
    .pci_rst_o(pci_rst_o), .pci_rst_oe_o(pci_rst_oe_o),
    .pci_inta_i(pci_inta_i), .pci_inta_o(pci_inta_o), .pci_inta_oe_o(pci_inta_oe_o),
    .pci_req_o(pci_req_o), .pci_req_oe_o(pci_req_oe_o),
    .pci_gnt_i(pci_gnt_i),
    .pci_frame_i(pci_frame_i), .pci_frame_o(pci_frame_o),
    .pci_frame_oe_o(pci_frame_oe_o), .pci_irdy_oe_o(pci_irdy_oe_o),
    .pci_devsel_oe_o(pci_devsel_oe_o), .pci_trdy_oe_o(pci_trdy_oe_o),
    .pci_stop_oe_o(pci_stop_oe_o), .pci_ad_oe_o(pci_ad_oe_o),
    .pci_cbe_oe_o(pci_cbe_oe_o),
    .pci_irdy_i(pci_irdy_i), .pci_irdy_o(pci_irdy_o),
    .pci_idsel_i(pci_idsel_i),
    .pci_devsel_i(pci_devsel_i), .pci_devsel_o(pci_devsel_o),
    .pci_trdy_i(pci_trdy_i), .pci_trdy_o(pci_trdy_o),
    .pci_stop_i(pci_stop_i), .pci_stop_o(pci_stop_o),
    .pci_ad_i(pci_ad_i), .pci_ad_o(pci_ad_o),
    .pci_cbe_i(pci_cbe_i), .pci_cbe_o(pci_cbe_o),
    .pci_par_i(pci_par_i), .pci_par_o(pci_par_o), .pci_par_oe_o(pci_par_oe_o),
    .pci_perr_i(pci_perr_i), .pci_perr_o(pci_perr_o), .pci_perr_oe_o(pci_perr_oe_o),
    .pci_serr_o(pci_serr_o), .pci_serr_oe_o(pci_serr_oe_o)
);

initial wb_clk_i = 0;
always #5 wb_clk_i = ~wb_clk_i;

// PCI clock: 33 MHz (~15ns period)
initial pci_clk_i = 0;
always #15 pci_clk_i = ~pci_clk_i;

initial begin
    $dumpfile("tb_pci.vcd");
    $dumpvars(0, tb_pci);
    wb_rst_i = 1; pci_rst_i = 1;
    wb_int_i = 0;
    wbs_adr_i = 0; wbs_dat_i = 0; wbs_sel_i = 4'hf;
    wbs_cyc_i = 0; wbs_stb_i = 0; wbs_we_i = 0;
    wbs_cti_i = 0; wbs_bte_i = 0;
    wbm_dat_i = 0; wbm_ack_i = 0; wbm_rty_i = 0; wbm_err_i = 0;
    pci_gnt_i = 1; pci_frame_i = 1; pci_irdy_i = 1;
    pci_devsel_i = 1; pci_trdy_i = 1; pci_stop_i = 1;
    pci_idsel_i = 0; pci_inta_i = 1;
    pci_ad_i = 0; pci_cbe_i = 4'hf;
    pci_par_i = 0; pci_perr_i = 1;
    repeat(8) @(posedge wb_clk_i);
    wb_rst_i = 0; pci_rst_i = 0;
    repeat(2000) @(posedge wb_clk_i);
    $finish;
end

endmodule
