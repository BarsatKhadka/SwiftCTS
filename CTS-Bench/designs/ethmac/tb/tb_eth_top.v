`include "tb_eth_defines.v"
`include "ethmac_defines.v"
`include "timescale.v"

module tb_eth_top();

parameter Tp = 1;

// Wishbone Signals
reg            WB_CLK_I;
reg            WB_RST_I;
reg   [31:0]   WB_DAT_I;
reg   [31:0]   WB_ADR_I;
reg    [3:0]   WB_SEL_I;
reg            WB_WE_I;
reg            WB_CYC_I;
reg            WB_STB_I;
wire  [31:0]   WB_DAT_O;
wire           WB_ACK_O;
wire           WB_ERR_O;

// Master Interface (Tied off for GLS stability)
wire    [31:0]    m_wb_adr_o;
wire     [3:0]    m_wb_sel_o;
wire              m_wb_we_o;
reg     [31:0]    m_wb_dat_i;
wire    [31:0]    m_wb_dat_o;
wire              m_wb_cyc_o;
wire              m_wb_stb_o;
reg               m_wb_ack_i;
reg               m_wb_err_i;

// PHY Interface
reg            MTxClk;
wire   [3:0]   MTxD;
wire           MTxEn;
wire           MTxErr;
reg            MRxClk;
reg    [3:0]   MRxD;
reg            MRxDV;
reg            MRxErr;
reg            MColl;
reg            MCrs;
reg            Mdi_I;
wire           Mdo_O;
wire           Mdo_OE;
wire           Mdc_O;

reg [7:0] memory0 [0:65535];
reg [7:0] memory1 [0:65535];
reg [7:0] memory2 [0:65535];
reg [7:0] memory3 [0:65535];

reg WishboneBusy;
reg StartTB;
integer i;
reg [31:0] random_addr;

// --- 1. SAIF & VCD SCOPE CONTROL ---
initial begin
  if ($test$plusargs("vcd")) begin
    $dumpfile("testbench.vcd");
    // Scope Level 0 but targeted at ethtop instance only.
    // This captures internal gates for the GNN but skips testbench bloat.
    $dumpvars(0, tb_eth_top.ethtop); 
  end
end

// HARD WATCHDOG: If the simulation hangs (common in GLS), kill it after 30us.
initial begin
    #30000;
    $display("!!! WATCHDOG TIMEOUT !!!");
    $finish;
end

// --- 2. CORE INSTANTIATION ---
ethmac ethtop (
  .wb_clk_i(WB_CLK_I), .wb_rst_i(WB_RST_I), .wb_dat_i(WB_DAT_I), .wb_dat_o(WB_DAT_O), 
  .wb_adr_i(WB_ADR_I[11:2]), .wb_sel_i(WB_SEL_I), .wb_we_i(WB_WE_I), .wb_cyc_i(WB_CYC_I), 
  .wb_stb_i(WB_STB_I), .wb_ack_o(WB_ACK_O), .wb_err_o(WB_ERR_O), 
  .m_wb_adr_o(m_wb_adr_o), .m_wb_sel_o(m_wb_sel_o), .m_wb_we_o(m_wb_we_o), .m_wb_dat_i(m_wb_dat_i), 
  .m_wb_dat_o(m_wb_dat_o), .m_wb_cyc_o(m_wb_cyc_o), .m_wb_stb_o(m_wb_stb_o), .m_wb_ack_i(m_wb_ack_i), 
  .m_wb_err_i(m_wb_err_i), 
  .mtx_clk_pad_i(MTxClk), .mtxd_pad_o(MTxD), .mtxen_pad_o(MTxEn), .mtxerr_pad_o(MTxErr),
  .mrx_clk_pad_i(MRxClk), .mrxd_pad_i(MRxD), .mrxdv_pad_i(MRxDV), .mrxerr_pad_i(MRxErr), 
  .mcoll_pad_i(MColl), .mcrs_pad_i(MCrs), 
  .mdc_pad_o(Mdc_O), .md_pad_i(Mdi_I), .md_pad_o(Mdo_O), .md_padoe_o(Mdo_OE),
  .int_o()
);

// --- 3. ROBUST INITIALIZATION (Clears X-States) ---
initial begin
  // Drive all inputs to 0 immediately to prevent X-propagation
  WB_CLK_I = 0; WB_RST_I = 1; WB_DAT_I = 0; WB_ADR_I = 0;
  WB_SEL_I = 0; WB_WE_I = 0; WB_CYC_I = 0; WB_STB_I = 0;
  m_wb_ack_i = 0; m_wb_err_i = 0; MTxClk = 0; MRxClk = 0;
  MRxD = 0; MRxDV = 0; MRxErr = 0; MColl = 0; MCrs = 0; Mdi_I = 0;
  WishboneBusy = 0; StartTB = 0;

  // Wait 1000ns for the Gate-Level model to settle
  #1000 WB_RST_I = 0; 
  #100  StartTB = 1;
end

always #12.5 WB_CLK_I = ~WB_CLK_I; // 40MHz
always #200  MTxClk  = ~MTxClk;   // 2.5MHz
always #200  MRxClk  = ~MRxClk;

// --- 4. REPRESENTATIVE TRAFFIC BURST ---
initial begin
  wait(StartTB);
  $display("Starting Robust Activity Burst...");

  // Force some register writes to trigger internal gate transitions
  // ETH_MODER
  WishboneWrite(32'h0000a40b, {26'h0, `ETH_MODER_ADR<<2});
  // MAC Addresses
  WishboneWrite(32'h00020304, {26'h0, `ETH_MAC_ADDR1_ADR<<2});
  WishboneWrite(32'h05060708, {26'h0, `ETH_MAC_ADDR0_ADR<<2});

  // PERFORM RANDOM TRANSFERS
  // This creates the "Entropy" your GNN needs to distinguish placements.
  for (i = 0; i < 60; i = i + 1) begin
    random_addr = ($random % 64) << 2; // Random register address
    WishboneWrite($random, random_addr); 
    #10; // Propagation delay
  end

  $display("Simulation Snapshot Captured. Closing VCD.");
  $finish; // Required to properly exit the vvp process
end

// --- 5. ROBUST WISHBONE TASK ---
task WishboneWrite;
  input [31:0] Data;
  input [31:0] Address;
  begin
    wait (~WishboneBusy);
    WishboneBusy = 1;
    @(posedge WB_CLK_I);
    #1;
    WB_ADR_I = Address;
    WB_DAT_I = Data;
    WB_WE_I  = 1'b1;
    WB_CYC_I = 1'b1;
    WB_STB_I = 1'b1;
    WB_SEL_I = 4'hf;
    
    // Wait for ACK or Timeout (Prevents GLS Hangs)
    fork : wait_ack
      begin
        wait(WB_ACK_O);
        disable wait_ack;
      end
      begin
        #1000; // If no ACK in 1000ns, something failed.
        $display("WARNING: Wishbone Timeout at Addr %x", Address);
        disable wait_ack;
      end
    join

    @(posedge WB_CLK_I);
    #1;
    WB_WE_I  = 0;
    WB_CYC_I = 0;
    WB_STB_I = 0;
    WishboneBusy = 0;
  end
endtask

// --- 6. DUMMY RESPONDERS ---
// Responds to Master requests if the DUT attempts DMA
always @ (posedge WB_CLK_I) begin
  if(m_wb_cyc_o & m_wb_stb_o) begin
    m_wb_ack_i <= #Tp 1'b1;
    @(posedge WB_CLK_I);
    m_wb_ack_i <= #Tp 1'b0;
  end
end

endmodule

// Fake module to satisfy the instance in top
module bench_cop(input wb_clk_i, wb_rst_i, input [31:0] wb_dat_i, output [31:0] wb_dat_o, input [11:2] wb_adr_i, input [3:0] wb_sel_i, input wb_we_i, wb_cyc_i, wb_stb_i, output wb_ack_o, wb_err_o, output [31:0] m_wb_adr_o, output [3:0] m_wb_sel_o, output m_wb_we_o, input [31:0] m_wb_dat_i, output [31:0] m_wb_dat_o, output m_wb_cyc_o, m_wb_stb_o, input m_wb_ack_i, m_wb_err_i, input mtx_clk_pad_i, output [3:0] mtxd_pad_o, output mtxen_pad_o, mtxerr_pad_o, input mrx_clk_pad_i, input [3:0] mrxd_pad_i, input mrxdv_pad_i, mrxerr_pad_i, input mcoll_pad_i, mcrs_pad_i, output mdc_pad_o, input md_pad_i, output md_pad_o, md_padoe_o, output int_o);
  assign wb_ack_o = 0; assign wb_err_o = 0; assign m_wb_cyc_o = 0; assign m_wb_stb_o = 0; assign m_wb_we_o = 0;
endmodule