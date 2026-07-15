`timescale 1ns/1ps
module tb_usb_phy;

reg        clk;
reg        rst;
reg        phy_tx_mode;
reg        rxd;
reg        rxdp;
reg        rxdn;
reg  [7:0] DataOut_i;
reg        TxValid_i;
wire       TxReady_o;
wire       txdp, txdn, txoe;
wire       usb_rst;
wire [7:0] DataIn_o;
wire       RxValid_o;
wire       RxActive_o;
wire       RxError_o;
wire [1:0] LineState_o;

usb_phy dut (
    .clk(clk), .rst(rst),
    .phy_tx_mode(phy_tx_mode), .usb_rst(usb_rst),
    .txdp(txdp), .txdn(txdn), .txoe(txoe),
    .rxd(rxd), .rxdp(rxdp), .rxdn(rxdn),
    .DataOut_i(DataOut_i), .TxValid_i(TxValid_i), .TxReady_o(TxReady_o),
    .RxValid_o(RxValid_o), .RxActive_o(RxActive_o), .RxError_o(RxError_o),
    .DataIn_o(DataIn_o), .LineState_o(LineState_o)
);

initial clk = 0;
always #5 clk = ~clk;

initial begin
    $dumpfile("tb_usb_phy.vcd");
    $dumpvars(0, tb_usb_phy);
    rst = 1; phy_tx_mode = 1;
    rxd = 0; rxdp = 1; rxdn = 0;
    DataOut_i = 8'h00; TxValid_i = 0;
    repeat(8) @(posedge clk);
    rst = 0;
    repeat(2000) @(posedge clk);
    $finish;
end

endmodule
