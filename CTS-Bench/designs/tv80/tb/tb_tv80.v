`timescale 1ns/1ps
module tb_tv80;

reg        clk;
reg        reset_n;
reg        wait_n;
reg        int_n;
reg        nmi_n;
reg        busrq_n;
reg  [7:0] di;
wire       m1_n, mreq_n, iorq_n, rd_n, wr_n, rfsh_n, halt_n, busak_n;
wire [15:0] A;
wire [7:0]  dout;

tv80s dut (
    .clk(clk), .reset_n(reset_n),
    .wait_n(wait_n), .int_n(int_n), .nmi_n(nmi_n), .busrq_n(busrq_n),
    .di(di),
    .m1_n(m1_n), .mreq_n(mreq_n), .iorq_n(iorq_n),
    .rd_n(rd_n), .wr_n(wr_n), .rfsh_n(rfsh_n),
    .halt_n(halt_n), .busak_n(busak_n),
    .A(A), .dout(dout)
);

initial clk = 0;
always #5 clk = ~clk;

initial begin
    $dumpfile("tb_tv80.vcd");
    $dumpvars(0, tb_tv80);
    reset_n = 0; wait_n = 1; int_n = 1; nmi_n = 1; busrq_n = 1;
    di = 8'h00;
    repeat(4) @(posedge clk);
    reset_n = 1;
    repeat(5000) @(posedge clk);
    $finish;
end

// NOP feed
always @(posedge clk)
    if (!mreq_n && !rd_n) di <= 8'h00;

endmodule
