`timescale 1ns / 1ps
`default_nettype none

module tb_zipdiv;
    // 1. Declare signals exactly as they will be used in the port map
    reg i_clk;
    reg i_reset;
    reg i_wr;
    reg i_signed;
    reg [31:0] num;
    reg [31:0] den;
    
    wire o_busy;
    wire o_valid;
    wire o_err;
    wire [31:0] quotient;
    wire [3:0] flags;

    // 2. Instantiate the Unit Under Test (UUT)
    // Note: No #(.BW(BW)) here because gate-level netlists are fixed-width
    zipdiv uut (
        .i_clk(i_clk),
        .i_reset(i_reset),
        .i_wr(i_wr),
        .i_signed(i_signed),
        .i_numerator(num),
        .i_denominator(den),
        .o_busy(o_busy),
        .o_valid(o_valid),
        .o_err(o_err),
        .o_quotient(quotient),
        .o_flags(flags)
    );

    // 3. Clock Generation
    initial i_clk = 0;
    always #5 i_clk = ~i_clk;

    // 4. VCD and SAIF Recording
    initial begin
        $dumpfile("tb_zipdiv.vcd");
        $dumpvars(0, tb_zipdiv);
    end

    integer i;
    initial begin
        // Initialize
        i_reset = 1;
        i_wr = 0;
        i_signed = 0;
        num = 0;
        den = 0;

        repeat(10) @(posedge i_clk);
        i_reset = 0;
        @(posedge i_clk);

        // STRESS TEST: 50 random divisions for high toggle activity
        for (i = 0; i < 50; i = i + 1) begin
            wait(!o_busy);
            @(posedge i_clk);
            num = $urandom;
            den = $urandom % 5000 + 1; 
            i_signed = $urandom % 2;
            i_wr = 1;
            @(posedge i_clk);
            i_wr = 0;
            wait(o_valid);
            repeat(2) @(posedge i_clk);
        end

        $display("Simulation Complete for Unseen Design: zipdiv");
        $finish;
    end
endmodule