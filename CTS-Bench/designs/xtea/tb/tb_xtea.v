`timescale 1ns / 1ps
`default_nettype none

module tb_xtea;
    reg          clock;
    reg          reset;
    reg          mode;
    reg  [31:0]  data_in1;
    reg  [31:0]  data_in2;
    reg  [127:0] key_in;
    wire [31:0]  data_out1;
    wire [31:0]  data_out2;
    wire         all_done;

    xtea uut (
        .clock(clock),
        .reset(reset),
        .mode(mode),
        .data_in1(data_in1),
        .data_in2(data_in2),
        .key_in(key_in),
        .data_out1(data_out1),
        .data_out2(data_out2),
        .all_done(all_done)
    );

    initial clock = 0;
    always #5 clock = ~clock;

    initial begin
        $dumpfile("tb_xtea.vcd");
        $dumpvars(0, tb_xtea);
    end

    integer i;
    initial begin
        reset    = 1;
        mode     = 0;
        data_in1 = 0;
        data_in2 = 0;
        key_in   = 0;
        repeat(5) @(posedge clock);
        reset = 0;
        @(posedge clock);

        for (i = 0; i < 20; i = i + 1) begin
            mode     = i[0];
            data_in1 = $random;
            data_in2 = $random;
            key_in   = {$random, $random, $random, $random};
            @(posedge clock);
            wait(all_done);
            @(posedge clock);
        end

        repeat(20) @(posedge clock);
        $finish;
    end
endmodule
