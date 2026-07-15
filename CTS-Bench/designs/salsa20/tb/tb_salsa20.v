`timescale 1ns / 1ps
`default_nettype none

module tb_salsa20;
    reg         clk;
    reg         reset_n;
    reg         cs;
    reg         we;
    reg  [7:0]  address;
    reg  [31:0] write_data;
    wire [31:0] read_data;
    wire        error;

    salsa20 uut (
        .clk(clk),
        .reset_n(reset_n),
        .cs(cs),
        .we(we),
        .address(address),
        .write_data(write_data),
        .read_data(read_data),
        .error(error)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    initial begin
        $dumpfile("tb_salsa20.vcd");
        $dumpvars(0, tb_salsa20);
    end

    integer i;
    initial begin
        reset_n    = 0;
        cs         = 0;
        we         = 0;
        address    = 0;
        write_data = 0;
        repeat(10) @(posedge clk);
        reset_n = 1;
        @(posedge clk);

        for (i = 0; i < 200; i = i + 1) begin
            cs      = 1;
            we      = i[0];
            address = i[7:0];
            write_data = {i[7:0], i[7:0], i[7:0], i[7:0]};
            @(posedge clk);
        end
        cs = 0;
        repeat(50) @(posedge clk);
        $finish;
    end
endmodule
