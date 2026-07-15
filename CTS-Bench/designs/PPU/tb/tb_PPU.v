`timescale 1ns / 1ps
`default_nettype none

module tb_PPU;
    reg        clk;
    reg        ce;
    reg        reset;
    reg  [7:0] din;
    reg  [2:0] ain;
    reg        read;
    reg        write;
    reg  [7:0] vram_din;

    wire [5:0]  color;
    wire [7:0]  dout;
    wire        nmi;
    wire        vram_r;
    wire        vram_w;
    wire [13:0] vram_a;
    wire [7:0]  vram_dout;
    wire [8:0]  scanline;
    wire [8:0]  cycle;
    wire [19:0] mapper_ppu_flags;

    PPU uut (
        .clk(clk), .ce(ce), .reset(reset),
        .color(color),
        .din(din), .dout(dout),
        .ain(ain), .read(read), .write(write),
        .nmi(nmi),
        .vram_r(vram_r), .vram_w(vram_w),
        .vram_a(vram_a), .vram_din(vram_din), .vram_dout(vram_dout),
        .scanline(scanline), .cycle(cycle),
        .mapper_ppu_flags(mapper_ppu_flags)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    initial begin
        $dumpfile("tb_PPU.vcd");
        $dumpvars(0, tb_PPU);
    end

    integer i;
    initial begin
        reset    = 1;
        ce       = 1;
        din      = 0;
        ain      = 0;
        read     = 0;
        write    = 0;
        vram_din = 0;
        repeat(10) @(posedge clk);
        reset = 0;
        @(posedge clk);

        for (i = 0; i < 500; i = i + 1) begin
            ce       = 1;
            din      = $random;
            ain      = $random;
            read     = i[0];
            write    = ~i[0];
            vram_din = $random;
            @(posedge clk);
        end
        repeat(50) @(posedge clk);
        $finish;
    end
endmodule
