`timescale 1ns / 1ps
`default_nettype none

module tb_y_huff;
    reg        clk;
    reg        rst;
    reg        enable;
    reg [10:0] Y11, Y12, Y13, Y14, Y15, Y16, Y17, Y18;
    reg [10:0] Y21, Y22, Y23, Y24, Y25, Y26, Y27, Y28;
    reg [10:0] Y31, Y32, Y33, Y34, Y35, Y36, Y37, Y38;
    reg [10:0] Y41, Y42, Y43, Y44, Y45, Y46, Y47, Y48;
    reg [10:0] Y51, Y52, Y53, Y54, Y55, Y56, Y57, Y58;
    reg [10:0] Y61, Y62, Y63, Y64, Y65, Y66, Y67, Y68;
    reg [10:0] Y71, Y72, Y73, Y74, Y75, Y76, Y77, Y78;
    reg [10:0] Y81, Y82, Y83, Y84, Y85, Y86, Y87, Y88;

    wire [31:0] JPEG_bitstream;
    wire        data_ready;
    wire [4:0]  output_reg_count;
    wire        end_of_block_output;
    wire        end_of_block_empty;

    y_huff uut (
        .clk(clk), .rst(rst), .enable(enable),
        .Y11(Y11), .Y12(Y12), .Y13(Y13), .Y14(Y14), .Y15(Y15), .Y16(Y16), .Y17(Y17), .Y18(Y18),
        .Y21(Y21), .Y22(Y22), .Y23(Y23), .Y24(Y24), .Y25(Y25), .Y26(Y26), .Y27(Y27), .Y28(Y28),
        .Y31(Y31), .Y32(Y32), .Y33(Y33), .Y34(Y34), .Y35(Y35), .Y36(Y36), .Y37(Y37), .Y38(Y38),
        .Y41(Y41), .Y42(Y42), .Y43(Y43), .Y44(Y44), .Y45(Y45), .Y46(Y46), .Y47(Y47), .Y48(Y48),
        .Y51(Y51), .Y52(Y52), .Y53(Y53), .Y54(Y54), .Y55(Y55), .Y56(Y56), .Y57(Y57), .Y58(Y58),
        .Y61(Y61), .Y62(Y62), .Y63(Y63), .Y64(Y64), .Y65(Y65), .Y66(Y66), .Y67(Y67), .Y68(Y68),
        .Y71(Y71), .Y72(Y72), .Y73(Y73), .Y74(Y74), .Y75(Y75), .Y76(Y76), .Y77(Y77), .Y78(Y78),
        .Y81(Y81), .Y82(Y82), .Y83(Y83), .Y84(Y84), .Y85(Y85), .Y86(Y86), .Y87(Y87), .Y88(Y88),
        .JPEG_bitstream(JPEG_bitstream),
        .data_ready(data_ready),
        .output_reg_count(output_reg_count),
        .end_of_block_output(end_of_block_output),
        .end_of_block_empty(end_of_block_empty)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    initial begin
        $dumpfile("tb_y_huff.vcd");
        $dumpvars(0, tb_y_huff);
    end

    integer i;
    initial begin
        rst    = 1;
        enable = 0;
        {Y11,Y12,Y13,Y14,Y15,Y16,Y17,Y18} = 0;
        {Y21,Y22,Y23,Y24,Y25,Y26,Y27,Y28} = 0;
        {Y31,Y32,Y33,Y34,Y35,Y36,Y37,Y38} = 0;
        {Y41,Y42,Y43,Y44,Y45,Y46,Y47,Y48} = 0;
        {Y51,Y52,Y53,Y54,Y55,Y56,Y57,Y58} = 0;
        {Y61,Y62,Y63,Y64,Y65,Y66,Y67,Y68} = 0;
        {Y71,Y72,Y73,Y74,Y75,Y76,Y77,Y78} = 0;
        {Y81,Y82,Y83,Y84,Y85,Y86,Y87,Y88} = 0;
        repeat(10) @(posedge clk);
        rst = 0;
        @(posedge clk);

        for (i = 0; i < 30; i = i + 1) begin
            enable = 1;
            Y11 = $random; Y12 = $random; Y13 = $random; Y14 = $random;
            Y21 = $random; Y22 = $random; Y23 = $random; Y24 = $random;
            Y31 = $random; Y32 = $random; Y33 = $random; Y34 = $random;
            Y41 = $random; Y42 = $random; Y43 = $random; Y44 = $random;
            Y51 = $random; Y52 = $random; Y53 = $random; Y54 = $random;
            Y61 = $random; Y62 = $random; Y63 = $random; Y64 = $random;
            Y71 = $random; Y72 = $random; Y73 = $random; Y74 = $random;
            Y81 = $random; Y82 = $random; Y83 = $random; Y84 = $random;
            repeat(10) @(posedge clk);
        end
        enable = 0;
        repeat(50) @(posedge clk);
        $finish;
    end
endmodule
