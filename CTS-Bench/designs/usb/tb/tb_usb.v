`timescale 1ns / 1ps
`default_nettype none

module tb_usb;
    reg        rst_n;
    reg        clk_48;
    reg        rx_j;
    reg        rx_se0;
    reg [6:0]  usb_address;
    reg        data_toggle;
    reg [1:0]  handshake;
    reg [7:0]  data_in;
    reg        data_in_valid;

    wire        tx_en;
    wire        tx_j;
    wire        tx_se0;
    wire        usb_rst;
    wire        transaction_active;
    wire [3:0]  endpoint;
    wire        direction_in;
    wire        setup;
    wire [7:0]  data_out;
    wire        data_strobe;
    wire        success;

    usb uut (
        .rst_n(rst_n),
        .clk_48(clk_48),
        .rx_j(rx_j),
        .rx_se0(rx_se0),
        .usb_address(usb_address),
        .usb_rst(usb_rst),
        .transaction_active(transaction_active),
        .endpoint(endpoint),
        .direction_in(direction_in),
        .setup(setup),
        .data_toggle(data_toggle),
        .handshake(handshake),
        .data_out(data_out),
        .data_in(data_in),
        .data_in_valid(data_in_valid),
        .data_strobe(data_strobe),
        .success(success),
        .tx_en(tx_en),
        .tx_j(tx_j),
        .tx_se0(tx_se0)
    );

    initial clk_48 = 0;
    always #10 clk_48 = ~clk_48;  // 50MHz approx (USB 48MHz)

    initial begin
        $dumpfile("tb_usb.vcd");
        $dumpvars(0, tb_usb);
    end

    integer i;
    initial begin
        rst_n         = 0;
        rx_j          = 1;
        rx_se0        = 0;
        usb_address   = 7'h00;
        data_toggle   = 0;
        handshake     = 2'b00;
        data_in       = 0;
        data_in_valid = 0;
        repeat(10) @(posedge clk_48);
        rst_n = 1;
        @(posedge clk_48);

        for (i = 0; i < 300; i = i + 1) begin
            rx_j          = $random;
            rx_se0        = 0;
            usb_address   = i[6:0];
            data_toggle   = i[0];
            handshake     = i[1:0];
            data_in       = $random;
            data_in_valid = i[0];
            @(posedge clk_48);
        end
        repeat(50) @(posedge clk_48);
        $finish;
    end
endmodule
