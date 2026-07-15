// Behavioral replacement for Xilinx RAM16X1D-based dual-port RAM.
// Interface identical to original; SPO output (same-port read) tied to 0.
module pci_ram_16x40d (data_out, we, data_in, read_address, write_address, wclk);
    parameter addr_width = 4;
    output reg [39:0] data_out;
    input              we, wclk;
    input      [39:0]  data_in;
    input      [addr_width-1:0] write_address, read_address;

    reg [39:0] mem [0:15];

    always @(posedge wclk)
        if (we) mem[write_address] <= data_in;

    always @(*) data_out = mem[read_address];

endmodule
