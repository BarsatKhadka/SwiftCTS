`timescale 1 ns / 1 ps

module testbench;
    // ============================================================
    // 1. CONFIGURATION
    // ============================================================
    // 32KB Memory (8192 words)
    parameter MEM_SIZE = 8192; 
    
    reg clk = 1;
    reg resetn = 0;
    wire trap;

    // 100MHz Clock (10ns period)
    always #5 clk = ~clk; 

    // Loop variables declared OUTSIDE initial blocks (Verilog-2005 safe)
    integer i;
    integer cycle_count = 0;

    // ============================================================
    // 2. SIMULATION CONTROL
    // ============================================================
    initial begin
        // A. Setup VCD Dump (Captures signals for SAIF)
        if ($test$plusargs("vcd")) begin
            $dumpfile("testbench.vcd");
            $dumpvars(0, testbench);
        end

        // B. Reset Sequence
        repeat (10) @(posedge clk);
        resetn <= 1;
        $display("--- SIMULATION STARTED ---");

        // C. Failsafe Timeout
        // 20,000 cycles limit (Prevents infinite hangs)
        repeat (20000) @(posedge clk);
        
        $display("\n[TIMEOUT] Simulation ran too long!");
        $display("Total Cycles: %0d", cycle_count);
        $finish;
    end

    // Cycle Counter & Heartbeat
    always @(posedge clk) begin
        if (resetn) begin
            cycle_count <= cycle_count + 1;
            // Print status every 1000 cycles so you know it's alive
            if (cycle_count % 1000 == 0) 
                $display("[Running] Cycle: %0d ...", cycle_count);
        end
    end

    // Trap Detection (Success/Crash Signal)
    always @(posedge clk) begin
        if (resetn && trap) begin
            $display("\n[TRAP] Processor halted (Program Finished).");
            $display("Total Cycles: %0d", cycle_count);
            $finish;
        end
    end

    // ============================================================
    // 3. PICORV32 CORE INSTANTIATION
    // ============================================================
    wire mem_valid;
    wire mem_instr;
    reg mem_ready;
    wire [31:0] mem_addr;
    wire [31:0] mem_wdata;
    wire [3:0] mem_wstrb;
    reg  [31:0] mem_rdata;

    // CRITICAL FIX: No parameters #() passed to the Netlist!
    picorv32 uut (
        .clk         (clk        ),
        .resetn      (resetn     ),
        .trap        (trap       ),
        .mem_valid   (mem_valid  ),
        .mem_instr   (mem_instr  ),
        .mem_ready   (mem_ready  ),
        .mem_addr    (mem_addr   ),
        .mem_wdata   (mem_wdata  ),
        .mem_wstrb   (mem_wstrb  ),
        .mem_rdata   (mem_rdata  ),
        .irq         (32'b0      )
    );

    // ============================================================
    // 4. ROBUST MEMORY MODEL (WITH CLEAN RESET)
    // ============================================================
    reg [31:0] memory [0:MEM_SIZE-1];
    reg [1023:0] firmware_file;

    initial begin
        // 1. Zero out memory
        for (i=0; i<MEM_SIZE; i=i+1) memory[i] = 0;

        // 2. Load Firmware
        if ($value$plusargs("firmware=%s", firmware_file)) begin
            $display("[INFO] Loading firmware: %s", firmware_file);
            $readmemh(firmware_file, memory);
        end else begin
            $display("[WARNING] No firmware loaded. Expecting infinite loop at 0x0.");
            memory[0] = 32'h 0000006f; // j .
        end
    end

    // Memory Logic: Registered Response (1 cycle latency)
    // Includes CLEAN RESET logic to prevent X-Propagation.
    always @(posedge clk) begin
        // ----------------------------------------------------
        // CRITICAL FIX: FORCE ZERO DURING RESET
        // ----------------------------------------------------
        if (!resetn) begin
            mem_ready <= 0;
            mem_rdata <= 32'b0; // Force pure zeros (NOPs)
        end 
        
        // ----------------------------------------------------
        // NORMAL OPERATION
        // ----------------------------------------------------
        else begin
            // Default: Not ready
            mem_ready <= 0;

            if (mem_valid && !mem_ready) begin
                // A. MEMORY MAPPED IO
                // -------------------
                // 1. Output Char (0x10000000)
                if (mem_addr == 32'h10000000 && mem_wstrb) begin
                    $write("%c", mem_wdata[7:0]); // Print to console
                    mem_ready <= 1;
                end
                
                // 2. Pass/Fail Check (0x20000000)
                else if (mem_addr == 32'h20000000 && mem_wstrb) begin
                    if (mem_wdata == 123456789) 
                        $display("\n[SUCCESS] Firmware signaled PASS.");
                    else 
                        $display("\n[FAIL] Firmware signaled FAIL (Code: %d).", mem_wdata);
                    mem_ready <= 1;
                end

                // B. STANDARD RAM (0x00000000 - Limit)
                // ------------------------------------
                else if (mem_addr < (MEM_SIZE*4)) begin
                    mem_ready <= 1;
                    mem_rdata <= memory[mem_addr >> 2];
                    
                    // Write Logic
                    if (mem_wstrb[0]) memory[mem_addr >> 2][ 7: 0] <= mem_wdata[ 7: 0];
                    if (mem_wstrb[1]) memory[mem_addr >> 2][15: 8] <= mem_wdata[15: 8];
                    if (mem_wstrb[2]) memory[mem_addr >> 2][23:16] <= mem_wdata[23:16];
                    if (mem_wstrb[3]) memory[mem_addr >> 2][31:24] <= mem_wdata[31:24];
                end
                
                // C. OUT OF BOUNDS
                // ----------------
                else begin
                    mem_ready <= 1;
                    mem_rdata <= 32'h 00000000;
                end
            end
        end
    end

endmodule