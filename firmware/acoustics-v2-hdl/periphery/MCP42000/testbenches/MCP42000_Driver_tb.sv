`timescale 1ns/1ps

module MCP42000_Driver_tb
   ();

    logic clk, reset_n;

    logic wiper_sel, valid, ready, cs, mosi, sck;
    logic [7:0] val;

    MCP42000_Driver dut(.*);

    initial begin
        clk <= 0;
        forever #10 clk = ~clk;
    end

    initial begin
        @(posedge clk)
        reset_n <= 0;
        @(posedge clk)
        reset_n <= 1;
        wait (ready);
        @(posedge clk);
        val <= 8'd42;
        wiper_sel <= 0;
        valid <= 1;
        @(posedge clk);
        valid <= 0;
        @(posedge clk);
        @(posedge clk);
        @(posedge clk);
                
        wait (ready);
        @(posedge clk);
        val <= 8'd73;
        wiper_sel <= 1;
        valid <= 1;
        @(posedge clk);
        valid <= 0;
        @(posedge clk);
        @(posedge clk);

        wait (ready);
        $finish;
    end

endmodule