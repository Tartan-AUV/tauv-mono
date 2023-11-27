`timescale 1ns / 1ps

module MCP42000_Driver
  #(parameter SPI_CLK_DOUBLE_DIV = 1000,
    parameter SCK_OFFSET = 100)
   (input  logic clk, reset_n,
    input  logic wiper_sel,
    input  logic [7:0] val,
    input  logic valid,
    output logic ready,
    output logic cs, mosi, sck);

    	/* SCK GENERATION */
	integer clk_div;
    integer sck_offset_cntr;
    logic sck_delayed_pulse;
    logic sck_offset_cntr_en;
	
	always_ff @(posedge clk)
	begin
        if (reset_n == 1'b0) begin
            sck_offset_cntr_en <= 1'b0;
            clk_div <= 32'h0;
            sck     <= 1'b0;
        end else if (clk_div == SPI_CLK_DOUBLE_DIV - 1) begin
            if (sck == 1'b0) // rising edge
                sck_offset_cntr_en <= 1'b1;
            clk_div  <= 32'h0;
            sck      <= ~sck;
        end else
            clk_div <= clk_div + 1;
	end


    always_ff @(posedge clk)
    begin
        if (reset_n == 1'b0) begin
            sck_delayed_pulse     <= 1'b0;
            sck_offset_cntr <= 32'h0;
        end else if (sck_offset_cntr == SCK_OFFSET - 1) begin
            sck_delayed_pulse   <= 1'b1;
            sck_offset_cntr     <= 32'h0;
            sck_offset_cntr_en  <= 1'b0;
        end else if (sck_offset_cntr_en == 1'b1) begin
            sck_offset_cntr     <= sck_offset_cntr + 32'h1;
            sck_delayed_pulse   <= 1'b0;
        end else begin
            sck_delayed_pulse <= 1'b0;
        end
    end
	
	/* SPI BIT-BANGING */
	
	// Control signals
	logic shift_cntr_clr,
         shift_en,
         wiper_ld;
	
    // Registers
	integer shift_cntr;
    logic [15:0] data_reg;

    // Data register loading
    always_ff @(posedge clk)
    begin
        if (reset_n == 1'b0) begin
            data_reg[15:8] <= 8'b00010000;
            data_reg[7:0]  <= 8'b0;
        end else begin
            if (ready & valid) begin
                case (wiper_sel)
                    1'b0: begin
                        data_reg[9:8] <= 2'b01;
                        data_reg[7:0] <= val;
                    end
                    1'b1: begin
                        data_reg[9:8] <= 2'b10;
                        data_reg[7:0] <= val;
                    end
                endcase
            end
        end
    end

    // Data ready flag
    logic data_ready_ff;
    always_ff @(posedge clk)
        if (~reset_n) 
            data_ready_ff <= 1'b0;
        else if (ready & valid)
            data_ready_ff <= 1'b1;
        else if (data_ready_ff & ~cs)
            data_ready_ff <= 1'b0;
	
    // Bit shifting
	always_ff @(posedge clk)
	begin
        if (reset_n == 1'b0 || shift_cntr_clr) begin
            shift_cntr <= 32'h0;
        end if (shift_en) begin
            data_reg <= data_reg << 1;
            shift_cntr <= shift_cntr + 1;
        end
	end
	
	assign mosi = data_reg[15];

	
	/* FSM */
    // Status points
    logic shift_done;
    assign shift_done = shift_cntr >= 15 ? 1'b1 : 1'b0;

    enum logic [1:0] {IDLE, SCK_WAIT, TXN0} state, nextState;

    always_ff @(posedge clk) begin
        if (reset_n == 1'b0)
            state <= IDLE;
        else
            state <= nextState;
    end
    
    always_comb begin
        nextState = state;
        cs = 1; 
        shift_cntr_clr = 0;
        shift_en = 0;
        ready = 0;
        case (state)
            IDLE: begin
                if (data_ready_ff) begin
                    nextState = SCK_WAIT;
                    ready = 0;
                end else begin
                    ready = 1;
                end
            end

            SCK_WAIT: begin
                if (sck_delayed_pulse) begin
                    nextState = TXN0;
                    cs = 0;
                    shift_cntr_clr = 1;
                end else begin
                    ready = 0;
                end
            end

            TXN0: begin
                if (sck_delayed_pulse && shift_done) begin
                    nextState = IDLE;
                end else if (sck_delayed_pulse & ~shift_done) begin
                    shift_en = 1'b1;
                    cs = 1'b0;
                end else begin
                    cs = 1'b0;
                end
            end
        endcase
    end

endmodule
