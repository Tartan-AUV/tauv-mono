//Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
//Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
//--------------------------------------------------------------------------------
//Tool Version: Vivado v.2024.1.1 (lin64) Build 5094488 Fri Jun 14 08:57:50 MDT 2024
//Date        : Sun Jul 28 21:19:24 2024
//Host        : fedora running 64-bit unknown
//Command     : generate_target RxEmulator_inst_0_wrapper.bd
//Design      : RxEmulator_inst_0_wrapper
//Purpose     : IP block netlist
//--------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

module RxEmulator_inst_0_wrapper
   (DMA_MM2S_ACLK,
    Reset,
    S_AXI_CLK,
    S_AXI_LITE_araddr,
    S_AXI_LITE_arready,
    S_AXI_LITE_arvalid,
    S_AXI_LITE_awaddr,
    S_AXI_LITE_awready,
    S_AXI_LITE_awvalid,
    S_AXI_LITE_bready,
    S_AXI_LITE_bresp,
    S_AXI_LITE_bvalid,
    S_AXI_LITE_rdata,
    S_AXI_LITE_rready,
    S_AXI_LITE_rresp,
    S_AXI_LITE_rvalid,
    S_AXI_LITE_wdata,
    S_AXI_LITE_wready,
    S_AXI_LITE_wvalid);
  input DMA_MM2S_ACLK;
  input Reset;
  input S_AXI_CLK;
  input [31:0]S_AXI_LITE_araddr;
  output S_AXI_LITE_arready;
  input S_AXI_LITE_arvalid;
  input [31:0]S_AXI_LITE_awaddr;
  output S_AXI_LITE_awready;
  input S_AXI_LITE_awvalid;
  input S_AXI_LITE_bready;
  output [1:0]S_AXI_LITE_bresp;
  output S_AXI_LITE_bvalid;
  output [31:0]S_AXI_LITE_rdata;
  input S_AXI_LITE_rready;
  output [1:0]S_AXI_LITE_rresp;
  output S_AXI_LITE_rvalid;
  input [31:0]S_AXI_LITE_wdata;
  output S_AXI_LITE_wready;
  input S_AXI_LITE_wvalid;

  wire DMA_MM2S_ACLK;
  wire Reset;
  wire S_AXI_CLK;
  wire [31:0]S_AXI_LITE_araddr;
  wire S_AXI_LITE_arready;
  wire S_AXI_LITE_arvalid;
  wire [31:0]S_AXI_LITE_awaddr;
  wire S_AXI_LITE_awready;
  wire S_AXI_LITE_awvalid;
  wire S_AXI_LITE_bready;
  wire [1:0]S_AXI_LITE_bresp;
  wire S_AXI_LITE_bvalid;
  wire [31:0]S_AXI_LITE_rdata;
  wire S_AXI_LITE_rready;
  wire [1:0]S_AXI_LITE_rresp;
  wire S_AXI_LITE_rvalid;
  wire [31:0]S_AXI_LITE_wdata;
  wire S_AXI_LITE_wready;
  wire S_AXI_LITE_wvalid;

  RxEmulator_inst_0 RxEmulator_inst_0_i
       (.DMA_MM2S_ACLK(DMA_MM2S_ACLK),
        .Reset(Reset),
        .S_AXI_CLK(S_AXI_CLK),
        .S_AXI_LITE_araddr(S_AXI_LITE_araddr),
        .S_AXI_LITE_arready(S_AXI_LITE_arready),
        .S_AXI_LITE_arvalid(S_AXI_LITE_arvalid),
        .S_AXI_LITE_awaddr(S_AXI_LITE_awaddr),
        .S_AXI_LITE_awready(S_AXI_LITE_awready),
        .S_AXI_LITE_awvalid(S_AXI_LITE_awvalid),
        .S_AXI_LITE_bready(S_AXI_LITE_bready),
        .S_AXI_LITE_bresp(S_AXI_LITE_bresp),
        .S_AXI_LITE_bvalid(S_AXI_LITE_bvalid),
        .S_AXI_LITE_rdata(S_AXI_LITE_rdata),
        .S_AXI_LITE_rready(S_AXI_LITE_rready),
        .S_AXI_LITE_rresp(S_AXI_LITE_rresp),
        .S_AXI_LITE_rvalid(S_AXI_LITE_rvalid),
        .S_AXI_LITE_wdata(S_AXI_LITE_wdata),
        .S_AXI_LITE_wready(S_AXI_LITE_wready),
        .S_AXI_LITE_wvalid(S_AXI_LITE_wvalid));
endmodule
