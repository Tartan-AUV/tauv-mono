//Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
//Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
//--------------------------------------------------------------------------------
//Tool Version: Vivado v.2024.1.1 (lin64) Build 5094488 Fri Jun 14 08:57:50 MDT 2024
//Date        : Sun Jul 28 21:19:24 2024
//Host        : fedora running 64-bit unknown
//Command     : generate_target RxEmulator_inst_0.bd
//Design      : RxEmulator_inst_0
//Purpose     : IP block netlist
//--------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

/* Make clock?
 */
(* CORE_GENERATION_INFO = "RxEmulator_inst_0,IP_Integrator,{x_ipVendor=xilinx.com,x_ipLibrary=BlockDiagram,x_ipName=RxEmulator_inst_0,x_ipVersion=1.00.a,x_ipLanguage=VERILOG,numBlks=3,numReposBlks=3,numNonXlnxBlks=0,numHierBlks=0,maxHierDepth=0,numSysgenBlks=0,numHlsBlks=0,numHdlrefBlks=0,numPkgbdBlks=0,bdsource=/home/gleb/dev-shared/TAUV-ROS-Packages/firmware/AcousticsV2/HIL/ZyboStandalone/Acoustics_V2_HIL_Zybo_Standalone.srcs/sources_1/bd/RxEmulator/RxEmulator.bd,synth_mode=Hierarchical}" *) (* HW_HANDOFF = "RxEmulator_inst_0.hwdef" *) 
module RxEmulator_inst_0
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
  (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 RST.RESET RST" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME RST.RESET, INSERT_VIP 0, POLARITY ACTIVE_LOW" *) input Reset;
  (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 CLK.S_AXI_CLK CLK" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME CLK.S_AXI_CLK, ASSOCIATED_BUSIF S_AXI_LITE, ASSOCIATED_RESET Reset, CLK_DOMAIN top_processing_system7_0_0_FCLK_CLK0, FREQ_HZ 50000000, FREQ_TOLERANCE_HZ 0, INSERT_VIP 0, PHASE 0.0" *) input S_AXI_CLK;
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_LITE ARADDR" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME S_AXI_LITE, ADDR_WIDTH 32, ARUSER_WIDTH 0, AWUSER_WIDTH 0, BUSER_WIDTH 0, CLK_DOMAIN top_processing_system7_0_0_FCLK_CLK0, DATA_WIDTH 32, FREQ_HZ 50000000, HAS_BRESP 1, HAS_BURST 1, HAS_CACHE 1, HAS_LOCK 1, HAS_PROT 1, HAS_QOS 1, HAS_REGION 0, HAS_RRESP 1, HAS_WSTRB 1, ID_WIDTH 0, INSERT_VIP 0, MAX_BURST_LENGTH 1, NUM_READ_OUTSTANDING 8, NUM_READ_THREADS 4, NUM_WRITE_OUTSTANDING 8, NUM_WRITE_THREADS 4, PHASE 0.0, PROTOCOL AXI4LITE, READ_WRITE_MODE READ_WRITE, RUSER_BITS_PER_BYTE 0, RUSER_WIDTH 0, SUPPORTS_NARROW_BURST 0, WUSER_BITS_PER_BYTE 0, WUSER_WIDTH 0" *) input [31:0]S_AXI_LITE_araddr;
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_LITE ARREADY" *) output S_AXI_LITE_arready;
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_LITE ARVALID" *) input S_AXI_LITE_arvalid;
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_LITE AWADDR" *) input [31:0]S_AXI_LITE_awaddr;
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_LITE AWREADY" *) output S_AXI_LITE_awready;
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_LITE AWVALID" *) input S_AXI_LITE_awvalid;
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_LITE BREADY" *) input S_AXI_LITE_bready;
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_LITE BRESP" *) output [1:0]S_AXI_LITE_bresp;
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_LITE BVALID" *) output S_AXI_LITE_bvalid;
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_LITE RDATA" *) output [31:0]S_AXI_LITE_rdata;
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_LITE RREADY" *) input S_AXI_LITE_rready;
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_LITE RRESP" *) output [1:0]S_AXI_LITE_rresp;
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_LITE RVALID" *) output S_AXI_LITE_rvalid;
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_LITE WDATA" *) input [31:0]S_AXI_LITE_wdata;
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_LITE WREADY" *) output S_AXI_LITE_wready;
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_LITE WVALID" *) input S_AXI_LITE_wvalid;

  wire Net;
  wire Reset_1;
  wire S_AXI_CLK_1;
  wire [31:0]S_AXI_LITE_1_ARADDR;
  wire S_AXI_LITE_1_ARREADY;
  wire S_AXI_LITE_1_ARVALID;
  wire [31:0]S_AXI_LITE_1_AWADDR;
  wire S_AXI_LITE_1_AWREADY;
  wire S_AXI_LITE_1_AWVALID;
  wire S_AXI_LITE_1_BREADY;
  wire [1:0]S_AXI_LITE_1_BRESP;
  wire S_AXI_LITE_1_BVALID;
  wire [31:0]S_AXI_LITE_1_RDATA;
  wire S_AXI_LITE_1_RREADY;
  wire [1:0]S_AXI_LITE_1_RRESP;
  wire S_AXI_LITE_1_RVALID;
  wire [31:0]S_AXI_LITE_1_WDATA;
  wire S_AXI_LITE_1_WREADY;
  wire S_AXI_LITE_1_WVALID;
  wire [63:0]axi_dma_0_M_AXIS_MM2S_TDATA;
  wire axi_dma_0_M_AXIS_MM2S_TVALID;
  wire [15:0]axis_broadcaster_0_M00_AXIS_TDATA;
  wire [0:0]axis_broadcaster_0_M00_AXIS_TVALID;

  assign Net = DMA_MM2S_ACLK;
  assign Reset_1 = Reset;
  assign S_AXI_CLK_1 = S_AXI_CLK;
  assign S_AXI_LITE_1_ARADDR = S_AXI_LITE_araddr[31:0];
  assign S_AXI_LITE_1_ARVALID = S_AXI_LITE_arvalid;
  assign S_AXI_LITE_1_AWADDR = S_AXI_LITE_awaddr[31:0];
  assign S_AXI_LITE_1_AWVALID = S_AXI_LITE_awvalid;
  assign S_AXI_LITE_1_BREADY = S_AXI_LITE_bready;
  assign S_AXI_LITE_1_RREADY = S_AXI_LITE_rready;
  assign S_AXI_LITE_1_WDATA = S_AXI_LITE_wdata[31:0];
  assign S_AXI_LITE_1_WVALID = S_AXI_LITE_wvalid;
  assign S_AXI_LITE_arready = S_AXI_LITE_1_ARREADY;
  assign S_AXI_LITE_awready = S_AXI_LITE_1_AWREADY;
  assign S_AXI_LITE_bresp[1:0] = S_AXI_LITE_1_BRESP;
  assign S_AXI_LITE_bvalid = S_AXI_LITE_1_BVALID;
  assign S_AXI_LITE_rdata[31:0] = S_AXI_LITE_1_RDATA;
  assign S_AXI_LITE_rresp[1:0] = S_AXI_LITE_1_RRESP;
  assign S_AXI_LITE_rvalid = S_AXI_LITE_1_RVALID;
  assign S_AXI_LITE_wready = S_AXI_LITE_1_WREADY;
  RxEmulator_inst_0_axi_dma_0_0 axi_dma_0
       (.axi_resetn(Reset_1),
        .m_axi_mm2s_aclk(Net),
        .m_axi_mm2s_arready(1'b0),
        .m_axi_mm2s_rdata({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .m_axi_mm2s_rlast(1'b0),
        .m_axi_mm2s_rresp({1'b0,1'b0}),
        .m_axi_mm2s_rvalid(1'b0),
        .m_axis_mm2s_tdata(axi_dma_0_M_AXIS_MM2S_TDATA),
        .m_axis_mm2s_tready(1'b1),
        .m_axis_mm2s_tvalid(axi_dma_0_M_AXIS_MM2S_TVALID),
        .s_axi_lite_aclk(S_AXI_CLK_1),
        .s_axi_lite_araddr(S_AXI_LITE_1_ARADDR[9:0]),
        .s_axi_lite_arready(S_AXI_LITE_1_ARREADY),
        .s_axi_lite_arvalid(S_AXI_LITE_1_ARVALID),
        .s_axi_lite_awaddr(S_AXI_LITE_1_AWADDR[9:0]),
        .s_axi_lite_awready(S_AXI_LITE_1_AWREADY),
        .s_axi_lite_awvalid(S_AXI_LITE_1_AWVALID),
        .s_axi_lite_bready(S_AXI_LITE_1_BREADY),
        .s_axi_lite_bresp(S_AXI_LITE_1_BRESP),
        .s_axi_lite_bvalid(S_AXI_LITE_1_BVALID),
        .s_axi_lite_rdata(S_AXI_LITE_1_RDATA),
        .s_axi_lite_rready(S_AXI_LITE_1_RREADY),
        .s_axi_lite_rresp(S_AXI_LITE_1_RRESP),
        .s_axi_lite_rvalid(S_AXI_LITE_1_RVALID),
        .s_axi_lite_wdata(S_AXI_LITE_1_WDATA),
        .s_axi_lite_wready(S_AXI_LITE_1_WREADY),
        .s_axi_lite_wvalid(S_AXI_LITE_1_WVALID));
  RxEmulator_inst_0_axis_broadcaster_0_0 axis_broadcaster_0
       (.aclk(Net),
        .aresetn(Reset_1),
        .m_axis_tdata(axis_broadcaster_0_M00_AXIS_TDATA),
        .m_axis_tvalid(axis_broadcaster_0_M00_AXIS_TVALID),
        .s_axis_tdata(axi_dma_0_M_AXIS_MM2S_TDATA),
        .s_axis_tvalid(axi_dma_0_M_AXIS_MM2S_TVALID));
  RxEmulator_inst_0_ila_0_0 ila_0
       (.clk(Net),
        .probe0(1'b1),
        .probe1(axis_broadcaster_0_M00_AXIS_TDATA),
        .probe2({1'b1,1'b1}),
        .probe3(axis_broadcaster_0_M00_AXIS_TVALID),
        .probe4(1'b0),
        .probe5(1'b0),
        .probe6({1'b1,1'b1}),
        .probe7(1'b0),
        .probe8(1'b0));
endmodule
