// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2024.1.1 (lin64) Build 5094488 Fri Jun 14 08:57:50 MDT 2024
// Date        : Sun Jul 28 21:19:58 2024
// Host        : fedora running 64-bit unknown
// Command     : write_verilog -force -mode synth_stub
//               /home/gleb/dev-shared/TAUV-ROS-Packages/firmware/AcousticsV2/HIL/ZyboStandalone/Acoustics_V2_HIL_Zybo_Standalone.gen/sources_1/bd/top/ip/top_clk_wiz_0_0/top_clk_wiz_0_0_stub.v
// Design      : top_clk_wiz_0_0
// Purpose     : Stub declaration of top-level module interface
// Device      : xc7z020clg400-1
// --------------------------------------------------------------------------------

// This empty module with port declaration file causes synthesis tools to infer a black box for IP.
// The synthesis directives are for Synopsys Synplify support to prevent IO buffer insertion.
// Please paste the declaration into a Verilog source file or add the file as an additional source.
module top_clk_wiz_0_0(clk_out1, resetn, locked, clk_in1)
/* synthesis syn_black_box black_box_pad_pin="resetn,locked,clk_in1" */
/* synthesis syn_force_seq_prim="clk_out1" */;
  output clk_out1 /* synthesis syn_isclock = 1 */;
  input resetn;
  output locked;
  input clk_in1;
endmodule
