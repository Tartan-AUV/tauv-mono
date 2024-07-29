// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2024.1.1 (lin64) Build 5094488 Fri Jun 14 08:57:50 MDT 2024
// Date        : Sun Jul 28 21:19:58 2024
// Host        : fedora running 64-bit unknown
// Command     : write_verilog -force -mode synth_stub
//               /home/gleb/dev-shared/TAUV-ROS-Packages/firmware/AcousticsV2/HIL/ZyboStandalone/Acoustics_V2_HIL_Zybo_Standalone.gen/sources_1/bd/top/bd/RxEmulator_inst_0/ip/RxEmulator_inst_0_axis_broadcaster_0_0/RxEmulator_inst_0_axis_broadcaster_0_0_stub.v
// Design      : RxEmulator_inst_0_axis_broadcaster_0_0
// Purpose     : Stub declaration of top-level module interface
// Device      : xc7z020clg400-1
// --------------------------------------------------------------------------------

// This empty module with port declaration file causes synthesis tools to infer a black box for IP.
// The synthesis directives are for Synopsys Synplify support to prevent IO buffer insertion.
// Please paste the declaration into a Verilog source file or add the file as an additional source.
(* X_CORE_INFO = "top_RxEmulator_inst_0_axis_broadcaster_0_0,Vivado 2024.1.1" *)
module RxEmulator_inst_0_axis_broadcaster_0_0(aclk, aresetn, s_axis_tvalid, s_axis_tdata, 
  m_axis_tvalid, m_axis_tdata)
/* synthesis syn_black_box black_box_pad_pin="aclk,aresetn,s_axis_tvalid,s_axis_tdata[63:0],m_axis_tvalid[4:0],m_axis_tdata[79:0]" */;
  input aclk;
  input aresetn;
  input s_axis_tvalid;
  input [63:0]s_axis_tdata;
  output [4:0]m_axis_tvalid;
  output [79:0]m_axis_tdata;
endmodule
