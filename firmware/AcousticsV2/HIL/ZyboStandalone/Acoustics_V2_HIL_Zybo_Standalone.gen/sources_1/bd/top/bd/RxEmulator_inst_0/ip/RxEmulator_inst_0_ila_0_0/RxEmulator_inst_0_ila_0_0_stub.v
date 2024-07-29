// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2024.1.1 (lin64) Build 5094488 Fri Jun 14 08:57:50 MDT 2024
// Date        : Sun Jul 28 21:20:19 2024
// Host        : fedora running 64-bit unknown
// Command     : write_verilog -force -mode synth_stub
//               /home/gleb/dev-shared/TAUV-ROS-Packages/firmware/AcousticsV2/HIL/ZyboStandalone/Acoustics_V2_HIL_Zybo_Standalone.gen/sources_1/bd/top/bd/RxEmulator_inst_0/ip/RxEmulator_inst_0_ila_0_0/RxEmulator_inst_0_ila_0_0_stub.v
// Design      : RxEmulator_inst_0_ila_0_0
// Purpose     : Stub declaration of top-level module interface
// Device      : xc7z020clg400-1
// --------------------------------------------------------------------------------

// This empty module with port declaration file causes synthesis tools to infer a black box for IP.
// The synthesis directives are for Synopsys Synplify support to prevent IO buffer insertion.
// Please paste the declaration into a Verilog source file or add the file as an additional source.
(* X_CORE_INFO = "ila,Vivado 2024.1.1" *)
module RxEmulator_inst_0_ila_0_0(clk, probe0, probe1, probe2, probe3, probe4, probe5, 
  probe6, probe7, probe8)
/* synthesis syn_black_box black_box_pad_pin="probe0[0:0],probe1[15:0],probe2[1:0],probe3[0:0],probe4[0:0],probe5[0:0],probe6[1:0],probe7[0:0],probe8[0:0]" */
/* synthesis syn_force_seq_prim="clk" */;
  input clk /* synthesis syn_isclock = 1 */;
  input [0:0]probe0;
  input [15:0]probe1;
  input [1:0]probe2;
  input [0:0]probe3;
  input [0:0]probe4;
  input [0:0]probe5;
  input [1:0]probe6;
  input [0:0]probe7;
  input [0:0]probe8;
endmodule
