-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
-- --------------------------------------------------------------------------------
-- Tool Version: Vivado v.2024.1.1 (lin64) Build 5094488 Fri Jun 14 08:57:50 MDT 2024
-- Date        : Sun Jul 28 21:19:58 2024
-- Host        : fedora running 64-bit unknown
-- Command     : write_vhdl -force -mode synth_stub
--               /home/gleb/dev-shared/TAUV-ROS-Packages/firmware/AcousticsV2/HIL/ZyboStandalone/Acoustics_V2_HIL_Zybo_Standalone.gen/sources_1/bd/top/bd/RxEmulator_inst_0/ip/RxEmulator_inst_0_axis_broadcaster_0_0/RxEmulator_inst_0_axis_broadcaster_0_0_stub.vhdl
-- Design      : RxEmulator_inst_0_axis_broadcaster_0_0
-- Purpose     : Stub declaration of top-level module interface
-- Device      : xc7z020clg400-1
-- --------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity RxEmulator_inst_0_axis_broadcaster_0_0 is
  Port ( 
    aclk : in STD_LOGIC;
    aresetn : in STD_LOGIC;
    s_axis_tvalid : in STD_LOGIC;
    s_axis_tdata : in STD_LOGIC_VECTOR ( 63 downto 0 );
    m_axis_tvalid : out STD_LOGIC_VECTOR ( 4 downto 0 );
    m_axis_tdata : out STD_LOGIC_VECTOR ( 79 downto 0 )
  );

end RxEmulator_inst_0_axis_broadcaster_0_0;

architecture stub of RxEmulator_inst_0_axis_broadcaster_0_0 is
attribute syn_black_box : boolean;
attribute black_box_pad_pin : string;
attribute syn_black_box of stub : architecture is true;
attribute black_box_pad_pin of stub : architecture is "aclk,aresetn,s_axis_tvalid,s_axis_tdata[63:0],m_axis_tvalid[4:0],m_axis_tdata[79:0]";
attribute X_CORE_INFO : string;
attribute X_CORE_INFO of stub : architecture is "top_RxEmulator_inst_0_axis_broadcaster_0_0,Vivado 2024.1.1";
begin
end;
