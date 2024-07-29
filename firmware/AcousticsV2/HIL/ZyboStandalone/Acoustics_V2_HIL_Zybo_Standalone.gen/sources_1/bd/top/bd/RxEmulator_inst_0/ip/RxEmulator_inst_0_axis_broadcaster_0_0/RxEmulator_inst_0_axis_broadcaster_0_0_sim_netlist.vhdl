-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
-- --------------------------------------------------------------------------------
-- Tool Version: Vivado v.2024.1.1 (lin64) Build 5094488 Fri Jun 14 08:57:50 MDT 2024
-- Date        : Sun Jul 28 21:19:58 2024
-- Host        : fedora running 64-bit unknown
-- Command     : write_vhdl -force -mode funcsim
--               /home/gleb/dev-shared/TAUV-ROS-Packages/firmware/AcousticsV2/HIL/ZyboStandalone/Acoustics_V2_HIL_Zybo_Standalone.gen/sources_1/bd/top/bd/RxEmulator_inst_0/ip/RxEmulator_inst_0_axis_broadcaster_0_0/RxEmulator_inst_0_axis_broadcaster_0_0_sim_netlist.vhdl
-- Design      : RxEmulator_inst_0_axis_broadcaster_0_0
-- Purpose     : This VHDL netlist is a functional simulation representation of the design and should not be modified or
--               synthesized. This netlist cannot be used for SDF annotated simulation.
-- Device      : xc7z020clg400-1
-- --------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity RxEmulator_inst_0_axis_broadcaster_0_0_top_RxEmulator_inst_0_axis_broadcaster_0_0 is
  port (
    aclk : in STD_LOGIC;
    aresetn : in STD_LOGIC;
    aclken : in STD_LOGIC;
    s_axis_tvalid : in STD_LOGIC;
    s_axis_tready : out STD_LOGIC;
    s_axis_tdata : in STD_LOGIC_VECTOR ( 63 downto 0 );
    s_axis_tstrb : in STD_LOGIC_VECTOR ( 7 downto 0 );
    s_axis_tkeep : in STD_LOGIC_VECTOR ( 7 downto 0 );
    s_axis_tlast : in STD_LOGIC;
    s_axis_tid : in STD_LOGIC_VECTOR ( 0 to 0 );
    s_axis_tdest : in STD_LOGIC_VECTOR ( 0 to 0 );
    s_axis_tuser : in STD_LOGIC_VECTOR ( 0 to 0 );
    m_axis_tvalid : out STD_LOGIC_VECTOR ( 4 downto 0 );
    m_axis_tready : in STD_LOGIC_VECTOR ( 4 downto 0 );
    m_axis_tdata : out STD_LOGIC_VECTOR ( 79 downto 0 );
    m_axis_tstrb : out STD_LOGIC_VECTOR ( 9 downto 0 );
    m_axis_tkeep : out STD_LOGIC_VECTOR ( 9 downto 0 );
    m_axis_tlast : out STD_LOGIC_VECTOR ( 4 downto 0 );
    m_axis_tid : out STD_LOGIC_VECTOR ( 4 downto 0 );
    m_axis_tdest : out STD_LOGIC_VECTOR ( 4 downto 0 );
    m_axis_tuser : out STD_LOGIC_VECTOR ( 4 downto 0 )
  );
  attribute C_AXIS_SIGNAL_SET : integer;
  attribute C_AXIS_SIGNAL_SET of RxEmulator_inst_0_axis_broadcaster_0_0_top_RxEmulator_inst_0_axis_broadcaster_0_0 : entity is 2;
  attribute C_AXIS_TDEST_WIDTH : integer;
  attribute C_AXIS_TDEST_WIDTH of RxEmulator_inst_0_axis_broadcaster_0_0_top_RxEmulator_inst_0_axis_broadcaster_0_0 : entity is 1;
  attribute C_AXIS_TID_WIDTH : integer;
  attribute C_AXIS_TID_WIDTH of RxEmulator_inst_0_axis_broadcaster_0_0_top_RxEmulator_inst_0_axis_broadcaster_0_0 : entity is 1;
  attribute C_FAMILY : string;
  attribute C_FAMILY of RxEmulator_inst_0_axis_broadcaster_0_0_top_RxEmulator_inst_0_axis_broadcaster_0_0 : entity is "zynq";
  attribute C_M_AXIS_TDATA_WIDTH : integer;
  attribute C_M_AXIS_TDATA_WIDTH of RxEmulator_inst_0_axis_broadcaster_0_0_top_RxEmulator_inst_0_axis_broadcaster_0_0 : entity is 16;
  attribute C_M_AXIS_TUSER_WIDTH : integer;
  attribute C_M_AXIS_TUSER_WIDTH of RxEmulator_inst_0_axis_broadcaster_0_0_top_RxEmulator_inst_0_axis_broadcaster_0_0 : entity is 1;
  attribute C_NUM_MI_SLOTS : integer;
  attribute C_NUM_MI_SLOTS of RxEmulator_inst_0_axis_broadcaster_0_0_top_RxEmulator_inst_0_axis_broadcaster_0_0 : entity is 5;
  attribute C_S_AXIS_TDATA_WIDTH : integer;
  attribute C_S_AXIS_TDATA_WIDTH of RxEmulator_inst_0_axis_broadcaster_0_0_top_RxEmulator_inst_0_axis_broadcaster_0_0 : entity is 64;
  attribute C_S_AXIS_TUSER_WIDTH : integer;
  attribute C_S_AXIS_TUSER_WIDTH of RxEmulator_inst_0_axis_broadcaster_0_0_top_RxEmulator_inst_0_axis_broadcaster_0_0 : entity is 1;
  attribute G_INDX_SS_TDATA : integer;
  attribute G_INDX_SS_TDATA of RxEmulator_inst_0_axis_broadcaster_0_0_top_RxEmulator_inst_0_axis_broadcaster_0_0 : entity is 1;
  attribute G_INDX_SS_TDEST : integer;
  attribute G_INDX_SS_TDEST of RxEmulator_inst_0_axis_broadcaster_0_0_top_RxEmulator_inst_0_axis_broadcaster_0_0 : entity is 6;
  attribute G_INDX_SS_TID : integer;
  attribute G_INDX_SS_TID of RxEmulator_inst_0_axis_broadcaster_0_0_top_RxEmulator_inst_0_axis_broadcaster_0_0 : entity is 5;
  attribute G_INDX_SS_TKEEP : integer;
  attribute G_INDX_SS_TKEEP of RxEmulator_inst_0_axis_broadcaster_0_0_top_RxEmulator_inst_0_axis_broadcaster_0_0 : entity is 3;
  attribute G_INDX_SS_TLAST : integer;
  attribute G_INDX_SS_TLAST of RxEmulator_inst_0_axis_broadcaster_0_0_top_RxEmulator_inst_0_axis_broadcaster_0_0 : entity is 4;
  attribute G_INDX_SS_TREADY : integer;
  attribute G_INDX_SS_TREADY of RxEmulator_inst_0_axis_broadcaster_0_0_top_RxEmulator_inst_0_axis_broadcaster_0_0 : entity is 0;
  attribute G_INDX_SS_TSTRB : integer;
  attribute G_INDX_SS_TSTRB of RxEmulator_inst_0_axis_broadcaster_0_0_top_RxEmulator_inst_0_axis_broadcaster_0_0 : entity is 2;
  attribute G_INDX_SS_TUSER : integer;
  attribute G_INDX_SS_TUSER of RxEmulator_inst_0_axis_broadcaster_0_0_top_RxEmulator_inst_0_axis_broadcaster_0_0 : entity is 7;
  attribute G_MASK_SS_TDATA : integer;
  attribute G_MASK_SS_TDATA of RxEmulator_inst_0_axis_broadcaster_0_0_top_RxEmulator_inst_0_axis_broadcaster_0_0 : entity is 2;
  attribute G_MASK_SS_TDEST : integer;
  attribute G_MASK_SS_TDEST of RxEmulator_inst_0_axis_broadcaster_0_0_top_RxEmulator_inst_0_axis_broadcaster_0_0 : entity is 64;
  attribute G_MASK_SS_TID : integer;
  attribute G_MASK_SS_TID of RxEmulator_inst_0_axis_broadcaster_0_0_top_RxEmulator_inst_0_axis_broadcaster_0_0 : entity is 32;
  attribute G_MASK_SS_TKEEP : integer;
  attribute G_MASK_SS_TKEEP of RxEmulator_inst_0_axis_broadcaster_0_0_top_RxEmulator_inst_0_axis_broadcaster_0_0 : entity is 8;
  attribute G_MASK_SS_TLAST : integer;
  attribute G_MASK_SS_TLAST of RxEmulator_inst_0_axis_broadcaster_0_0_top_RxEmulator_inst_0_axis_broadcaster_0_0 : entity is 16;
  attribute G_MASK_SS_TREADY : integer;
  attribute G_MASK_SS_TREADY of RxEmulator_inst_0_axis_broadcaster_0_0_top_RxEmulator_inst_0_axis_broadcaster_0_0 : entity is 1;
  attribute G_MASK_SS_TSTRB : integer;
  attribute G_MASK_SS_TSTRB of RxEmulator_inst_0_axis_broadcaster_0_0_top_RxEmulator_inst_0_axis_broadcaster_0_0 : entity is 4;
  attribute G_MASK_SS_TUSER : integer;
  attribute G_MASK_SS_TUSER of RxEmulator_inst_0_axis_broadcaster_0_0_top_RxEmulator_inst_0_axis_broadcaster_0_0 : entity is 128;
  attribute G_TASK_SEVERITY_ERR : integer;
  attribute G_TASK_SEVERITY_ERR of RxEmulator_inst_0_axis_broadcaster_0_0_top_RxEmulator_inst_0_axis_broadcaster_0_0 : entity is 2;
  attribute G_TASK_SEVERITY_INFO : integer;
  attribute G_TASK_SEVERITY_INFO of RxEmulator_inst_0_axis_broadcaster_0_0_top_RxEmulator_inst_0_axis_broadcaster_0_0 : entity is 0;
  attribute G_TASK_SEVERITY_WARNING : integer;
  attribute G_TASK_SEVERITY_WARNING of RxEmulator_inst_0_axis_broadcaster_0_0_top_RxEmulator_inst_0_axis_broadcaster_0_0 : entity is 1;
  attribute ORIG_REF_NAME : string;
  attribute ORIG_REF_NAME of RxEmulator_inst_0_axis_broadcaster_0_0_top_RxEmulator_inst_0_axis_broadcaster_0_0 : entity is "top_RxEmulator_inst_0_axis_broadcaster_0_0";
  attribute P_TPAYLOAD_WIDTH : integer;
  attribute P_TPAYLOAD_WIDTH of RxEmulator_inst_0_axis_broadcaster_0_0_top_RxEmulator_inst_0_axis_broadcaster_0_0 : entity is 64;
end RxEmulator_inst_0_axis_broadcaster_0_0_top_RxEmulator_inst_0_axis_broadcaster_0_0;

architecture STRUCTURE of RxEmulator_inst_0_axis_broadcaster_0_0_top_RxEmulator_inst_0_axis_broadcaster_0_0 is
  signal \<const0>\ : STD_LOGIC;
  signal \^s_axis_tdata\ : STD_LOGIC_VECTOR ( 63 downto 0 );
  signal \^s_axis_tvalid\ : STD_LOGIC;
begin
  \^s_axis_tdata\(49 downto 0) <= s_axis_tdata(49 downto 0);
  \^s_axis_tvalid\ <= s_axis_tvalid;
  m_axis_tdata(79) <= \<const0>\;
  m_axis_tdata(78) <= \<const0>\;
  m_axis_tdata(77) <= \<const0>\;
  m_axis_tdata(76) <= \<const0>\;
  m_axis_tdata(75) <= \<const0>\;
  m_axis_tdata(74) <= \<const0>\;
  m_axis_tdata(73 downto 64) <= \^s_axis_tdata\(49 downto 40);
  m_axis_tdata(63) <= \<const0>\;
  m_axis_tdata(62) <= \<const0>\;
  m_axis_tdata(61) <= \<const0>\;
  m_axis_tdata(60) <= \<const0>\;
  m_axis_tdata(59) <= \<const0>\;
  m_axis_tdata(58) <= \<const0>\;
  m_axis_tdata(57 downto 48) <= \^s_axis_tdata\(39 downto 30);
  m_axis_tdata(47) <= \<const0>\;
  m_axis_tdata(46) <= \<const0>\;
  m_axis_tdata(45) <= \<const0>\;
  m_axis_tdata(44) <= \<const0>\;
  m_axis_tdata(43) <= \<const0>\;
  m_axis_tdata(42) <= \<const0>\;
  m_axis_tdata(41 downto 32) <= \^s_axis_tdata\(29 downto 20);
  m_axis_tdata(31) <= \<const0>\;
  m_axis_tdata(30) <= \<const0>\;
  m_axis_tdata(29) <= \<const0>\;
  m_axis_tdata(28) <= \<const0>\;
  m_axis_tdata(27) <= \<const0>\;
  m_axis_tdata(26) <= \<const0>\;
  m_axis_tdata(25 downto 16) <= \^s_axis_tdata\(19 downto 10);
  m_axis_tdata(15) <= \<const0>\;
  m_axis_tdata(14) <= \<const0>\;
  m_axis_tdata(13) <= \<const0>\;
  m_axis_tdata(12) <= \<const0>\;
  m_axis_tdata(11) <= \<const0>\;
  m_axis_tdata(10) <= \<const0>\;
  m_axis_tdata(9 downto 0) <= \^s_axis_tdata\(9 downto 0);
  m_axis_tdest(4) <= \<const0>\;
  m_axis_tdest(3) <= \<const0>\;
  m_axis_tdest(2) <= \<const0>\;
  m_axis_tdest(1) <= \<const0>\;
  m_axis_tdest(0) <= \<const0>\;
  m_axis_tid(4) <= \<const0>\;
  m_axis_tid(3) <= \<const0>\;
  m_axis_tid(2) <= \<const0>\;
  m_axis_tid(1) <= \<const0>\;
  m_axis_tid(0) <= \<const0>\;
  m_axis_tkeep(9) <= \<const0>\;
  m_axis_tkeep(8) <= \<const0>\;
  m_axis_tkeep(7) <= \<const0>\;
  m_axis_tkeep(6) <= \<const0>\;
  m_axis_tkeep(5) <= \<const0>\;
  m_axis_tkeep(4) <= \<const0>\;
  m_axis_tkeep(3) <= \<const0>\;
  m_axis_tkeep(2) <= \<const0>\;
  m_axis_tkeep(1) <= \<const0>\;
  m_axis_tkeep(0) <= \<const0>\;
  m_axis_tlast(4) <= \<const0>\;
  m_axis_tlast(3) <= \<const0>\;
  m_axis_tlast(2) <= \<const0>\;
  m_axis_tlast(1) <= \<const0>\;
  m_axis_tlast(0) <= \<const0>\;
  m_axis_tstrb(9) <= \<const0>\;
  m_axis_tstrb(8) <= \<const0>\;
  m_axis_tstrb(7) <= \<const0>\;
  m_axis_tstrb(6) <= \<const0>\;
  m_axis_tstrb(5) <= \<const0>\;
  m_axis_tstrb(4) <= \<const0>\;
  m_axis_tstrb(3) <= \<const0>\;
  m_axis_tstrb(2) <= \<const0>\;
  m_axis_tstrb(1) <= \<const0>\;
  m_axis_tstrb(0) <= \<const0>\;
  m_axis_tuser(4) <= \<const0>\;
  m_axis_tuser(3) <= \<const0>\;
  m_axis_tuser(2) <= \<const0>\;
  m_axis_tuser(1) <= \<const0>\;
  m_axis_tuser(0) <= \<const0>\;
  m_axis_tvalid(4) <= \^s_axis_tvalid\;
  m_axis_tvalid(3) <= \^s_axis_tvalid\;
  m_axis_tvalid(2) <= \^s_axis_tvalid\;
  m_axis_tvalid(1) <= \^s_axis_tvalid\;
  m_axis_tvalid(0) <= \^s_axis_tvalid\;
  s_axis_tready <= \<const0>\;
GND: unisim.vcomponents.GND
     port map (
      G => \<const0>\
    );
end STRUCTURE;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity RxEmulator_inst_0_axis_broadcaster_0_0 is
  port (
    aclk : in STD_LOGIC;
    aresetn : in STD_LOGIC;
    s_axis_tvalid : in STD_LOGIC;
    s_axis_tdata : in STD_LOGIC_VECTOR ( 63 downto 0 );
    m_axis_tvalid : out STD_LOGIC_VECTOR ( 4 downto 0 );
    m_axis_tdata : out STD_LOGIC_VECTOR ( 79 downto 0 )
  );
  attribute NotValidForBitStream : boolean;
  attribute NotValidForBitStream of RxEmulator_inst_0_axis_broadcaster_0_0 : entity is true;
  attribute CHECK_LICENSE_TYPE : string;
  attribute CHECK_LICENSE_TYPE of RxEmulator_inst_0_axis_broadcaster_0_0 : entity is "RxEmulator_inst_0_axis_broadcaster_0_0,top_RxEmulator_inst_0_axis_broadcaster_0_0,{}";
  attribute DowngradeIPIdentifiedWarnings : string;
  attribute DowngradeIPIdentifiedWarnings of RxEmulator_inst_0_axis_broadcaster_0_0 : entity is "yes";
  attribute X_CORE_INFO : string;
  attribute X_CORE_INFO of RxEmulator_inst_0_axis_broadcaster_0_0 : entity is "top_RxEmulator_inst_0_axis_broadcaster_0_0,Vivado 2024.1.1";
end RxEmulator_inst_0_axis_broadcaster_0_0;

architecture STRUCTURE of RxEmulator_inst_0_axis_broadcaster_0_0 is
  signal \<const0>\ : STD_LOGIC;
  signal \^m_axis_tdata\ : STD_LOGIC_VECTOR ( 73 downto 0 );
  signal NLW_inst_s_axis_tready_UNCONNECTED : STD_LOGIC;
  signal NLW_inst_m_axis_tdata_UNCONNECTED : STD_LOGIC_VECTOR ( 79 downto 10 );
  signal NLW_inst_m_axis_tdest_UNCONNECTED : STD_LOGIC_VECTOR ( 4 downto 0 );
  signal NLW_inst_m_axis_tid_UNCONNECTED : STD_LOGIC_VECTOR ( 4 downto 0 );
  signal NLW_inst_m_axis_tkeep_UNCONNECTED : STD_LOGIC_VECTOR ( 9 downto 0 );
  signal NLW_inst_m_axis_tlast_UNCONNECTED : STD_LOGIC_VECTOR ( 4 downto 0 );
  signal NLW_inst_m_axis_tstrb_UNCONNECTED : STD_LOGIC_VECTOR ( 9 downto 0 );
  signal NLW_inst_m_axis_tuser_UNCONNECTED : STD_LOGIC_VECTOR ( 4 downto 0 );
  attribute C_AXIS_SIGNAL_SET : integer;
  attribute C_AXIS_SIGNAL_SET of inst : label is 2;
  attribute C_AXIS_TDEST_WIDTH : integer;
  attribute C_AXIS_TDEST_WIDTH of inst : label is 1;
  attribute C_AXIS_TID_WIDTH : integer;
  attribute C_AXIS_TID_WIDTH of inst : label is 1;
  attribute C_FAMILY : string;
  attribute C_FAMILY of inst : label is "zynq";
  attribute C_M_AXIS_TDATA_WIDTH : integer;
  attribute C_M_AXIS_TDATA_WIDTH of inst : label is 16;
  attribute C_M_AXIS_TUSER_WIDTH : integer;
  attribute C_M_AXIS_TUSER_WIDTH of inst : label is 1;
  attribute C_NUM_MI_SLOTS : integer;
  attribute C_NUM_MI_SLOTS of inst : label is 5;
  attribute C_S_AXIS_TDATA_WIDTH : integer;
  attribute C_S_AXIS_TDATA_WIDTH of inst : label is 64;
  attribute C_S_AXIS_TUSER_WIDTH : integer;
  attribute C_S_AXIS_TUSER_WIDTH of inst : label is 1;
  attribute G_INDX_SS_TDATA : integer;
  attribute G_INDX_SS_TDATA of inst : label is 1;
  attribute G_INDX_SS_TDEST : integer;
  attribute G_INDX_SS_TDEST of inst : label is 6;
  attribute G_INDX_SS_TID : integer;
  attribute G_INDX_SS_TID of inst : label is 5;
  attribute G_INDX_SS_TKEEP : integer;
  attribute G_INDX_SS_TKEEP of inst : label is 3;
  attribute G_INDX_SS_TLAST : integer;
  attribute G_INDX_SS_TLAST of inst : label is 4;
  attribute G_INDX_SS_TREADY : integer;
  attribute G_INDX_SS_TREADY of inst : label is 0;
  attribute G_INDX_SS_TSTRB : integer;
  attribute G_INDX_SS_TSTRB of inst : label is 2;
  attribute G_INDX_SS_TUSER : integer;
  attribute G_INDX_SS_TUSER of inst : label is 7;
  attribute G_MASK_SS_TDATA : integer;
  attribute G_MASK_SS_TDATA of inst : label is 2;
  attribute G_MASK_SS_TDEST : integer;
  attribute G_MASK_SS_TDEST of inst : label is 64;
  attribute G_MASK_SS_TID : integer;
  attribute G_MASK_SS_TID of inst : label is 32;
  attribute G_MASK_SS_TKEEP : integer;
  attribute G_MASK_SS_TKEEP of inst : label is 8;
  attribute G_MASK_SS_TLAST : integer;
  attribute G_MASK_SS_TLAST of inst : label is 16;
  attribute G_MASK_SS_TREADY : integer;
  attribute G_MASK_SS_TREADY of inst : label is 1;
  attribute G_MASK_SS_TSTRB : integer;
  attribute G_MASK_SS_TSTRB of inst : label is 4;
  attribute G_MASK_SS_TUSER : integer;
  attribute G_MASK_SS_TUSER of inst : label is 128;
  attribute G_TASK_SEVERITY_ERR : integer;
  attribute G_TASK_SEVERITY_ERR of inst : label is 2;
  attribute G_TASK_SEVERITY_INFO : integer;
  attribute G_TASK_SEVERITY_INFO of inst : label is 0;
  attribute G_TASK_SEVERITY_WARNING : integer;
  attribute G_TASK_SEVERITY_WARNING of inst : label is 1;
  attribute P_TPAYLOAD_WIDTH : integer;
  attribute P_TPAYLOAD_WIDTH of inst : label is 64;
  attribute X_INTERFACE_INFO : string;
  attribute X_INTERFACE_INFO of aclk : signal is "xilinx.com:signal:clock:1.0 CLKIF CLK";
  attribute X_INTERFACE_PARAMETER : string;
  attribute X_INTERFACE_PARAMETER of aclk : signal is "XIL_INTERFACENAME CLKIF, ASSOCIATED_BUSIF M00_AXIS:M01_AXIS:M02_AXIS:M03_AXIS:M04_AXIS:M05_AXIS:M06_AXIS:M07_AXIS:M08_AXIS:M09_AXIS:M10_AXIS:M11_AXIS:M12_AXIS:M13_AXIS:M14_AXIS:M15_AXIS:S_AXIS, ASSOCIATED_RESET aresetn, ASSOCIATED_CLKEN aclken, FREQ_HZ 50000000, FREQ_TOLERANCE_HZ 0, PHASE 0.0, CLK_DOMAIN top_processing_system7_0_0_FCLK_CLK0, INSERT_VIP 0";
  attribute X_INTERFACE_INFO of aresetn : signal is "xilinx.com:signal:reset:1.0 RSTIF RST";
  attribute X_INTERFACE_PARAMETER of aresetn : signal is "XIL_INTERFACENAME RSTIF, POLARITY ACTIVE_LOW, INSERT_VIP 0, TYPE INTERCONNECT";
  attribute X_INTERFACE_INFO of s_axis_tvalid : signal is "xilinx.com:interface:axis:1.0 S_AXIS TVALID";
  attribute X_INTERFACE_INFO of m_axis_tdata : signal is "xilinx.com:interface:axis:1.0 M00_AXIS TDATA [15:0] [15:0], xilinx.com:interface:axis:1.0 M01_AXIS TDATA [15:0] [31:16], xilinx.com:interface:axis:1.0 M02_AXIS TDATA [15:0] [47:32], xilinx.com:interface:axis:1.0 M03_AXIS TDATA [15:0] [63:48], xilinx.com:interface:axis:1.0 M04_AXIS TDATA [15:0] [79:64]";
  attribute X_INTERFACE_PARAMETER of m_axis_tdata : signal is "XIL_INTERFACENAME M00_AXIS, TDATA_NUM_BYTES 2, TDEST_WIDTH 0, TID_WIDTH 0, TUSER_WIDTH 0, HAS_TREADY 0, HAS_TSTRB 0, HAS_TKEEP 0, HAS_TLAST 0, FREQ_HZ 50000000, PHASE 0.0, CLK_DOMAIN top_processing_system7_0_0_FCLK_CLK0, LAYERED_METADATA undef, INSERT_VIP 0, XIL_INTERFACENAME M01_AXIS, TDATA_NUM_BYTES 2, TDEST_WIDTH 0, TID_WIDTH 0, TUSER_WIDTH 0, HAS_TREADY 0, HAS_TSTRB 0, HAS_TKEEP 0, HAS_TLAST 0, FREQ_HZ 50000000, PHASE 0.0, CLK_DOMAIN top_processing_system7_0_0_FCLK_CLK0, LAYERED_METADATA undef, INSERT_VIP 0, XIL_INTERFACENAME M02_AXIS, TDATA_NUM_BYTES 2, TDEST_WIDTH 0, TID_WIDTH 0, TUSER_WIDTH 0, HAS_TREADY 0, HAS_TSTRB 0, HAS_TKEEP 0, HAS_TLAST 0, FREQ_HZ 50000000, PHASE 0.0, CLK_DOMAIN top_processing_system7_0_0_FCLK_CLK0, LAYERED_METADATA undef, INSERT_VIP 0, XIL_INTERFACENAME M03_AXIS, TDATA_NUM_BYTES 2, TDEST_WIDTH 0, TID_WIDTH 0, TUSER_WIDTH 0, HAS_TREADY 0, HAS_TSTRB 0, HAS_TKEEP 0, HAS_TLAST 0, FREQ_HZ 50000000, PHASE 0.0, CLK_DOMAIN top_processing_system7_0_0_FCLK_CLK0, LAYERED_METADATA undef, INSERT_VIP 0, XIL_INTERFACENAME M04_AXIS, TDATA_NUM_BYTES 2, TDEST_WIDTH 0, TID_WIDTH 0, TUSER_WIDTH 0, HAS_TREADY 0, HAS_TSTRB 0, HAS_TKEEP 0, HAS_TLAST 0, FREQ_HZ 50000000, PHASE 0.0, CLK_DOMAIN top_processing_system7_0_0_FCLK_CLK0, LAYERED_METADATA undef, INSERT_VIP 0";
  attribute X_INTERFACE_INFO of m_axis_tvalid : signal is "xilinx.com:interface:axis:1.0 M00_AXIS TVALID [0:0] [0:0], xilinx.com:interface:axis:1.0 M01_AXIS TVALID [0:0] [1:1], xilinx.com:interface:axis:1.0 M02_AXIS TVALID [0:0] [2:2], xilinx.com:interface:axis:1.0 M03_AXIS TVALID [0:0] [3:3], xilinx.com:interface:axis:1.0 M04_AXIS TVALID [0:0] [4:4]";
  attribute X_INTERFACE_INFO of s_axis_tdata : signal is "xilinx.com:interface:axis:1.0 S_AXIS TDATA";
  attribute X_INTERFACE_PARAMETER of s_axis_tdata : signal is "XIL_INTERFACENAME S_AXIS, TDATA_NUM_BYTES 8, TDEST_WIDTH 0, TID_WIDTH 0, TUSER_WIDTH 0, HAS_TREADY 0, HAS_TSTRB 0, HAS_TKEEP 0, HAS_TLAST 0, FREQ_HZ 50000000, PHASE 0.0, CLK_DOMAIN top_processing_system7_0_0_FCLK_CLK0, LAYERED_METADATA undef, INSERT_VIP 0";
begin
  m_axis_tdata(79) <= \<const0>\;
  m_axis_tdata(78) <= \<const0>\;
  m_axis_tdata(77) <= \<const0>\;
  m_axis_tdata(76) <= \<const0>\;
  m_axis_tdata(75) <= \<const0>\;
  m_axis_tdata(74) <= \<const0>\;
  m_axis_tdata(73 downto 64) <= \^m_axis_tdata\(73 downto 64);
  m_axis_tdata(63) <= \<const0>\;
  m_axis_tdata(62) <= \<const0>\;
  m_axis_tdata(61) <= \<const0>\;
  m_axis_tdata(60) <= \<const0>\;
  m_axis_tdata(59) <= \<const0>\;
  m_axis_tdata(58) <= \<const0>\;
  m_axis_tdata(57 downto 48) <= \^m_axis_tdata\(57 downto 48);
  m_axis_tdata(47) <= \<const0>\;
  m_axis_tdata(46) <= \<const0>\;
  m_axis_tdata(45) <= \<const0>\;
  m_axis_tdata(44) <= \<const0>\;
  m_axis_tdata(43) <= \<const0>\;
  m_axis_tdata(42) <= \<const0>\;
  m_axis_tdata(41 downto 32) <= \^m_axis_tdata\(41 downto 32);
  m_axis_tdata(31) <= \<const0>\;
  m_axis_tdata(30) <= \<const0>\;
  m_axis_tdata(29) <= \<const0>\;
  m_axis_tdata(28) <= \<const0>\;
  m_axis_tdata(27) <= \<const0>\;
  m_axis_tdata(26) <= \<const0>\;
  m_axis_tdata(25 downto 16) <= \^m_axis_tdata\(25 downto 16);
  m_axis_tdata(15) <= \<const0>\;
  m_axis_tdata(14) <= \<const0>\;
  m_axis_tdata(13) <= \<const0>\;
  m_axis_tdata(12) <= \<const0>\;
  m_axis_tdata(11) <= \<const0>\;
  m_axis_tdata(10) <= \<const0>\;
  m_axis_tdata(9 downto 0) <= \^m_axis_tdata\(9 downto 0);
GND: unisim.vcomponents.GND
     port map (
      G => \<const0>\
    );
inst: entity work.RxEmulator_inst_0_axis_broadcaster_0_0_top_RxEmulator_inst_0_axis_broadcaster_0_0
     port map (
      aclk => '0',
      aclken => '1',
      aresetn => '0',
      m_axis_tdata(79 downto 74) => NLW_inst_m_axis_tdata_UNCONNECTED(79 downto 74),
      m_axis_tdata(73 downto 64) => \^m_axis_tdata\(73 downto 64),
      m_axis_tdata(63 downto 58) => NLW_inst_m_axis_tdata_UNCONNECTED(63 downto 58),
      m_axis_tdata(57 downto 48) => \^m_axis_tdata\(57 downto 48),
      m_axis_tdata(47 downto 42) => NLW_inst_m_axis_tdata_UNCONNECTED(47 downto 42),
      m_axis_tdata(41 downto 32) => \^m_axis_tdata\(41 downto 32),
      m_axis_tdata(31 downto 26) => NLW_inst_m_axis_tdata_UNCONNECTED(31 downto 26),
      m_axis_tdata(25 downto 16) => \^m_axis_tdata\(25 downto 16),
      m_axis_tdata(15 downto 10) => NLW_inst_m_axis_tdata_UNCONNECTED(15 downto 10),
      m_axis_tdata(9 downto 0) => \^m_axis_tdata\(9 downto 0),
      m_axis_tdest(4 downto 0) => NLW_inst_m_axis_tdest_UNCONNECTED(4 downto 0),
      m_axis_tid(4 downto 0) => NLW_inst_m_axis_tid_UNCONNECTED(4 downto 0),
      m_axis_tkeep(9 downto 0) => NLW_inst_m_axis_tkeep_UNCONNECTED(9 downto 0),
      m_axis_tlast(4 downto 0) => NLW_inst_m_axis_tlast_UNCONNECTED(4 downto 0),
      m_axis_tready(4 downto 0) => B"11111",
      m_axis_tstrb(9 downto 0) => NLW_inst_m_axis_tstrb_UNCONNECTED(9 downto 0),
      m_axis_tuser(4 downto 0) => NLW_inst_m_axis_tuser_UNCONNECTED(4 downto 0),
      m_axis_tvalid(4 downto 0) => m_axis_tvalid(4 downto 0),
      s_axis_tdata(63 downto 50) => B"00000000000000",
      s_axis_tdata(49 downto 0) => s_axis_tdata(49 downto 0),
      s_axis_tdest(0) => '0',
      s_axis_tid(0) => '0',
      s_axis_tkeep(7 downto 0) => B"11111111",
      s_axis_tlast => '1',
      s_axis_tready => NLW_inst_s_axis_tready_UNCONNECTED,
      s_axis_tstrb(7 downto 0) => B"11111111",
      s_axis_tuser(0) => '0',
      s_axis_tvalid => s_axis_tvalid
    );
end STRUCTURE;
