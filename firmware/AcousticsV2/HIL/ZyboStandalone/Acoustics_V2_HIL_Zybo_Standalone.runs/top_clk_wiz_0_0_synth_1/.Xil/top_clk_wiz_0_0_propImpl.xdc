set_property SRC_FILE_INFO {cfile:/home/gleb/dev-shared/TAUV-ROS-Packages/firmware/AcousticsV2/HIL/ZyboStandalone/Acoustics_V2_HIL_Zybo_Standalone.gen/sources_1/bd/top/ip/top_clk_wiz_0_0/top_clk_wiz_0_0.xdc rfile:../../../Acoustics_V2_HIL_Zybo_Standalone.gen/sources_1/bd/top/ip/top_clk_wiz_0_0/top_clk_wiz_0_0.xdc id:1 order:EARLY scoped_inst:inst} [current_design]
current_instance inst
set_property src_info {type:SCOPED_XDC file:1 line:54 export:INPUT save:INPUT read:READ} [current_design]
set_input_jitter [get_clocks -of_objects [get_ports clk_in1]] 0.200
