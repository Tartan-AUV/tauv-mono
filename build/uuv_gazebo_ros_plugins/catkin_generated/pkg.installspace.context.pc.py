# generated from catkin/cmake/template/pkg.context.pc.in
CATKIN_PACKAGE_PREFIX = ""
PROJECT_PKG_CONFIG_INCLUDE_DIRS = "${prefix}/include".split(';') if "${prefix}/include" != "" else []
PROJECT_CATKIN_DEPENDS = "uuv_gazebo_plugins;sensor_msgs;geometry_msgs;std_msgs;uuv_gazebo_ros_plugins_msgs;visualization_msgs".replace(';', ' ')
PKG_CONFIG_LIBRARIES_WITH_PREFIX = "-luuv_fin_ros_plugin;-luuv_thruster_ros_plugin;-luuv_underwater_object_ros_plugin;-luuv_joint_state_publisher;-luuv_accelerations_test_plugin".split(';') if "-luuv_fin_ros_plugin;-luuv_thruster_ros_plugin;-luuv_underwater_object_ros_plugin;-luuv_joint_state_publisher;-luuv_accelerations_test_plugin" != "" else []
PROJECT_NAME = "uuv_gazebo_ros_plugins"
PROJECT_SPACE_DIR = "/home/tom/workspaces/tauv_ws/install"
PROJECT_VERSION = "0.6.13"
