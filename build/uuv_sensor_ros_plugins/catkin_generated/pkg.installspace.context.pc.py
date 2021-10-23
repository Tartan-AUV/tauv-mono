# generated from catkin/cmake/template/pkg.context.pc.in
CATKIN_PACKAGE_PREFIX = ""
PROJECT_PKG_CONFIG_INCLUDE_DIRS = "${prefix}/include;/usr/include/gazebo-11/gazebo/msgs;/usr/include/OGRE;/usr/include;/usr/include/OGRE/Paging;/usr/include/opencv4;/usr/include/eigen3".split(';') if "${prefix}/include;/usr/include/gazebo-11/gazebo/msgs;/usr/include/OGRE;/usr/include;/usr/include/OGRE/Paging;/usr/include/opencv4;/usr/include/eigen3" != "" else []
PROJECT_CATKIN_DEPENDS = "uuv_gazebo_plugins;sensor_msgs;uuv_sensor_plugins_ros_msgs;tauv_msgs".replace(';', ' ')
PKG_CONFIG_LIBRARIES_WITH_PREFIX = "-luuv_gazebo_ros_base_model_plugin;-luuv_gazebo_ros_base_sensor_plugin;-luuv_gazebo_ros_gps_plugin;-luuv_gazebo_ros_pose_gt_plugin;-luuv_gazebo_ros_subsea_pressure_plugin;-luuv_gazebo_ros_dvl_plugin;-luuv_gazebo_ros_magnetometer_plugin;-luuv_gazebo_ros_cpc_plugin;-luuv_gazebo_ros_imu_plugin;-luuv_gazebo_ros_rpt_plugin;-luuv_gazebo_ros_camera_plugin".split(';') if "-luuv_gazebo_ros_base_model_plugin;-luuv_gazebo_ros_base_sensor_plugin;-luuv_gazebo_ros_gps_plugin;-luuv_gazebo_ros_pose_gt_plugin;-luuv_gazebo_ros_subsea_pressure_plugin;-luuv_gazebo_ros_dvl_plugin;-luuv_gazebo_ros_magnetometer_plugin;-luuv_gazebo_ros_cpc_plugin;-luuv_gazebo_ros_imu_plugin;-luuv_gazebo_ros_rpt_plugin;-luuv_gazebo_ros_camera_plugin" != "" else []
PROJECT_NAME = "uuv_sensor_ros_plugins"
PROJECT_SPACE_DIR = "/home/tom/workspaces/tauv_ws/install"
PROJECT_VERSION = "0.6.1"
