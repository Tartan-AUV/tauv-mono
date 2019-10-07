//
// Created by tom on 10/6/19.
//

#ifndef TAUV_ROS_PACKAGES_IMUCALIBRATOR_H
#define TAUV_ROS_PACKAGES_IMUCALIBRATOR_H

//some generically useful stuff to include...
#include <math.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include <ros/ros.h>

#include <sensor_msgs/Imu.h>
#include <std_msgs/Header.h>
#include <tf/LinearMath/Transform.h>
#include <tf/LinearMath/Quaternion.h>

#define NODE_NAME "imu_calibrator"
#define AXIS_MAP_PARAM "axis_map"
#define AUTOLEVEL_MAT_PARAM "autolevel_matrix"
#define IMU_DATA_TOPIC_PARAM "imu_data_topic"
#define IMU_DATA_OUTPUT_TOPIC "data"

class ImuCalibrator {
public:
    ImuCalibrator();
    void imu_data_callback(const sensor_msgs::Imu::ConstPtr& msg);
    void autolevel_service_callback();

protected:
    void load_axis_map();
    void load_autolevel_matrix();
    void save_autolevel_matrix();

    template <class T>
    std::vector<T> load_yaml_array(std::string param, unsigned int len, bool exception_on_missing=true);

    template <class T>
    void write_yaml_array(std::string param, std::vector<T> vec);

    tf::Matrix3x3 _reorient_mat; // matrix to reorient vectors to REP 105
    tf::Transform _reorient; // Transformation to reorient vectors to REP 105
    tf::Transform _autolevel; // Transformation to correct for autolevel. (applied after reorientation matrix)

    std::vector<int> _axis_map; // map from new axes to imu. 0,1,2 = x,y,z respectively.

    ros::NodeHandle _nh;
    ros::Subscriber _imu_sub;
    ros::Publisher _imu_pub;

    int _autolevel_samples = -1; // set this to take n samples for autoleveling
    tf::Quaternion _autolevel_quaternion;
};


#endif //TAUV_ROS_PACKAGES_IMUCALIBRATOR_H
