//
// Created by tom on 10/6/19.
//

#include "ImuCalibrator.h"
#include <yaml-cpp/yaml.h>

#include <string>
#include <exception>
#include <tf/LinearMath/Transform.h>
#include <tf/LinearMath/Matrix3x3.h>
#include <tf/LinearMath/Quaternion.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/Vector3.h>
#include <tf/transform_datatypes.h>
#include <fstream>
#include <cmath>

ImuCalibrator::ImuCalibrator() : _nh() {
    load_axis_map();
    load_autolevel_matrix();

    std::string data_input_topic;
    if (!_nh.getParam(IMU_DATA_TOPIC_PARAM, data_input_topic))
    {
        throw std::runtime_error("No param defined for imu data in topic!");
    }
    _imu_sub = _nh.subscribe(data_input_topic, 10, &ImuCalibrator::imu_data_callback, this);
    _imu_pub = _nh.advertise<sensor_msgs::Imu>(IMU_DATA_OUTPUT_TOPIC, true);
}

void ImuCalibrator::imu_data_callback(const sensor_msgs::Imu::ConstPtr& msg) {
    sensor_msgs::Imu new_msg = *msg;

    geometry_msgs::Quaternion ori_geom = new_msg.orientation;
    geometry_msgs::Vector3 ang_geom = new_msg.angular_velocity;
    geometry_msgs::Vector3 acc_geom = new_msg.linear_acceleration;
    tf::Quaternion ori_raw;
    tf::Vector3 ang_raw;
    tf::Vector3 acc_raw;

    quaternionMsgToTF(ori_geom, ori_raw);
    vector3MsgToTF(ang_geom, ang_raw);
    vector3MsgToTF(acc_geom, acc_raw);

    //std::cout << "----------------------------\n";

    //std::cout << "reorient matrix: ";
    //print_matrix3x3(_reorient.getBasis());
    //std::cout << "acc raw: ";
    //print_vector3(ang_raw);
    tf::Quaternion ori_reo_raw = _reorient * ori_raw;
    ori_reo_raw.normalize();
    tf::Vector3 ang_reo_raw = _reorient * ang_raw;
    tf::Vector3 acc_reo_raw = _reorient * acc_raw;
    //std::cout << "acc reoriented: ";
    //print_vector3(ang_reo_raw);

    if (_autolevel_samples > 0) {
        _autolevel_samples--;
        _autolevel_quaternion += ori_reo_raw;
    }
    if (_autolevel_samples == 0) {
        // zero out the z rotation from the quaternion:
        _autolevel_quaternion[2] = 0;
        _autolevel_quaternion.normalize();

        _autolevel = tf::Transform(_autolevel_quaternion);
        _autolevel_samples = -1;
    }

    tf::Quaternion ori_reo = _autolevel * ori_reo_raw;
    ori_reo.normalize();
    tf::Vector3 ang_reo = _autolevel * ang_reo_raw;
    tf::Vector3 acc_reo = _autolevel * acc_reo_raw;

    if (_zero_heading_samples > 0) {
        _zero_heading_samples--;
        _zero_heading_quaternion += ori_reo;
    }
    if (_zero_heading_samples == 0) {
        _zero_heading_quaternion[0] = 0;
        _zero_heading_quaternion[1] = 0;
        _zero_heading_quaternion.normalize();

        _zero_heading = tf::Transform(_zero_heading_quaternion);
        _zero_heading_samples = -1;
    }

    quaternionTFToMsg(ori_reo, new_msg.orientation);
    vector3TFToMsg(ang_reo, new_msg.angular_velocity);
    vector3TFToMsg(acc_reo, new_msg.linear_acceleration);

    _imu_pub.publish(new_msg);
}

void ImuCalibrator::autolevel_service_callback() {
    _autolevel_quaternion = tf::Quaternion(0,0,0,0);
    _autolevel_samples = 20;
}

void ImuCalibrator::zero_heading_service_callback() {
    _zero_heading_quaternion = tf::Quaternion(0,0,0,0);
    _zero_heading_samples = 20;
}


void ImuCalibrator::load_axis_map() {
    _axis_map = load_yaml_array<int>(AXIS_MAP_PARAM, 3);
    _reorient_mat = tf::Matrix3x3(0,0,0,0,0,0,0,0,0);
    for (int i = 0; i < 3; i++) {
        if (std::abs(_axis_map[i]) > 3 || std::abs(_axis_map[i]) < 1) {
            throw std::runtime_error("Error: Invalid axis map. Values must be -3,-2,-1,1,2,3 only");
        }
        int val = _axis_map[i];
        _reorient_mat[i][std::abs(val)-1] = (double)(val/std::abs(val));
    }

    std::cout << "Calculated reorientation matrix: \n";
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            std::cout << _reorient_mat[r][c] << "\n";
        }
    }

    _reorient = tf::Transform(_reorient_mat);
}

void ImuCalibrator::load_autolevel_matrix() {
    auto autolevel_mat_flat = load_yaml_array<double>(AUTOLEVEL_MAT_PARAM, 9, false);
    if (autolevel_mat_flat.size() == 0) {
        std::cout << "No autolevel matrix found! Defining one using the identity matrix...\n";
        _autolevel = tf::Transform();
        _autolevel.setIdentity();
        save_autolevel_matrix();
    }
    else {
        _autolevel = tf::Transform(
            tf::Matrix3x3(
                autolevel_mat_flat[0],
                autolevel_mat_flat[1],
                autolevel_mat_flat[2],
                autolevel_mat_flat[3],
                autolevel_mat_flat[4],
                autolevel_mat_flat[5],
                autolevel_mat_flat[6],
                autolevel_mat_flat[7],
                autolevel_mat_flat[8]));
    }
}

void ImuCalibrator::save_autolevel_matrix(){
    auto vec = std::vector<double>(9);
    auto autolevel_mat = _autolevel.getBasis();
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            vec[r*3+c] = autolevel_mat[r][c];
        }
    }
    write_yaml_array(AUTOLEVEL_MAT_PARAM, vec);
}

template <class T>
std::vector<T> ImuCalibrator::load_yaml_array(std::string param, unsigned int len, bool exception_on_missing) {
    std::string path;

    //Check if the parameter exists:
    if (_nh.getParam(param, path)) {
        std::cout << "Loading yaml array from " << path << "\n";
    }
    else {
        std::cout << "Could not find parameter \"" << param << "\" defined in rosparams list!\n";
        throw std::runtime_error("Error: No path defined for parameter!");

    }

    // Try to read the YAML file:
    YAML::Node map;
    try {
        map = YAML::LoadFile(path);
    }
    catch(...) {
        std::cout << "Could not load file at " << path <<"\n";
        if (exception_on_missing) {
            throw std::runtime_error("Error: Could not find required YAML file!");
        }
        return std::vector<T>(0);
    }

    // Check YAML file validity:
    if (!map[param]) {
        throw std::runtime_error("Error: axis_map variable missing from axis map yaml file!");
    }
    if (map[param].size() != len) {
        std::cout << "axis map invalid size! Expected: " << len << ". Actual: " << map[param].size() << "\n";
        throw std::runtime_error("Error: axis_map variable incorrect size!");
    }

    // Read the array from the YAML file:
    std::vector<T> res = std::vector<T>(len);
    std::cout << "Vector read: [ ";
    for (unsigned int i = 0; i < len; i++)
    {
        res[i] = map[param][i].as<T>();
        std::cout << res[i] << " ";
    }
    std::cout << "]\n";
    return res;
}

template <class T>
void ImuCalibrator::write_yaml_array(std::string param, std::vector<T> vec) {

    std::string path;
    //Check if the parameter exists:
    if (_nh.getParam(param, path)) {
        std::cout << "Writing YAML array to " << path << "\n";
    }
    else {
        std::cout << "Could not find parameter \"" << param << "\" defined in rosparams list!\n";
        throw std::runtime_error("Error: No path defined for parameter!");

    }

    YAML::Emitter out;
    out << YAML::Comment("This is the autolevel calibration matrix, which transforms from");
    out << YAML::Newline;
    out << YAML::Comment("the IMU frame to the world frame. Generated by imu_calibrator node");
    out << YAML::Newline;
    out << YAML::Comment("It is not recommended to manually edit this matrix.");
    out << YAML::Newline;
    out << YAML::BeginMap;
    out << YAML::Key << param;
    out << vec;
    out << YAML::EndMap;

    std::ofstream fout(path);
    fout << out.c_str();
    fout.close();
}

void ImuCalibrator::print_matrix3x3(tf::Matrix3x3 mat) {
    std::cout << "-Mat: \n";
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            std::cout << mat[r][c] << ", ";
        }
        std::cout << "\n";
    }
}

void ImuCalibrator::print_vector3(tf::Vector3 vec) {
    std::cout << "-Vec: \n";
    for (int i = 0; i < 3; i++) {
        std::cout << vec[i] << ", ";
    }
    std::cout << "\n";
}

int main(int argc, char** argv) {
    std::cout << "Starting IMU calibrator\n";
    ros::init(argc, argv, NODE_NAME);
    ImuCalibrator ic = ImuCalibrator();
    ros::spin();
    return 0;
}
