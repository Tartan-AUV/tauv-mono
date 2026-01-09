/*
 * Lucid Camera Driver ROS2 Node
 */

#include <chrono>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <iostream>
#include <filesystem>
#include <iomanip>
#include <csignal>
#include <atomic>

// ROS2 includes
#include "cv_bridge/cv_bridge.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

// Arena SDK includes
#include "ArenaApi.h"

// OpenCV includes
#include <opencv2/core/version.hpp>
#include <opencv2/opencv.hpp>
#if CV_MAJOR_VERSION >= 3
#include <opencv2/imgcodecs.hpp>
#else
#include <opencv2/highgui/highgui.hpp>
#endif

// OpenCV CUDA includes
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

class LucidDriver : public rclcpp::Node {
 public:
  LucidDriver()
      : Node("lucid"), shutdown_requested_(false) {
    // Declare parameters
    this->declare_parameter("camera_ip", "10.0.1.11");
    this->declare_parameter("topic_name", "/image_raw");
    this->declare_parameter("horizontal_binning", 1);
    this->declare_parameter("vertical_binning", 1);
    this->declare_parameter("image_height", 5000);
    this->declare_parameter("image_width", 2500);
    this->declare_parameter("save_folder", "/data/test2");
    this->declare_parameter("publish_downsampling_factor", 8);
    this->declare_parameter("image_offset_x", 0);
    this->declare_parameter("image_offset_y", 0);

    // Get parameters
    camera_ip_ = this->get_parameter("camera_ip").as_string();
    topic_name_ = this->get_parameter("topic_name").as_string();
    horizontal_binning_ = this->get_parameter("horizontal_binning").as_int();
    vertical_binning_ = this->get_parameter("vertical_binning").as_int();
    image_height_ = this->get_parameter("image_height").as_int();
    image_width_ = this->get_parameter("image_width").as_int();
    save_folder_ = this->get_parameter("save_folder").as_string();
    publish_downsampling_factor_ = this->get_parameter("publish_downsampling_factor").as_int();
    image_offset_x_ = this->get_parameter("image_offset_x").as_int();
    image_offset_y_ = this->get_parameter("image_offset_y").as_int();

    // Validate downsampling factor
    if (publish_downsampling_factor_ < 1) {
      RCLCPP_WARN(this->get_logger(), "Invalid publish_downsampling_factor (%d), setting to 1", publish_downsampling_factor_);
      publish_downsampling_factor_ = 1;
    }

    // Setup save folder if specified
    if (!save_folder_.empty()) {
      try {
        std::filesystem::create_directories(save_folder_);
        RCLCPP_INFO(this->get_logger(), "Saving full-resolution images to: %s", save_folder_.c_str());
        image_counter_ = 0;
      } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to create save folder: %s", e.what());
        save_folder_.clear();
      }
    }

    // Set up QoS profile
    auto qos = rclcpp::QoS(10);
    qos.reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);
    qos.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);

    // Create publisher
    image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(topic_name_, qos);

    // Check CUDA availability
    int cuda_device_count = cv::cuda::getCudaEnabledDeviceCount();
    if (cuda_device_count > 0) {
      cv::cuda::setDevice(0);
      cv::cuda::DeviceInfo dev_info;
    } else {
      RCLCPP_ERROR(this->get_logger(), "No CUDA-enabled devices found!");
      throw std::runtime_error("CUDA required but not available");
    }

    RCLCPP_INFO(this->get_logger(),
                "Lucid Node Initialized with Camera IP: %s, Topic: %s, Downsampling: %dx",
                camera_ip_.c_str(), topic_name_.c_str(), publish_downsampling_factor_);
    
    // Register signal handler for graceful shutdown
    std::signal(SIGINT, LucidDriver::signalHandler);
    std::signal(SIGTERM, LucidDriver::signalHandler);
    instance_ = this;
  }

  ~LucidDriver() {

    // Clean up Arena SDK resources
    if (device_ != nullptr && system_ != nullptr) {
      try {
        device_->StopStream();
        system_->DestroyDevice(device_);
      } catch (...) {
        // Ignore exceptions during cleanup
      }
    }
    if (system_ != nullptr) {
      Arena::CloseSystem(system_);
    }
  }

  void start() {
    // Initialize Arena SDK system
    system_ = Arena::OpenSystem();

    // Find device with specified IP
    getDeviceByIP(camera_ip_);
    if (device_ == nullptr) {
      throw std::runtime_error("Device with IP " + camera_ip_ + " not found");
    }
    
    // Set device nodemap after device is found
    device_nodemap_ = device_->GetNodeMap();

    // Setup camera configuration
    setupCamera();

    // Start streaming
    streamImages();
  }

  static void signalHandler(int signal) {
    if (instance_ != nullptr) {
      RCLCPP_INFO(instance_->get_logger(), "Shutdown signal received (%d), stopping stream gracefully...", signal);
      instance_->shutdown_requested_.store(true);
      rclcpp::shutdown();
    }
  }

 private:
  bool getDeviceByIP(const std::string& ip) {
    system_->UpdateDevices(100);
    std::vector<Arena::DeviceInfo> deviceInfos = system_->GetDevices();

    for (const auto& deviceInfo : deviceInfos) {
      if (std::string(const_cast<Arena::DeviceInfo&>(deviceInfo).IpAddressStr()) == ip) {
        RCLCPP_INFO(this->get_logger(), "Found device at IP: %s", ip.c_str());
        device_ = system_->CreateDevice(deviceInfo);
        device_ip_ = const_cast<Arena::DeviceInfo&>(deviceInfo).IpAddress();
        return true;
      }
    }

    return false;
    // todo: handle retries
  }

  void setupImageFormat() {
    Arena::SetNodeValue<GenICam::gcstring>(device_nodemap_, "PixelFormat", "BayerRG8");

    Arena::SetNodeValue<GenICam::gcstring>(device_nodemap_, "BinningSelector", "Digital");
    Arena::SetNodeValue<GenICam::gcstring>(device_nodemap_, "BinningHorizontalMode", "Sum");
    Arena::SetNodeValue<GenICam::gcstring>(device_nodemap_, "BinningVerticalMode", "Sum");

    Arena::SetNodeValue<int64_t>(device_nodemap_, "BinningHorizontal", horizontal_binning_);
    Arena::SetNodeValue<int64_t>(device_nodemap_, "BinningVertical", vertical_binning_);
    
    Arena::SetNodeValue<int64_t>(device_nodemap_, "Width", image_width_);
    Arena::SetNodeValue<int64_t>(device_nodemap_, "Height", image_height_);
    Arena::SetNodeValue<int64_t>(device_nodemap_, "OffsetX", image_offset_x_);
    Arena::SetNodeValue<int64_t>(device_nodemap_, "OffsetY", image_offset_y_);
  }

  void setupAcquisition() {
    Arena::SetNodeValue<bool>(device_nodemap_, "AcquisitionFrameRateEnable", false);
    Arena::SetNodeValue<GenICam::gcstring>(device_nodemap_, "AcquisitionMode", "Continuous");
  }

  void setupTransport(GenApi::INodeMap* stream_nodemap) {
    Arena::SetNodeValue<int64_t>(device_nodemap_, "DeviceStreamChannelPacketSize", 9000);
    Arena::SetNodeValue<GenICam::gcstring>(device_nodemap_, "TransportStreamProtocol", "TCP");

    Arena::SetNodeValue<GenICam::gcstring>(stream_nodemap, "StreamBufferHandlingMode",
                                           "NewestOnly");
    Arena::SetNodeValue<bool>(stream_nodemap, "StreamAutoNegotiatePacketSize", true);
    Arena::SetNodeValue<bool>(stream_nodemap, "StreamPacketResendEnable", true);
  }

  void setupPtp() {
    Arena::SetNodeValue<bool>(device_nodemap_, "PtpEnable", true);
    Arena::SetNodeValue<bool>(device_nodemap_, "PtpSlaveOnly", true);

    RCLCPP_INFO(this->get_logger(), "PTP enabled, waiting for master...");
    std::string currPtpStatus = static_cast<std::string>(Arena::GetNodeValue<GenICam::gcstring>(device_nodemap_, "PtpStatus"));
    while (currPtpStatus != "Slave") {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      currPtpStatus = static_cast<std::string>(Arena::GetNodeValue<GenICam::gcstring>(device_nodemap_, "PtpStatus"));
      RCLCPP_INFO(this->get_logger(), "PTP status: %s", currPtpStatus.c_str());
    }
    RCLCPP_INFO(this->get_logger(), "The camera is a PTP slave"); 
  }

  void setupCamera() {
    auto stream_nodemap = device_->GetTLStreamNodeMap();

    // Stop stream if running
    try {
      device_->StopStream();
    } catch (...) {
      RCLCPP_WARN(this->get_logger(), "Failed to stop stream before configuring...");
    }

    // setupImageFormat();

    // setupAcquisition();

    setupPtp();

    // setupTransport(stream_nodemap);
  }

  void setupPtpTriggering() {
    Arena::SetNodeValue<GenICam::gcstring>(device_nodemap_, "TriggerSelector", "FrameStart");
    Arena::SetNodeValue<GenICam::gcstring>(device_nodemap_, "TriggerMode", "On");
    Arena::SetNodeValue<GenICam::gcstring>(device_nodemap_, "TriggerSource", "Action0");

    Arena::SetNodeValue<GenICam::gcstring>(device_nodemap_, "ActionUnconditionalMode", "On");
    Arena::SetNodeValue<int64_t>(device_nodemap_, "ActionSelector", 0);
    Arena::SetNodeValue<int64_t>(device_nodemap_, "ActionDeviceKey", _g_action_device_key);
    Arena::SetNodeValue<int64_t>(device_nodemap_, "ActionGroupKey", _g_action_group_key);
    Arena::SetNodeValue<int64_t>(device_nodemap_, "ActionGroupMask", _g_action_device_key);
  }

  void streamImages() {
    device_->StartStream();

    Arena::ExecuteNode(device_nodemap_, "PtpDataSetLatch");
    int64_t cameraPtp = Arena::GetNodeValue<int64_t>(device_nodemap_, "PtpDataSetLatchValue");
    
    auto camera_time = std::chrono::time_point<std::chrono::system_clock>(std::chrono::nanoseconds(cameraPtp));
    auto system_time = std::chrono::time_point<std::chrono::steady_clock>(std::chrono::steady_clock::now());
    auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(system_time.time_since_epoch() - camera_time.time_since_epoch());
    RCLCPP_INFO(this->get_logger(), "Time difference: %ld ms", time_diff.count());

    return;

    while (rclcpp::ok() && !shutdown_requested_.load()) {
      auto curr_time = std::chrono::steady_clock::now();

      try {
        // Arena::SetNodeValue<int64_t>(system_->GetTLSystemNodeMap(), "ActionCommandTargetIP", 0xFFFFFFFF);
        // Arena::SetNodeValue<int64_t>(system_->GetTLSystemNodeMap(), "ActionCommandExecuteTime", curr_ptp);
        // Arena::ExecuteNode(system_->GetTLSystemNodeMap(), "ActionCommandFireCommand");

        // Get image from camera with 2 second timeout
        Arena::IImage* arena_image = device_->GetImage(10000000);

        // Convert Arena image to OpenCV Mat (full resolution, debayered)
        cv::Mat cv_frame = convertArenaImageToCV(arena_image);

        // Calculate and log FPS
        // auto duration =
        //     std::chrono::duration_cast<std::chrono::milliseconds>(curr_time - prev_time);
        // double fps = 1000.0 / duration.count();
        // RCLCPP_INFO(this->get_logger(), "FPS: %.2f", fps);
        // RCLCPP_INFO(this->get_logger(), "Image Size (%d, %d)", cv_frame.cols,
        //             cv_frame.rows);

        // Save full-resolution image if save folder is specified
        if (!save_folder_.empty()) {
          saveImage(cv_frame);
        }

        // Downsample and publish image
        cv::Mat downsampled_frame = downsampleImage(cv_frame);
        publishImage(downsampled_frame);

        // Requeue buffer
        device_->RequeueBuffer(arena_image);

        // prev_time = curr_time;

      } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error getting image: %s", e.what());
      }

      // Check for ROS shutdown
      rclcpp::spin_some(this->get_node_base_interface());
    }

    RCLCPP_INFO(this->get_logger(), "Stopping camera stream...");
    device_->StopStream();
    RCLCPP_INFO(this->get_logger(), "Camera stream stopped successfully");
  }

  cv::Mat convertArenaImageToCV(Arena::IImage* arena_image) {
    // Get image properties
    size_t width = arena_image->GetWidth();
    size_t height = arena_image->GetHeight();

    // Get pointer to raw Bayer image data
    const uint8_t* data = static_cast<const uint8_t*>(arena_image->GetData());

    // Create OpenCV Mat from raw Bayer data (single channel, 8-bit)
    cv::Mat bayer_frame(height, width, CV_8UC1);
    memcpy(bayer_frame.data, data, height * width);

    // Upload Bayer image to GPU
    gpu_bayer_.upload(bayer_frame);

    // Perform CUDA-accelerated Bayer demosaicing (BayerRG to BGR)
    // cv::COLOR_BayerRG2BGR is the correct code for BayerRG8 pattern
    cv::cuda::demosaicing(gpu_bayer_, gpu_bgr_, cv::COLOR_BayerRG2BGR);

    // Download result from GPU to CPU
    cv::Mat bgr_frame;
    gpu_bgr_.download(bgr_frame);

    return bgr_frame;
  }


  void saveImage(const cv::Mat& frame) {
    try {
      // Generate filename with timestamp and counter
      auto now = std::chrono::system_clock::now();
      auto time_t_now = std::chrono::system_clock::to_time_t(now);
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
          now.time_since_epoch()) % 1000;
      
      std::stringstream filename;
      filename << save_folder_ << "/image_"
               << std::put_time(std::localtime(&time_t_now), "%Y%m%d_%H%M%S")
               << "_" << std::setfill('0') << std::setw(3) << ms.count()
               << "_" << std::setfill('0') << std::setw(6) << image_counter_
               << ".png";
      
      cv::imwrite(filename.str(), frame);
      image_counter_++;
      
      RCLCPP_DEBUG(this->get_logger(), "Saved image: %s", filename.str().c_str());
    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Failed to save image: %s", e.what());
    }
  }

  cv::Mat downsampleImage(const cv::Mat& frame) {
    // If downsampling factor is 1, return original frame
    if (publish_downsampling_factor_ == 1) {
      return frame;
    }

    // Upload frame to GPU
    cv::cuda::GpuMat gpu_input, gpu_output;
    gpu_input.upload(frame);

    // Calculate output dimensions
    int output_width = frame.cols / publish_downsampling_factor_;
    int output_height = frame.rows / publish_downsampling_factor_;

    // Perform CUDA-accelerated resize
    cv::cuda::resize(gpu_input, gpu_output, 
                     cv::Size(output_width, output_height),
                     0, 0, cv::INTER_LINEAR);

    // Download result from GPU
    cv::Mat downsampled;
    gpu_output.download(downsampled);

    return downsampled;
  }

  void publishImage(const cv::Mat& frame) {
    try {
      // Convert OpenCV Mat to ROS Image message
      cv_bridge::CvImage cv_image;
      cv_image.header.stamp = this->now();
      cv_image.header.frame_id = "camera_optical_frame";
      cv_image.encoding = "bgr8";
      cv_image.image = frame;

      // Publish the image
      image_pub_->publish(*cv_image.toImageMsg());
      RCLCPP_INFO(this->get_logger(), "Published Image :D");

    } catch (const cv_bridge::Exception& e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
  }

 private:
  // Parameters
  std::string camera_ip_;
  std::string topic_name_;
  int horizontal_binning_;
  int vertical_binning_;
  int image_height_;
  int image_width_;
  std::string save_folder_;
  int publish_downsampling_factor_;
  int image_offset_x_;
  int image_offset_y_;
  size_t image_counter_;

  // Arena SDK objects
  Arena::ISystem* system_ = nullptr;
  Arena::IDevice* device_ = nullptr;
  GenApi::INodeMap* device_nodemap_ = nullptr; // todo: add destructors

  // Camera IP
  uint32_t device_ip_ = 0;

  // Scheduled Action Triggering Parameters
  int64_t _g_action_device_key = 1;
  int64_t _g_action_group_key = 1;
  int64_t _g_action_group_mask = 1;

  int output_width_;
  int output_height_;

  // ROS publisher
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;

  // CUDA GpuMat buffers for GPU-accelerated processing
  cv::cuda::GpuMat gpu_bayer_;
  cv::cuda::GpuMat gpu_bgr_;
  
  // Shutdown flag
  std::atomic<bool> shutdown_requested_;
  
  // Static instance pointer for signal handler
  static LucidDriver* instance_;
};

// Initialize static member
LucidDriver* LucidDriver::instance_ = nullptr;

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  try {
    auto node = std::make_shared<LucidDriver>();
    node->start();
    rclcpp::spin(node);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(rclcpp::get_logger("lucid_driver"), "Exception in main: %s", e.what());
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}