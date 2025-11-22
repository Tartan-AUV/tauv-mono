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

// VPI includes
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/Rescale.h>

#include <vpi/OpenCVInterop.hpp>

// VPI error checking macro
#define CHECK_VPI_STATUS(STMT)                          \
  do {                                                  \
    VPIStatus status = (STMT);                          \
    if (status != VPI_SUCCESS) {                        \
      char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];       \
      vpiGetLastStatusMessage(buffer, sizeof(buffer));  \
      std::ostringstream ss;                            \
      ss << vpiStatusGetName(status) << ": " << buffer; \
      throw std::runtime_error(ss.str());               \
    }                                                   \
  } while (0)

class LucidDriver : public rclcpp::Node {
 public:
  LucidDriver()
      : Node("lucid"),
        vpi_stream_(nullptr),
        vpi_input_(nullptr),
        vpi_temp_nv12_(nullptr),
        vpi_scaled_nv12_(nullptr),
        vpi_output_(nullptr) {
    // Declare parameters
    this->declare_parameter("camera_ip", "10.0.1.11");
    this->declare_parameter("topic_name", "/image_raw");
    this->declare_parameter("horizontal_binning", 4);
    this->declare_parameter("vertical_binning", 4);
    this->declare_parameter("vpi_backend", "cuda");  // Options: cuda, vic, cpu

    // Get parameters
    camera_ip_ = this->get_parameter("camera_ip").as_string();
    topic_name_ = this->get_parameter("topic_name").as_string();
    horizontal_binning_ = this->get_parameter("horizontal_binning").as_int();
    vertical_binning_ = this->get_parameter("vertical_binning").as_int();

    // Get VPI backend
    std::string backend_str = this->get_parameter("vpi_backend").as_string();
    if (backend_str == "cpu") {
      vpi_backend_ = VPI_BACKEND_CPU;
    } else if (backend_str == "vic") {
      vpi_backend_ = VPI_BACKEND_VIC;
    } else {
      vpi_backend_ = VPI_BACKEND_CUDA;
    }

    // Set up QoS profile
    auto qos = rclcpp::QoS(10);
    qos.reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);
    qos.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);

    // Create publisher
    image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(topic_name_, qos);

    RCLCPP_INFO(this->get_logger(),
                "Lucid Node Initialized with Camera IP: %s, Topic: %s, VPI Backend: %s",
                camera_ip_.c_str(), topic_name_.c_str(), backend_str.c_str());
  }

  ~LucidDriver() {
    // Clean up VPI resources
    cleanupVPI();

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
    device_ = getDeviceByIP(camera_ip_);
    if (device_ == nullptr) {
      throw std::runtime_error("Device with IP " + camera_ip_ + " not found");
    }

    // Setup camera configuration
    setupCamera();

    // Start streaming
    streamImages();
  }

 private:
  Arena::IDevice* getDeviceByIP(const std::string& ip) {
    system_->UpdateDevices(100);
    std::vector<Arena::DeviceInfo> deviceInfos = system_->GetDevices();

    for (const auto& deviceInfo : deviceInfos) {
      if (std::string(const_cast<Arena::DeviceInfo&>(deviceInfo).IpAddressStr()) == ip) {
        RCLCPP_INFO(this->get_logger(), "Found device at IP: %s", ip.c_str());
        return system_->CreateDevice(deviceInfo);
      }
    }

    // Device not found, wait and retry
    for (int tries = 0; tries < 6; tries++) {
      RCLCPP_INFO(this->get_logger(),
                  "Try %d of 6: waiting for 10 secs for device to be connected!", tries + 1);
      std::this_thread::sleep_for(std::chrono::seconds(10));

      system_->UpdateDevices(100);
      deviceInfos = system_->GetDevices();

      for (auto& deviceInfo : deviceInfos) {
        if (std::string(deviceInfo.IpAddressStr()) == ip) {
          RCLCPP_INFO(this->get_logger(), "Found device at IP: %s", ip.c_str());
          return system_->CreateDevice(deviceInfo);
        }
      }
    }

    return nullptr;
  }

  void setupCamera() {
    auto nodemap = device_->GetNodeMap();
    auto stream_nodemap = device_->GetTLStreamNodeMap();

    // Stop stream if running
    try {
      device_->StopStream();
    } catch (...) {
      // Stream might not be running
    }

    std::cout << "Configuring camera...\n";

    // Set Acquisition Frame Rate
    Arena::SetNodeValue<bool>(nodemap, "AcquisitionFrameRateEnable", true);
    Arena::SetNodeValue<double>(nodemap, "AcquisitionFrameRate", 10.0);

    std::cout << "Set acquisition frmae rate...\n";

    // Set Acquisition Mode
    Arena::SetNodeValue<GenICam::gcstring>(nodemap, "AcquisitionMode", "Continuous");

    // Configure Binning
    Arena::SetNodeValue<GenICam::gcstring>(nodemap, "BinningSelector", "Digital");
    Arena::SetNodeValue<GenICam::gcstring>(nodemap, "BinningHorizontalMode", "Sum");
    Arena::SetNodeValue<GenICam::gcstring>(nodemap, "BinningVerticalMode", "Sum");

    // For now, set binning to 1 (no binning) as in the Python version
    Arena::SetNodeValue<int64_t>(nodemap, "BinningHorizontal", 4);
    Arena::SetNodeValue<int64_t>(nodemap, "BinningVertical", 4);

    // Set Device Stream Channel Packet Size
    try {
      Arena::SetNodeValue<int64_t>(nodemap, "DeviceStreamChannelPacketSize", 9000);
    } catch (...) {
      RCLCPP_WARN(this->get_logger(), "Failed to set DeviceStreamChannelPacketSize");
    }

    // Set image dimensions
    Arena::SetNodeValue<int64_t>(nodemap, "Height", 758);
    Arena::SetNodeValue<int64_t>(nodemap, "Width", 1328);
    Arena::SetNodeValue<int64_t>(nodemap, "OffsetX", 0);
    Arena::SetNodeValue<int64_t>(nodemap, "OffsetY", 0);

    // Set pixel format
    Arena::SetNodeValue<GenICam::gcstring>(nodemap, "PixelFormat", "RGB8");

    // Set transport protocol to TCP
    try {
      Arena::SetNodeValue<GenICam::gcstring>(nodemap, "TransportStreamProtocol", "TCP");
    } catch (...) {
      RCLCPP_WARN(this->get_logger(), "Failed to set TransportStreamProtocol to TCP");
    }

    // Configure trigger for pulsing
    configureTriggering(nodemap);

    // Stream nodemap configuration
    Arena::SetNodeValue<GenICam::gcstring>(stream_nodemap, "StreamBufferHandlingMode",
                                           "NewestOnly");
    Arena::SetNodeValue<bool>(stream_nodemap, "StreamAutoNegotiatePacketSize", true);
    Arena::SetNodeValue<bool>(stream_nodemap, "StreamPacketResendEnable", true);
  }

  void configureTriggering(GenApi::INodeMap* nodemap) {
    try {
      // Set Line2 as input for trigger
      Arena::SetNodeValue<GenICam::gcstring>(nodemap, "LineSelector", "Line2");
      Arena::SetNodeValue<GenICam::gcstring>(nodemap, "LineMode", "Input");

      // Configure trigger
      Arena::SetNodeValue<GenICam::gcstring>(nodemap, "TriggerMode", "On");
      Arena::SetNodeValue<GenICam::gcstring>(nodemap, "TriggerSelector", "FrameStart");
      Arena::SetNodeValue<GenICam::gcstring>(nodemap, "TriggerSource", "Line0");
      Arena::SetNodeValue<GenICam::gcstring>(nodemap, "TriggerActivation", "FallingEdge");
    } catch (const std::exception& e) {
      RCLCPP_WARN(this->get_logger(), "Failed to configure triggering: %s", e.what());
    }
  }

  void streamImages() {
    device_->StartStream();

    // Initialize VPI resources after we know the image dimensions
    bool vpi_initialized = false;

    auto prev_time = std::chrono::steady_clock::now();

    while (rclcpp::ok()) {
      auto curr_time = std::chrono::steady_clock::now();

      try {
        // Get image from camera with 2 second timeout
        Arena::IImage* arena_image = device_->GetImage(2000);

        // Convert Arena image to OpenCV Mat
        cv::Mat cv_frame = convertArenaImageToCV(arena_image);

        // Initialize VPI on first frame
        if (!vpi_initialized) {
          initializeVPI(cv_frame.cols, cv_frame.rows);
          vpi_initialized = true;
        }

        // VPI processing
        cv::Mat output_frame = processWithVPI(cv_frame);

        // Calculate and log FPS
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(curr_time - prev_time);
        double fps = 1000.0 / duration.count();
        RCLCPP_INFO(this->get_logger(), "FPS: %.2f", fps);
        RCLCPP_INFO(this->get_logger(), "Image Size (%d, %d)", output_frame.cols,
                    output_frame.rows);

        // Publish image
        publishImage(output_frame);

        // Requeue buffer
        device_->RequeueBuffer(arena_image);

        prev_time = curr_time;

      } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error getting image: %s", e.what());
      }

      // Check for ROS shutdown
      rclcpp::spin_some(this->get_node_base_interface());
    }

    device_->StopStream();
  }

  cv::Mat convertArenaImageToCV(Arena::IImage* arena_image) {
    // Get image properties
    size_t width = arena_image->GetWidth();
    size_t height = arena_image->GetHeight();
    size_t bytes_per_pixel = 3;  // RGB8

    // Get pointer to image data
    const uint8_t* data = static_cast<const uint8_t*>(arena_image->GetData());

    // Create OpenCV Mat from Arena image data (RGB format)
    // Make a copy since cv::Mat constructor needs non-const data pointer
    cv::Mat rgb_frame(height, width, CV_8UC3);
    memcpy(rgb_frame.data, data, height * width * 3);

    // Convert RGB to BGR for OpenCV processing
    cv::Mat bgr_frame;
    cv::cvtColor(rgb_frame, bgr_frame, cv::COLOR_RGB2BGR);

    return bgr_frame;
  }

  void initializeVPI(int width, int height) {
    try {
      // Create VPI stream for the selected backend
      // Also enable CUDA for format conversion
      CHECK_VPI_STATUS(vpiStreamCreate(vpi_backend_ | VPI_BACKEND_CUDA, &vpi_stream_));

      // Calculate output dimensions
      output_width_ = width / horizontal_binning_;
      output_height_ = height / vertical_binning_;

      // Create VPI images for processing pipeline
      // Input will be wrapped from OpenCV Mat
      // Temporary NV12 image for input conversion
      CHECK_VPI_STATUS(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_NV12_ER, 0, &vpi_temp_nv12_));

      // Scaled NV12 image
      CHECK_VPI_STATUS(vpiImageCreate(output_width_, output_height_, VPI_IMAGE_FORMAT_NV12_ER, 0,
                                      &vpi_scaled_nv12_));

      // Output RGB8 image
      CHECK_VPI_STATUS(
          vpiImageCreate(output_width_, output_height_, VPI_IMAGE_FORMAT_RGB8, 0, &vpi_output_));

      RCLCPP_INFO(
          this->get_logger(), "VPI initialized: %dx%d -> %dx%d, Backend: %s", width, height,
          output_width_, output_height_,
          (vpi_backend_ == VPI_BACKEND_CPU ? "CPU"
                                           : (vpi_backend_ == VPI_BACKEND_CUDA ? "CUDA" : "VIC")));

    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Failed to initialize VPI: %s", e.what());
      throw;
    }
  }

  void cleanupVPI() {
    // Synchronize stream before destroying objects
    if (vpi_stream_ != nullptr) {
      vpiStreamSync(vpi_stream_);
    }

    // Destroy VPI images
    if (vpi_input_ != nullptr) {
      vpiImageDestroy(vpi_input_);
      vpi_input_ = nullptr;
    }
    if (vpi_temp_nv12_ != nullptr) {
      vpiImageDestroy(vpi_temp_nv12_);
      vpi_temp_nv12_ = nullptr;
    }
    if (vpi_scaled_nv12_ != nullptr) {
      vpiImageDestroy(vpi_scaled_nv12_);
      vpi_scaled_nv12_ = nullptr;
    }
    if (vpi_output_ != nullptr) {
      vpiImageDestroy(vpi_output_);
      vpi_output_ = nullptr;
    }

    // Destroy stream
    if (vpi_stream_ != nullptr) {
      vpiStreamDestroy(vpi_stream_);
      vpi_stream_ = nullptr;
    }
  }

  cv::Mat processWithVPI(const cv::Mat& input) {
    try {
      // Wrap input OpenCV Mat in VPI image
      // Note: VPI_IMAGE_FORMAT_BGR8 for OpenCV's BGR format
      // We recreate the wrapper each time since the cv::Mat data pointer might change
      if (vpi_input_ != nullptr) {
        vpiImageDestroy(vpi_input_);
      }
      CHECK_VPI_STATUS(
          vpiImageCreateWrapperOpenCVMat(input, VPI_IMAGE_FORMAT_BGR8, 0, &vpi_input_));

      // Convert BGR8 to NV12_ER using CUDA backend for efficiency
      CHECK_VPI_STATUS(vpiSubmitConvertImageFormat(vpi_stream_, VPI_BACKEND_CUDA, vpi_input_,
                                                   vpi_temp_nv12_, nullptr));

      // Rescale the image using the selected backend
      CHECK_VPI_STATUS(vpiSubmitRescale(vpi_stream_, vpi_backend_, vpi_temp_nv12_, vpi_scaled_nv12_,
                                        VPI_INTERP_LINEAR, VPI_BORDER_CLAMP, 0));

      // Convert back to RGB8 using CUDA
      CHECK_VPI_STATUS(vpiSubmitConvertImageFormat(vpi_stream_, VPI_BACKEND_CUDA, vpi_scaled_nv12_,
                                                   vpi_output_, nullptr));

      // Wait for processing to complete
      CHECK_VPI_STATUS(vpiStreamSync(vpi_stream_));

      // Get the output image data
      VPIImageData outData;
      CHECK_VPI_STATUS(vpiImageLockData(vpi_output_, VPI_LOCK_READ,
                                        VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &outData));

      // Create OpenCV Mat from VPI output
      VPIImageBufferPitchLinear& outPitch = outData.buffer.pitch;
      cv::Mat output(outPitch.planes[0].height, outPitch.planes[0].width, CV_8UC3,
                     outPitch.planes[0].data, outPitch.planes[0].pitchBytes);

      // Make a copy since we'll unlock the VPI image
      cv::Mat result = output.clone();

      // Unlock the VPI image
      CHECK_VPI_STATUS(vpiImageUnlock(vpi_output_));

      return result;

    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "VPI processing failed: %s", e.what());
      // Fall back to OpenCV resize
      cv::Mat output;
      cv::resize(input, output, cv::Size(output_width_, output_height_));
      return output;
    }
  }

  void publishImage(const cv::Mat& frame) {
    try {
      // Convert OpenCV Mat to ROS Image message
      cv_bridge::CvImage cv_image;
      cv_image.header.stamp = this->now();
      cv_image.header.frame_id = "camera_optical_frame";
      cv_image.encoding = "rgb8";
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

  // Arena SDK objects
  Arena::ISystem* system_ = nullptr;
  Arena::IDevice* device_ = nullptr;

  // VPI objects
  VPIStream vpi_stream_;
  VPIImage vpi_input_;
  VPIImage vpi_temp_nv12_;
  VPIImage vpi_scaled_nv12_;
  VPIImage vpi_output_;
  VPIBackend vpi_backend_;
  int output_width_;
  int output_height_;

  // ROS publisher
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
};

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