;; Auto-generated. Do not edit!


(when (boundp 'uuv_gazebo_ros_plugins_msgs::GetFloat)
  (if (not (find-package "UUV_GAZEBO_ROS_PLUGINS_MSGS"))
    (make-package "UUV_GAZEBO_ROS_PLUGINS_MSGS"))
  (shadow 'GetFloat (find-package "UUV_GAZEBO_ROS_PLUGINS_MSGS")))
(unless (find-package "UUV_GAZEBO_ROS_PLUGINS_MSGS::GETFLOAT")
  (make-package "UUV_GAZEBO_ROS_PLUGINS_MSGS::GETFLOAT"))
(unless (find-package "UUV_GAZEBO_ROS_PLUGINS_MSGS::GETFLOATREQUEST")
  (make-package "UUV_GAZEBO_ROS_PLUGINS_MSGS::GETFLOATREQUEST"))
(unless (find-package "UUV_GAZEBO_ROS_PLUGINS_MSGS::GETFLOATRESPONSE")
  (make-package "UUV_GAZEBO_ROS_PLUGINS_MSGS::GETFLOATRESPONSE"))

(in-package "ROS")





(defclass uuv_gazebo_ros_plugins_msgs::GetFloatRequest
  :super ros::object
  :slots ())

(defmethod uuv_gazebo_ros_plugins_msgs::GetFloatRequest
  (:init
   (&key
    )
   (send-super :init)
   self)
  (:serialization-length
   ()
   (+
    0
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;;
   self)
  )

(defclass uuv_gazebo_ros_plugins_msgs::GetFloatResponse
  :super ros::object
  :slots (_data ))

(defmethod uuv_gazebo_ros_plugins_msgs::GetFloatResponse
  (:init
   (&key
    ((:data __data) 0.0)
    )
   (send-super :init)
   (setq _data (float __data))
   self)
  (:data
   (&optional __data)
   (if __data (setq _data __data)) _data)
  (:serialization-length
   ()
   (+
    ;; float64 _data
    8
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float64 _data
       (sys::poke _data (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float64 _data
     (setq _data (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;;
   self)
  )

(defclass uuv_gazebo_ros_plugins_msgs::GetFloat
  :super ros::object
  :slots ())

(setf (get uuv_gazebo_ros_plugins_msgs::GetFloat :md5sum-) "fdb28210bfa9d7c91146260178d9a584")
(setf (get uuv_gazebo_ros_plugins_msgs::GetFloat :datatype-) "uuv_gazebo_ros_plugins_msgs/GetFloat")
(setf (get uuv_gazebo_ros_plugins_msgs::GetFloat :request) uuv_gazebo_ros_plugins_msgs::GetFloatRequest)
(setf (get uuv_gazebo_ros_plugins_msgs::GetFloat :response) uuv_gazebo_ros_plugins_msgs::GetFloatResponse)

(defmethod uuv_gazebo_ros_plugins_msgs::GetFloatRequest
  (:response () (instance uuv_gazebo_ros_plugins_msgs::GetFloatResponse :init)))

(setf (get uuv_gazebo_ros_plugins_msgs::GetFloatRequest :md5sum-) "fdb28210bfa9d7c91146260178d9a584")
(setf (get uuv_gazebo_ros_plugins_msgs::GetFloatRequest :datatype-) "uuv_gazebo_ros_plugins_msgs/GetFloatRequest")
(setf (get uuv_gazebo_ros_plugins_msgs::GetFloatRequest :definition-)
      "# Copyright (c) 2016 The UUV Simulator Authors.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

---
float64 data

")

(setf (get uuv_gazebo_ros_plugins_msgs::GetFloatResponse :md5sum-) "fdb28210bfa9d7c91146260178d9a584")
(setf (get uuv_gazebo_ros_plugins_msgs::GetFloatResponse :datatype-) "uuv_gazebo_ros_plugins_msgs/GetFloatResponse")
(setf (get uuv_gazebo_ros_plugins_msgs::GetFloatResponse :definition-)
      "# Copyright (c) 2016 The UUV Simulator Authors.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

---
float64 data

")



(provide :uuv_gazebo_ros_plugins_msgs/GetFloat "fdb28210bfa9d7c91146260178d9a584")


