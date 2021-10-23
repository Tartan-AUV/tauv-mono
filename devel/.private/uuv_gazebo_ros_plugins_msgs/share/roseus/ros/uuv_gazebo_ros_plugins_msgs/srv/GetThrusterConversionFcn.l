;; Auto-generated. Do not edit!


(when (boundp 'uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcn)
  (if (not (find-package "UUV_GAZEBO_ROS_PLUGINS_MSGS"))
    (make-package "UUV_GAZEBO_ROS_PLUGINS_MSGS"))
  (shadow 'GetThrusterConversionFcn (find-package "UUV_GAZEBO_ROS_PLUGINS_MSGS")))
(unless (find-package "UUV_GAZEBO_ROS_PLUGINS_MSGS::GETTHRUSTERCONVERSIONFCN")
  (make-package "UUV_GAZEBO_ROS_PLUGINS_MSGS::GETTHRUSTERCONVERSIONFCN"))
(unless (find-package "UUV_GAZEBO_ROS_PLUGINS_MSGS::GETTHRUSTERCONVERSIONFCNREQUEST")
  (make-package "UUV_GAZEBO_ROS_PLUGINS_MSGS::GETTHRUSTERCONVERSIONFCNREQUEST"))
(unless (find-package "UUV_GAZEBO_ROS_PLUGINS_MSGS::GETTHRUSTERCONVERSIONFCNRESPONSE")
  (make-package "UUV_GAZEBO_ROS_PLUGINS_MSGS::GETTHRUSTERCONVERSIONFCNRESPONSE"))

(in-package "ROS")





(defclass uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcnRequest
  :super ros::object
  :slots ())

(defmethod uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcnRequest
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

(defclass uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcnResponse
  :super ros::object
  :slots (_fcn ))

(defmethod uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcnResponse
  (:init
   (&key
    ((:fcn __fcn) (instance uuv_gazebo_ros_plugins_msgs::ThrusterConversionFcn :init))
    )
   (send-super :init)
   (setq _fcn __fcn)
   self)
  (:fcn
   (&rest __fcn)
   (if (keywordp (car __fcn))
       (send* _fcn __fcn)
     (progn
       (if __fcn (setq _fcn (car __fcn)))
       _fcn)))
  (:serialization-length
   ()
   (+
    ;; uuv_gazebo_ros_plugins_msgs/ThrusterConversionFcn _fcn
    (send _fcn :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; uuv_gazebo_ros_plugins_msgs/ThrusterConversionFcn _fcn
       (send _fcn :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; uuv_gazebo_ros_plugins_msgs/ThrusterConversionFcn _fcn
     (send _fcn :deserialize buf ptr-) (incf ptr- (send _fcn :serialization-length))
   ;;
   self)
  )

(defclass uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcn
  :super ros::object
  :slots ())

(setf (get uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcn :md5sum-) "b489744fdf1ea3660acd86f33ee041a7")
(setf (get uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcn :datatype-) "uuv_gazebo_ros_plugins_msgs/GetThrusterConversionFcn")
(setf (get uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcn :request) uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcnRequest)
(setf (get uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcn :response) uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcnResponse)

(defmethod uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcnRequest
  (:response () (instance uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcnResponse :init)))

(setf (get uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcnRequest :md5sum-) "b489744fdf1ea3660acd86f33ee041a7")
(setf (get uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcnRequest :datatype-) "uuv_gazebo_ros_plugins_msgs/GetThrusterConversionFcnRequest")
(setf (get uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcnRequest :definition-)
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
uuv_gazebo_ros_plugins_msgs/ThrusterConversionFcn fcn


================================================================================
MSG: uuv_gazebo_ros_plugins_msgs/ThrusterConversionFcn
# Copyright (c) 2016 The UUV Simulator Authors.
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

string function_name
string[] tags
float64[] data
float64[] lookup_table_input
float64[] lookup_table_output
")

(setf (get uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcnResponse :md5sum-) "b489744fdf1ea3660acd86f33ee041a7")
(setf (get uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcnResponse :datatype-) "uuv_gazebo_ros_plugins_msgs/GetThrusterConversionFcnResponse")
(setf (get uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcnResponse :definition-)
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
uuv_gazebo_ros_plugins_msgs/ThrusterConversionFcn fcn


================================================================================
MSG: uuv_gazebo_ros_plugins_msgs/ThrusterConversionFcn
# Copyright (c) 2016 The UUV Simulator Authors.
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

string function_name
string[] tags
float64[] data
float64[] lookup_table_input
float64[] lookup_table_output
")



(provide :uuv_gazebo_ros_plugins_msgs/GetThrusterConversionFcn "b489744fdf1ea3660acd86f33ee041a7")


