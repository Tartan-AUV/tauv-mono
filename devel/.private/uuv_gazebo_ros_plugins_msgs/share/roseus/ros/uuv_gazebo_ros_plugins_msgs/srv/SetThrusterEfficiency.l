;; Auto-generated. Do not edit!


(when (boundp 'uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiency)
  (if (not (find-package "UUV_GAZEBO_ROS_PLUGINS_MSGS"))
    (make-package "UUV_GAZEBO_ROS_PLUGINS_MSGS"))
  (shadow 'SetThrusterEfficiency (find-package "UUV_GAZEBO_ROS_PLUGINS_MSGS")))
(unless (find-package "UUV_GAZEBO_ROS_PLUGINS_MSGS::SETTHRUSTEREFFICIENCY")
  (make-package "UUV_GAZEBO_ROS_PLUGINS_MSGS::SETTHRUSTEREFFICIENCY"))
(unless (find-package "UUV_GAZEBO_ROS_PLUGINS_MSGS::SETTHRUSTEREFFICIENCYREQUEST")
  (make-package "UUV_GAZEBO_ROS_PLUGINS_MSGS::SETTHRUSTEREFFICIENCYREQUEST"))
(unless (find-package "UUV_GAZEBO_ROS_PLUGINS_MSGS::SETTHRUSTEREFFICIENCYRESPONSE")
  (make-package "UUV_GAZEBO_ROS_PLUGINS_MSGS::SETTHRUSTEREFFICIENCYRESPONSE"))

(in-package "ROS")





(defclass uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiencyRequest
  :super ros::object
  :slots (_efficiency ))

(defmethod uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiencyRequest
  (:init
   (&key
    ((:efficiency __efficiency) 0.0)
    )
   (send-super :init)
   (setq _efficiency (float __efficiency))
   self)
  (:efficiency
   (&optional __efficiency)
   (if __efficiency (setq _efficiency __efficiency)) _efficiency)
  (:serialization-length
   ()
   (+
    ;; float64 _efficiency
    8
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float64 _efficiency
       (sys::poke _efficiency (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float64 _efficiency
     (setq _efficiency (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;;
   self)
  )

(defclass uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiencyResponse
  :super ros::object
  :slots (_success ))

(defmethod uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiencyResponse
  (:init
   (&key
    ((:success __success) nil)
    )
   (send-super :init)
   (setq _success __success)
   self)
  (:success
   (&optional (__success :null))
   (if (not (eq __success :null)) (setq _success __success)) _success)
  (:serialization-length
   ()
   (+
    ;; bool _success
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _success
       (if _success (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _success
     (setq _success (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(defclass uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiency
  :super ros::object
  :slots ())

(setf (get uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiency :md5sum-) "60f827235457ddfd6a19b596030b49ad")
(setf (get uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiency :datatype-) "uuv_gazebo_ros_plugins_msgs/SetThrusterEfficiency")
(setf (get uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiency :request) uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiencyRequest)
(setf (get uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiency :response) uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiencyResponse)

(defmethod uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiencyRequest
  (:response () (instance uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiencyResponse :init)))

(setf (get uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiencyRequest :md5sum-) "60f827235457ddfd6a19b596030b49ad")
(setf (get uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiencyRequest :datatype-) "uuv_gazebo_ros_plugins_msgs/SetThrusterEfficiencyRequest")
(setf (get uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiencyRequest :definition-)
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

float64 efficiency
---
bool success

")

(setf (get uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiencyResponse :md5sum-) "60f827235457ddfd6a19b596030b49ad")
(setf (get uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiencyResponse :datatype-) "uuv_gazebo_ros_plugins_msgs/SetThrusterEfficiencyResponse")
(setf (get uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiencyResponse :definition-)
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

float64 efficiency
---
bool success

")



(provide :uuv_gazebo_ros_plugins_msgs/SetThrusterEfficiency "60f827235457ddfd6a19b596030b49ad")


