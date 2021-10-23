;; Auto-generated. Do not edit!


(when (boundp 'uuv_gazebo_ros_plugins_msgs::SetThrusterState)
  (if (not (find-package "UUV_GAZEBO_ROS_PLUGINS_MSGS"))
    (make-package "UUV_GAZEBO_ROS_PLUGINS_MSGS"))
  (shadow 'SetThrusterState (find-package "UUV_GAZEBO_ROS_PLUGINS_MSGS")))
(unless (find-package "UUV_GAZEBO_ROS_PLUGINS_MSGS::SETTHRUSTERSTATE")
  (make-package "UUV_GAZEBO_ROS_PLUGINS_MSGS::SETTHRUSTERSTATE"))
(unless (find-package "UUV_GAZEBO_ROS_PLUGINS_MSGS::SETTHRUSTERSTATEREQUEST")
  (make-package "UUV_GAZEBO_ROS_PLUGINS_MSGS::SETTHRUSTERSTATEREQUEST"))
(unless (find-package "UUV_GAZEBO_ROS_PLUGINS_MSGS::SETTHRUSTERSTATERESPONSE")
  (make-package "UUV_GAZEBO_ROS_PLUGINS_MSGS::SETTHRUSTERSTATERESPONSE"))

(in-package "ROS")





(defclass uuv_gazebo_ros_plugins_msgs::SetThrusterStateRequest
  :super ros::object
  :slots (_on ))

(defmethod uuv_gazebo_ros_plugins_msgs::SetThrusterStateRequest
  (:init
   (&key
    ((:on __on) nil)
    )
   (send-super :init)
   (setq _on __on)
   self)
  (:on
   (&optional (__on :null))
   (if (not (eq __on :null)) (setq _on __on)) _on)
  (:serialization-length
   ()
   (+
    ;; bool _on
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _on
       (if _on (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _on
     (setq _on (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(defclass uuv_gazebo_ros_plugins_msgs::SetThrusterStateResponse
  :super ros::object
  :slots (_success ))

(defmethod uuv_gazebo_ros_plugins_msgs::SetThrusterStateResponse
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

(defclass uuv_gazebo_ros_plugins_msgs::SetThrusterState
  :super ros::object
  :slots ())

(setf (get uuv_gazebo_ros_plugins_msgs::SetThrusterState :md5sum-) "7abefa06f2be45ab014b8164e7591960")
(setf (get uuv_gazebo_ros_plugins_msgs::SetThrusterState :datatype-) "uuv_gazebo_ros_plugins_msgs/SetThrusterState")
(setf (get uuv_gazebo_ros_plugins_msgs::SetThrusterState :request) uuv_gazebo_ros_plugins_msgs::SetThrusterStateRequest)
(setf (get uuv_gazebo_ros_plugins_msgs::SetThrusterState :response) uuv_gazebo_ros_plugins_msgs::SetThrusterStateResponse)

(defmethod uuv_gazebo_ros_plugins_msgs::SetThrusterStateRequest
  (:response () (instance uuv_gazebo_ros_plugins_msgs::SetThrusterStateResponse :init)))

(setf (get uuv_gazebo_ros_plugins_msgs::SetThrusterStateRequest :md5sum-) "7abefa06f2be45ab014b8164e7591960")
(setf (get uuv_gazebo_ros_plugins_msgs::SetThrusterStateRequest :datatype-) "uuv_gazebo_ros_plugins_msgs/SetThrusterStateRequest")
(setf (get uuv_gazebo_ros_plugins_msgs::SetThrusterStateRequest :definition-)
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

bool on
---
bool success

")

(setf (get uuv_gazebo_ros_plugins_msgs::SetThrusterStateResponse :md5sum-) "7abefa06f2be45ab014b8164e7591960")
(setf (get uuv_gazebo_ros_plugins_msgs::SetThrusterStateResponse :datatype-) "uuv_gazebo_ros_plugins_msgs/SetThrusterStateResponse")
(setf (get uuv_gazebo_ros_plugins_msgs::SetThrusterStateResponse :definition-)
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

bool on
---
bool success

")



(provide :uuv_gazebo_ros_plugins_msgs/SetThrusterState "7abefa06f2be45ab014b8164e7591960")


