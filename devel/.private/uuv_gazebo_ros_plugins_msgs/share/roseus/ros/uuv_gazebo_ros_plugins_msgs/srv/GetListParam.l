;; Auto-generated. Do not edit!


(when (boundp 'uuv_gazebo_ros_plugins_msgs::GetListParam)
  (if (not (find-package "UUV_GAZEBO_ROS_PLUGINS_MSGS"))
    (make-package "UUV_GAZEBO_ROS_PLUGINS_MSGS"))
  (shadow 'GetListParam (find-package "UUV_GAZEBO_ROS_PLUGINS_MSGS")))
(unless (find-package "UUV_GAZEBO_ROS_PLUGINS_MSGS::GETLISTPARAM")
  (make-package "UUV_GAZEBO_ROS_PLUGINS_MSGS::GETLISTPARAM"))
(unless (find-package "UUV_GAZEBO_ROS_PLUGINS_MSGS::GETLISTPARAMREQUEST")
  (make-package "UUV_GAZEBO_ROS_PLUGINS_MSGS::GETLISTPARAMREQUEST"))
(unless (find-package "UUV_GAZEBO_ROS_PLUGINS_MSGS::GETLISTPARAMRESPONSE")
  (make-package "UUV_GAZEBO_ROS_PLUGINS_MSGS::GETLISTPARAMRESPONSE"))

(in-package "ROS")





(defclass uuv_gazebo_ros_plugins_msgs::GetListParamRequest
  :super ros::object
  :slots ())

(defmethod uuv_gazebo_ros_plugins_msgs::GetListParamRequest
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

(defclass uuv_gazebo_ros_plugins_msgs::GetListParamResponse
  :super ros::object
  :slots (_description _tags _data ))

(defmethod uuv_gazebo_ros_plugins_msgs::GetListParamResponse
  (:init
   (&key
    ((:description __description) "")
    ((:tags __tags) (let (r) (dotimes (i 0) (push "" r)) r))
    ((:data __data) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _description (string __description))
   (setq _tags __tags)
   (setq _data __data)
   self)
  (:description
   (&optional __description)
   (if __description (setq _description __description)) _description)
  (:tags
   (&optional __tags)
   (if __tags (setq _tags __tags)) _tags)
  (:data
   (&optional __data)
   (if __data (setq _data __data)) _data)
  (:serialization-length
   ()
   (+
    ;; string _description
    4 (length _description)
    ;; string[] _tags
    (apply #'+ (mapcar #'(lambda (x) (+ 4 (length x))) _tags)) 4
    ;; float64[] _data
    (* 8    (length _data)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _description
       (write-long (length _description) s) (princ _description s)
     ;; string[] _tags
     (write-long (length _tags) s)
     (dolist (elem _tags)
       (write-long (length elem) s) (princ elem s)
       )
     ;; float64[] _data
     (write-long (length _data) s)
     (dotimes (i (length _data))
       (sys::poke (elt _data i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _description
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _description (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string[] _tags
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _tags (make-list n))
     (dotimes (i n)
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setf (elt _tags i) (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
     ))
   ;; float64[] _data
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _data (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _data i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;;
   self)
  )

(defclass uuv_gazebo_ros_plugins_msgs::GetListParam
  :super ros::object
  :slots ())

(setf (get uuv_gazebo_ros_plugins_msgs::GetListParam :md5sum-) "d3415e6ab074fdc7b0bd82ed3d6fccc7")
(setf (get uuv_gazebo_ros_plugins_msgs::GetListParam :datatype-) "uuv_gazebo_ros_plugins_msgs/GetListParam")
(setf (get uuv_gazebo_ros_plugins_msgs::GetListParam :request) uuv_gazebo_ros_plugins_msgs::GetListParamRequest)
(setf (get uuv_gazebo_ros_plugins_msgs::GetListParam :response) uuv_gazebo_ros_plugins_msgs::GetListParamResponse)

(defmethod uuv_gazebo_ros_plugins_msgs::GetListParamRequest
  (:response () (instance uuv_gazebo_ros_plugins_msgs::GetListParamResponse :init)))

(setf (get uuv_gazebo_ros_plugins_msgs::GetListParamRequest :md5sum-) "d3415e6ab074fdc7b0bd82ed3d6fccc7")
(setf (get uuv_gazebo_ros_plugins_msgs::GetListParamRequest :datatype-) "uuv_gazebo_ros_plugins_msgs/GetListParamRequest")
(setf (get uuv_gazebo_ros_plugins_msgs::GetListParamRequest :definition-)
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
string description
string[] tags
float64[] data

")

(setf (get uuv_gazebo_ros_plugins_msgs::GetListParamResponse :md5sum-) "d3415e6ab074fdc7b0bd82ed3d6fccc7")
(setf (get uuv_gazebo_ros_plugins_msgs::GetListParamResponse :datatype-) "uuv_gazebo_ros_plugins_msgs/GetListParamResponse")
(setf (get uuv_gazebo_ros_plugins_msgs::GetListParamResponse :definition-)
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
string description
string[] tags
float64[] data

")



(provide :uuv_gazebo_ros_plugins_msgs/GetListParam "d3415e6ab074fdc7b0bd82ed3d6fccc7")


