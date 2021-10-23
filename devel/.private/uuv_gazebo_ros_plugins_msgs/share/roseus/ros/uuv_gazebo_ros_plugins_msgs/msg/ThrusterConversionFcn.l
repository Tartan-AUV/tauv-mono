;; Auto-generated. Do not edit!


(when (boundp 'uuv_gazebo_ros_plugins_msgs::ThrusterConversionFcn)
  (if (not (find-package "UUV_GAZEBO_ROS_PLUGINS_MSGS"))
    (make-package "UUV_GAZEBO_ROS_PLUGINS_MSGS"))
  (shadow 'ThrusterConversionFcn (find-package "UUV_GAZEBO_ROS_PLUGINS_MSGS")))
(unless (find-package "UUV_GAZEBO_ROS_PLUGINS_MSGS::THRUSTERCONVERSIONFCN")
  (make-package "UUV_GAZEBO_ROS_PLUGINS_MSGS::THRUSTERCONVERSIONFCN"))

(in-package "ROS")
;;//! \htmlinclude ThrusterConversionFcn.msg.html


(defclass uuv_gazebo_ros_plugins_msgs::ThrusterConversionFcn
  :super ros::object
  :slots (_function_name _tags _data _lookup_table_input _lookup_table_output ))

(defmethod uuv_gazebo_ros_plugins_msgs::ThrusterConversionFcn
  (:init
   (&key
    ((:function_name __function_name) "")
    ((:tags __tags) (let (r) (dotimes (i 0) (push "" r)) r))
    ((:data __data) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:lookup_table_input __lookup_table_input) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:lookup_table_output __lookup_table_output) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _function_name (string __function_name))
   (setq _tags __tags)
   (setq _data __data)
   (setq _lookup_table_input __lookup_table_input)
   (setq _lookup_table_output __lookup_table_output)
   self)
  (:function_name
   (&optional __function_name)
   (if __function_name (setq _function_name __function_name)) _function_name)
  (:tags
   (&optional __tags)
   (if __tags (setq _tags __tags)) _tags)
  (:data
   (&optional __data)
   (if __data (setq _data __data)) _data)
  (:lookup_table_input
   (&optional __lookup_table_input)
   (if __lookup_table_input (setq _lookup_table_input __lookup_table_input)) _lookup_table_input)
  (:lookup_table_output
   (&optional __lookup_table_output)
   (if __lookup_table_output (setq _lookup_table_output __lookup_table_output)) _lookup_table_output)
  (:serialization-length
   ()
   (+
    ;; string _function_name
    4 (length _function_name)
    ;; string[] _tags
    (apply #'+ (mapcar #'(lambda (x) (+ 4 (length x))) _tags)) 4
    ;; float64[] _data
    (* 8    (length _data)) 4
    ;; float64[] _lookup_table_input
    (* 8    (length _lookup_table_input)) 4
    ;; float64[] _lookup_table_output
    (* 8    (length _lookup_table_output)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _function_name
       (write-long (length _function_name) s) (princ _function_name s)
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
     ;; float64[] _lookup_table_input
     (write-long (length _lookup_table_input) s)
     (dotimes (i (length _lookup_table_input))
       (sys::poke (elt _lookup_table_input i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64[] _lookup_table_output
     (write-long (length _lookup_table_output) s)
     (dotimes (i (length _lookup_table_output))
       (sys::poke (elt _lookup_table_output i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _function_name
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _function_name (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
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
   ;; float64[] _lookup_table_input
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _lookup_table_input (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _lookup_table_input i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;; float64[] _lookup_table_output
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _lookup_table_output (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _lookup_table_output i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;;
   self)
  )

(setf (get uuv_gazebo_ros_plugins_msgs::ThrusterConversionFcn :md5sum-) "5cc7c4f30276fbc995f2325f64846776")
(setf (get uuv_gazebo_ros_plugins_msgs::ThrusterConversionFcn :datatype-) "uuv_gazebo_ros_plugins_msgs/ThrusterConversionFcn")
(setf (get uuv_gazebo_ros_plugins_msgs::ThrusterConversionFcn :definition-)
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

string function_name
string[] tags
float64[] data
float64[] lookup_table_input
float64[] lookup_table_output

")



(provide :uuv_gazebo_ros_plugins_msgs/ThrusterConversionFcn "5cc7c4f30276fbc995f2325f64846776")


