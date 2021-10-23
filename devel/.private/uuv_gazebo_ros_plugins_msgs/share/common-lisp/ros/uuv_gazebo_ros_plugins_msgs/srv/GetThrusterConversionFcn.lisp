; Auto-generated. Do not edit!


(cl:in-package uuv_gazebo_ros_plugins_msgs-srv)


;//! \htmlinclude GetThrusterConversionFcn-request.msg.html

(cl:defclass <GetThrusterConversionFcn-request> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass GetThrusterConversionFcn-request (<GetThrusterConversionFcn-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetThrusterConversionFcn-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetThrusterConversionFcn-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name uuv_gazebo_ros_plugins_msgs-srv:<GetThrusterConversionFcn-request> is deprecated: use uuv_gazebo_ros_plugins_msgs-srv:GetThrusterConversionFcn-request instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetThrusterConversionFcn-request>) ostream)
  "Serializes a message object of type '<GetThrusterConversionFcn-request>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetThrusterConversionFcn-request>) istream)
  "Deserializes a message object of type '<GetThrusterConversionFcn-request>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetThrusterConversionFcn-request>)))
  "Returns string type for a service object of type '<GetThrusterConversionFcn-request>"
  "uuv_gazebo_ros_plugins_msgs/GetThrusterConversionFcnRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetThrusterConversionFcn-request)))
  "Returns string type for a service object of type 'GetThrusterConversionFcn-request"
  "uuv_gazebo_ros_plugins_msgs/GetThrusterConversionFcnRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetThrusterConversionFcn-request>)))
  "Returns md5sum for a message object of type '<GetThrusterConversionFcn-request>"
  "b489744fdf1ea3660acd86f33ee041a7")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetThrusterConversionFcn-request)))
  "Returns md5sum for a message object of type 'GetThrusterConversionFcn-request"
  "b489744fdf1ea3660acd86f33ee041a7")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetThrusterConversionFcn-request>)))
  "Returns full string definition for message of type '<GetThrusterConversionFcn-request>"
  (cl:format cl:nil "# Copyright (c) 2016 The UUV Simulator Authors.~%# All rights reserved.~%#~%# Licensed under the Apache License, Version 2.0 (the \"License\");~%# you may not use this file except in compliance with the License.~%# You may obtain a copy of the License at~%#~%#     http://www.apache.org/licenses/LICENSE-2.0~%#~%# Unless required by applicable law or agreed to in writing, software~%# distributed under the License is distributed on an \"AS IS\" BASIS,~%# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.~%# See the License for the specific language governing permissions and~%# limitations under the License.~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetThrusterConversionFcn-request)))
  "Returns full string definition for message of type 'GetThrusterConversionFcn-request"
  (cl:format cl:nil "# Copyright (c) 2016 The UUV Simulator Authors.~%# All rights reserved.~%#~%# Licensed under the Apache License, Version 2.0 (the \"License\");~%# you may not use this file except in compliance with the License.~%# You may obtain a copy of the License at~%#~%#     http://www.apache.org/licenses/LICENSE-2.0~%#~%# Unless required by applicable law or agreed to in writing, software~%# distributed under the License is distributed on an \"AS IS\" BASIS,~%# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.~%# See the License for the specific language governing permissions and~%# limitations under the License.~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetThrusterConversionFcn-request>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetThrusterConversionFcn-request>))
  "Converts a ROS message object to a list"
  (cl:list 'GetThrusterConversionFcn-request
))
;//! \htmlinclude GetThrusterConversionFcn-response.msg.html

(cl:defclass <GetThrusterConversionFcn-response> (roslisp-msg-protocol:ros-message)
  ((fcn
    :reader fcn
    :initarg :fcn
    :type uuv_gazebo_ros_plugins_msgs-msg:ThrusterConversionFcn
    :initform (cl:make-instance 'uuv_gazebo_ros_plugins_msgs-msg:ThrusterConversionFcn)))
)

(cl:defclass GetThrusterConversionFcn-response (<GetThrusterConversionFcn-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetThrusterConversionFcn-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetThrusterConversionFcn-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name uuv_gazebo_ros_plugins_msgs-srv:<GetThrusterConversionFcn-response> is deprecated: use uuv_gazebo_ros_plugins_msgs-srv:GetThrusterConversionFcn-response instead.")))

(cl:ensure-generic-function 'fcn-val :lambda-list '(m))
(cl:defmethod fcn-val ((m <GetThrusterConversionFcn-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader uuv_gazebo_ros_plugins_msgs-srv:fcn-val is deprecated.  Use uuv_gazebo_ros_plugins_msgs-srv:fcn instead.")
  (fcn m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetThrusterConversionFcn-response>) ostream)
  "Serializes a message object of type '<GetThrusterConversionFcn-response>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'fcn) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetThrusterConversionFcn-response>) istream)
  "Deserializes a message object of type '<GetThrusterConversionFcn-response>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'fcn) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetThrusterConversionFcn-response>)))
  "Returns string type for a service object of type '<GetThrusterConversionFcn-response>"
  "uuv_gazebo_ros_plugins_msgs/GetThrusterConversionFcnResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetThrusterConversionFcn-response)))
  "Returns string type for a service object of type 'GetThrusterConversionFcn-response"
  "uuv_gazebo_ros_plugins_msgs/GetThrusterConversionFcnResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetThrusterConversionFcn-response>)))
  "Returns md5sum for a message object of type '<GetThrusterConversionFcn-response>"
  "b489744fdf1ea3660acd86f33ee041a7")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetThrusterConversionFcn-response)))
  "Returns md5sum for a message object of type 'GetThrusterConversionFcn-response"
  "b489744fdf1ea3660acd86f33ee041a7")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetThrusterConversionFcn-response>)))
  "Returns full string definition for message of type '<GetThrusterConversionFcn-response>"
  (cl:format cl:nil "uuv_gazebo_ros_plugins_msgs/ThrusterConversionFcn fcn~%~%~%================================================================================~%MSG: uuv_gazebo_ros_plugins_msgs/ThrusterConversionFcn~%# Copyright (c) 2016 The UUV Simulator Authors.~%# All rights reserved.~%#~%# Licensed under the Apache License, Version 2.0 (the \"License\");~%# you may not use this file except in compliance with the License.~%# You may obtain a copy of the License at~%#~%#     http://www.apache.org/licenses/LICENSE-2.0~%#~%# Unless required by applicable law or agreed to in writing, software~%# distributed under the License is distributed on an \"AS IS\" BASIS,~%# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.~%# See the License for the specific language governing permissions and~%# limitations under the License.~%~%string function_name~%string[] tags~%float64[] data~%float64[] lookup_table_input~%float64[] lookup_table_output~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetThrusterConversionFcn-response)))
  "Returns full string definition for message of type 'GetThrusterConversionFcn-response"
  (cl:format cl:nil "uuv_gazebo_ros_plugins_msgs/ThrusterConversionFcn fcn~%~%~%================================================================================~%MSG: uuv_gazebo_ros_plugins_msgs/ThrusterConversionFcn~%# Copyright (c) 2016 The UUV Simulator Authors.~%# All rights reserved.~%#~%# Licensed under the Apache License, Version 2.0 (the \"License\");~%# you may not use this file except in compliance with the License.~%# You may obtain a copy of the License at~%#~%#     http://www.apache.org/licenses/LICENSE-2.0~%#~%# Unless required by applicable law or agreed to in writing, software~%# distributed under the License is distributed on an \"AS IS\" BASIS,~%# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.~%# See the License for the specific language governing permissions and~%# limitations under the License.~%~%string function_name~%string[] tags~%float64[] data~%float64[] lookup_table_input~%float64[] lookup_table_output~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetThrusterConversionFcn-response>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'fcn))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetThrusterConversionFcn-response>))
  "Converts a ROS message object to a list"
  (cl:list 'GetThrusterConversionFcn-response
    (cl:cons ':fcn (fcn msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'GetThrusterConversionFcn)))
  'GetThrusterConversionFcn-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'GetThrusterConversionFcn)))
  'GetThrusterConversionFcn-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetThrusterConversionFcn)))
  "Returns string type for a service object of type '<GetThrusterConversionFcn>"
  "uuv_gazebo_ros_plugins_msgs/GetThrusterConversionFcn")