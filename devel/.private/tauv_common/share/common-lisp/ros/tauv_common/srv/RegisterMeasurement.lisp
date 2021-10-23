; Auto-generated. Do not edit!


(cl:in-package tauv_common-srv)


;//! \htmlinclude RegisterMeasurement-request.msg.html

(cl:defclass <RegisterMeasurement-request> (roslisp-msg-protocol:ros-message)
  ((pg_meas
    :reader pg_meas
    :initarg :pg_meas
    :type tauv_msgs-msg:PoseGraphMeasurement
    :initform (cl:make-instance 'tauv_msgs-msg:PoseGraphMeasurement)))
)

(cl:defclass RegisterMeasurement-request (<RegisterMeasurement-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <RegisterMeasurement-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'RegisterMeasurement-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_common-srv:<RegisterMeasurement-request> is deprecated: use tauv_common-srv:RegisterMeasurement-request instead.")))

(cl:ensure-generic-function 'pg_meas-val :lambda-list '(m))
(cl:defmethod pg_meas-val ((m <RegisterMeasurement-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_common-srv:pg_meas-val is deprecated.  Use tauv_common-srv:pg_meas instead.")
  (pg_meas m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <RegisterMeasurement-request>) ostream)
  "Serializes a message object of type '<RegisterMeasurement-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'pg_meas) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <RegisterMeasurement-request>) istream)
  "Deserializes a message object of type '<RegisterMeasurement-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'pg_meas) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<RegisterMeasurement-request>)))
  "Returns string type for a service object of type '<RegisterMeasurement-request>"
  "tauv_common/RegisterMeasurementRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'RegisterMeasurement-request)))
  "Returns string type for a service object of type 'RegisterMeasurement-request"
  "tauv_common/RegisterMeasurementRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<RegisterMeasurement-request>)))
  "Returns md5sum for a message object of type '<RegisterMeasurement-request>"
  "f2167c58c6c958d121ce25746a34db61")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'RegisterMeasurement-request)))
  "Returns md5sum for a message object of type 'RegisterMeasurement-request"
  "f2167c58c6c958d121ce25746a34db61")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<RegisterMeasurement-request>)))
  "Returns full string definition for message of type '<RegisterMeasurement-request>"
  (cl:format cl:nil "tauv_msgs/PoseGraphMeasurement pg_meas~%~%================================================================================~%MSG: tauv_msgs/PoseGraphMeasurement~%Header header~%uint32 landmark_id~%geometry_msgs/Point position~%~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'RegisterMeasurement-request)))
  "Returns full string definition for message of type 'RegisterMeasurement-request"
  (cl:format cl:nil "tauv_msgs/PoseGraphMeasurement pg_meas~%~%================================================================================~%MSG: tauv_msgs/PoseGraphMeasurement~%Header header~%uint32 landmark_id~%geometry_msgs/Point position~%~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <RegisterMeasurement-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'pg_meas))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <RegisterMeasurement-request>))
  "Converts a ROS message object to a list"
  (cl:list 'RegisterMeasurement-request
    (cl:cons ':pg_meas (pg_meas msg))
))
;//! \htmlinclude RegisterMeasurement-response.msg.html

(cl:defclass <RegisterMeasurement-response> (roslisp-msg-protocol:ros-message)
  ((success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass RegisterMeasurement-response (<RegisterMeasurement-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <RegisterMeasurement-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'RegisterMeasurement-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_common-srv:<RegisterMeasurement-response> is deprecated: use tauv_common-srv:RegisterMeasurement-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <RegisterMeasurement-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_common-srv:success-val is deprecated.  Use tauv_common-srv:success instead.")
  (success m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <RegisterMeasurement-response>) ostream)
  "Serializes a message object of type '<RegisterMeasurement-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <RegisterMeasurement-response>) istream)
  "Deserializes a message object of type '<RegisterMeasurement-response>"
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<RegisterMeasurement-response>)))
  "Returns string type for a service object of type '<RegisterMeasurement-response>"
  "tauv_common/RegisterMeasurementResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'RegisterMeasurement-response)))
  "Returns string type for a service object of type 'RegisterMeasurement-response"
  "tauv_common/RegisterMeasurementResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<RegisterMeasurement-response>)))
  "Returns md5sum for a message object of type '<RegisterMeasurement-response>"
  "f2167c58c6c958d121ce25746a34db61")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'RegisterMeasurement-response)))
  "Returns md5sum for a message object of type 'RegisterMeasurement-response"
  "f2167c58c6c958d121ce25746a34db61")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<RegisterMeasurement-response>)))
  "Returns full string definition for message of type '<RegisterMeasurement-response>"
  (cl:format cl:nil "bool success~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'RegisterMeasurement-response)))
  "Returns full string definition for message of type 'RegisterMeasurement-response"
  (cl:format cl:nil "bool success~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <RegisterMeasurement-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <RegisterMeasurement-response>))
  "Converts a ROS message object to a list"
  (cl:list 'RegisterMeasurement-response
    (cl:cons ':success (success msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'RegisterMeasurement)))
  'RegisterMeasurement-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'RegisterMeasurement)))
  'RegisterMeasurement-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'RegisterMeasurement)))
  "Returns string type for a service object of type '<RegisterMeasurement>"
  "tauv_common/RegisterMeasurement")