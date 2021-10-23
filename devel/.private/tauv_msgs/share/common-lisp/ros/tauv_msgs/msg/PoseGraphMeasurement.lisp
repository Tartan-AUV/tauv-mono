; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-msg)


;//! \htmlinclude PoseGraphMeasurement.msg.html

(cl:defclass <PoseGraphMeasurement> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (landmark_id
    :reader landmark_id
    :initarg :landmark_id
    :type cl:integer
    :initform 0)
   (position
    :reader position
    :initarg :position
    :type geometry_msgs-msg:Point
    :initform (cl:make-instance 'geometry_msgs-msg:Point)))
)

(cl:defclass PoseGraphMeasurement (<PoseGraphMeasurement>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <PoseGraphMeasurement>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'PoseGraphMeasurement)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-msg:<PoseGraphMeasurement> is deprecated: use tauv_msgs-msg:PoseGraphMeasurement instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <PoseGraphMeasurement>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:header-val is deprecated.  Use tauv_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'landmark_id-val :lambda-list '(m))
(cl:defmethod landmark_id-val ((m <PoseGraphMeasurement>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:landmark_id-val is deprecated.  Use tauv_msgs-msg:landmark_id instead.")
  (landmark_id m))

(cl:ensure-generic-function 'position-val :lambda-list '(m))
(cl:defmethod position-val ((m <PoseGraphMeasurement>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:position-val is deprecated.  Use tauv_msgs-msg:position instead.")
  (position m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <PoseGraphMeasurement>) ostream)
  "Serializes a message object of type '<PoseGraphMeasurement>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'landmark_id)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'landmark_id)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'landmark_id)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'landmark_id)) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'position) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <PoseGraphMeasurement>) istream)
  "Deserializes a message object of type '<PoseGraphMeasurement>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'landmark_id)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'landmark_id)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'landmark_id)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'landmark_id)) (cl:read-byte istream))
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'position) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<PoseGraphMeasurement>)))
  "Returns string type for a message object of type '<PoseGraphMeasurement>"
  "tauv_msgs/PoseGraphMeasurement")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'PoseGraphMeasurement)))
  "Returns string type for a message object of type 'PoseGraphMeasurement"
  "tauv_msgs/PoseGraphMeasurement")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<PoseGraphMeasurement>)))
  "Returns md5sum for a message object of type '<PoseGraphMeasurement>"
  "7e27f94f75bfdd40c9e3cb1c46d27f36")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'PoseGraphMeasurement)))
  "Returns md5sum for a message object of type 'PoseGraphMeasurement"
  "7e27f94f75bfdd40c9e3cb1c46d27f36")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<PoseGraphMeasurement>)))
  "Returns full string definition for message of type '<PoseGraphMeasurement>"
  (cl:format cl:nil "Header header~%uint32 landmark_id~%geometry_msgs/Point position~%~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'PoseGraphMeasurement)))
  "Returns full string definition for message of type 'PoseGraphMeasurement"
  (cl:format cl:nil "Header header~%uint32 landmark_id~%geometry_msgs/Point position~%~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <PoseGraphMeasurement>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     4
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'position))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <PoseGraphMeasurement>))
  "Converts a ROS message object to a list"
  (cl:list 'PoseGraphMeasurement
    (cl:cons ':header (header msg))
    (cl:cons ':landmark_id (landmark_id msg))
    (cl:cons ':position (position msg))
))
