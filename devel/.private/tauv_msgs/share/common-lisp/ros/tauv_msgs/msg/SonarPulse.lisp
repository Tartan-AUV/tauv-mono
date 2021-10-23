; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-msg)


;//! \htmlinclude SonarPulse.msg.html

(cl:defclass <SonarPulse> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (mode
    :reader mode
    :initarg :mode
    :type cl:fixnum
    :initform 0)
   (gain_setting
    :reader gain_setting
    :initarg :gain_setting
    :type cl:fixnum
    :initform 0)
   (angle
    :reader angle
    :initarg :angle
    :type cl:float
    :initform 0.0)
   (transmit_duration
    :reader transmit_duration
    :initarg :transmit_duration
    :type cl:float
    :initform 0.0)
   (sample_period
    :reader sample_period
    :initarg :sample_period
    :type cl:float
    :initform 0.0)
   (transmit_frequency
    :reader transmit_frequency
    :initarg :transmit_frequency
    :type cl:float
    :initform 0.0)
   (number_of_samples
    :reader number_of_samples
    :initarg :number_of_samples
    :type cl:fixnum
    :initform 0)
   (data_length
    :reader data_length
    :initarg :data_length
    :type cl:fixnum
    :initform 0)
   (data
    :reader data
    :initarg :data
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 0 :element-type 'cl:fixnum :initial-element 0)))
)

(cl:defclass SonarPulse (<SonarPulse>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SonarPulse>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SonarPulse)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-msg:<SonarPulse> is deprecated: use tauv_msgs-msg:SonarPulse instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <SonarPulse>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:header-val is deprecated.  Use tauv_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'mode-val :lambda-list '(m))
(cl:defmethod mode-val ((m <SonarPulse>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:mode-val is deprecated.  Use tauv_msgs-msg:mode instead.")
  (mode m))

(cl:ensure-generic-function 'gain_setting-val :lambda-list '(m))
(cl:defmethod gain_setting-val ((m <SonarPulse>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:gain_setting-val is deprecated.  Use tauv_msgs-msg:gain_setting instead.")
  (gain_setting m))

(cl:ensure-generic-function 'angle-val :lambda-list '(m))
(cl:defmethod angle-val ((m <SonarPulse>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:angle-val is deprecated.  Use tauv_msgs-msg:angle instead.")
  (angle m))

(cl:ensure-generic-function 'transmit_duration-val :lambda-list '(m))
(cl:defmethod transmit_duration-val ((m <SonarPulse>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:transmit_duration-val is deprecated.  Use tauv_msgs-msg:transmit_duration instead.")
  (transmit_duration m))

(cl:ensure-generic-function 'sample_period-val :lambda-list '(m))
(cl:defmethod sample_period-val ((m <SonarPulse>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:sample_period-val is deprecated.  Use tauv_msgs-msg:sample_period instead.")
  (sample_period m))

(cl:ensure-generic-function 'transmit_frequency-val :lambda-list '(m))
(cl:defmethod transmit_frequency-val ((m <SonarPulse>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:transmit_frequency-val is deprecated.  Use tauv_msgs-msg:transmit_frequency instead.")
  (transmit_frequency m))

(cl:ensure-generic-function 'number_of_samples-val :lambda-list '(m))
(cl:defmethod number_of_samples-val ((m <SonarPulse>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:number_of_samples-val is deprecated.  Use tauv_msgs-msg:number_of_samples instead.")
  (number_of_samples m))

(cl:ensure-generic-function 'data_length-val :lambda-list '(m))
(cl:defmethod data_length-val ((m <SonarPulse>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:data_length-val is deprecated.  Use tauv_msgs-msg:data_length instead.")
  (data_length m))

(cl:ensure-generic-function 'data-val :lambda-list '(m))
(cl:defmethod data-val ((m <SonarPulse>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:data-val is deprecated.  Use tauv_msgs-msg:data instead.")
  (data m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SonarPulse>) ostream)
  "Serializes a message object of type '<SonarPulse>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'mode)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'gain_setting)) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'angle))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'transmit_duration))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'sample_period))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'transmit_frequency))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'number_of_samples)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'number_of_samples)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'data_length)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'data_length)) ostream)
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'data))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream))
   (cl:slot-value msg 'data))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SonarPulse>) istream)
  "Deserializes a message object of type '<SonarPulse>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'mode)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'gain_setting)) (cl:read-byte istream))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'angle) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'transmit_duration) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'sample_period) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'transmit_frequency) (roslisp-utils:decode-single-float-bits bits)))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'number_of_samples)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'number_of_samples)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'data_length)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'data_length)) (cl:read-byte istream))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'data) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'data)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SonarPulse>)))
  "Returns string type for a message object of type '<SonarPulse>"
  "tauv_msgs/SonarPulse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SonarPulse)))
  "Returns string type for a message object of type 'SonarPulse"
  "tauv_msgs/SonarPulse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SonarPulse>)))
  "Returns md5sum for a message object of type '<SonarPulse>"
  "5a4534993e7634b0f0c3a4e8eec771be")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SonarPulse)))
  "Returns md5sum for a message object of type 'SonarPulse"
  "5a4534993e7634b0f0c3a4e8eec771be")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SonarPulse>)))
  "Returns full string definition for message of type '<SonarPulse>"
  (cl:format cl:nil "Header header~%~%uint8 mode~%uint8 gain_setting~%float32 angle~%float32 transmit_duration~%float32 sample_period~%float32 transmit_frequency~%uint16 number_of_samples~%~%uint16 data_length~%uint8[] data~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SonarPulse)))
  "Returns full string definition for message of type 'SonarPulse"
  (cl:format cl:nil "Header header~%~%uint8 mode~%uint8 gain_setting~%float32 angle~%float32 transmit_duration~%float32 sample_period~%float32 transmit_frequency~%uint16 number_of_samples~%~%uint16 data_length~%uint8[] data~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SonarPulse>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     1
     1
     4
     4
     4
     4
     2
     2
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'data) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 1)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SonarPulse>))
  "Converts a ROS message object to a list"
  (cl:list 'SonarPulse
    (cl:cons ':header (header msg))
    (cl:cons ':mode (mode msg))
    (cl:cons ':gain_setting (gain_setting msg))
    (cl:cons ':angle (angle msg))
    (cl:cons ':transmit_duration (transmit_duration msg))
    (cl:cons ':sample_period (sample_period msg))
    (cl:cons ':transmit_frequency (transmit_frequency msg))
    (cl:cons ':number_of_samples (number_of_samples msg))
    (cl:cons ':data_length (data_length msg))
    (cl:cons ':data (data msg))
))
