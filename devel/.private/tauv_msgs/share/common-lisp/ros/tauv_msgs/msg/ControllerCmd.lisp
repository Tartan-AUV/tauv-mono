; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-msg)


;//! \htmlinclude ControllerCmd.msg.html

(cl:defclass <ControllerCmd> (roslisp-msg-protocol:ros-message)
  ((a_x
    :reader a_x
    :initarg :a_x
    :type cl:float
    :initform 0.0)
   (a_y
    :reader a_y
    :initarg :a_y
    :type cl:float
    :initform 0.0)
   (a_z
    :reader a_z
    :initarg :a_z
    :type cl:float
    :initform 0.0)
   (a_yaw
    :reader a_yaw
    :initarg :a_yaw
    :type cl:float
    :initform 0.0)
   (p_roll
    :reader p_roll
    :initarg :p_roll
    :type cl:float
    :initform 0.0)
   (p_pitch
    :reader p_pitch
    :initarg :p_pitch
    :type cl:float
    :initform 0.0))
)

(cl:defclass ControllerCmd (<ControllerCmd>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ControllerCmd>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ControllerCmd)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-msg:<ControllerCmd> is deprecated: use tauv_msgs-msg:ControllerCmd instead.")))

(cl:ensure-generic-function 'a_x-val :lambda-list '(m))
(cl:defmethod a_x-val ((m <ControllerCmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:a_x-val is deprecated.  Use tauv_msgs-msg:a_x instead.")
  (a_x m))

(cl:ensure-generic-function 'a_y-val :lambda-list '(m))
(cl:defmethod a_y-val ((m <ControllerCmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:a_y-val is deprecated.  Use tauv_msgs-msg:a_y instead.")
  (a_y m))

(cl:ensure-generic-function 'a_z-val :lambda-list '(m))
(cl:defmethod a_z-val ((m <ControllerCmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:a_z-val is deprecated.  Use tauv_msgs-msg:a_z instead.")
  (a_z m))

(cl:ensure-generic-function 'a_yaw-val :lambda-list '(m))
(cl:defmethod a_yaw-val ((m <ControllerCmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:a_yaw-val is deprecated.  Use tauv_msgs-msg:a_yaw instead.")
  (a_yaw m))

(cl:ensure-generic-function 'p_roll-val :lambda-list '(m))
(cl:defmethod p_roll-val ((m <ControllerCmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:p_roll-val is deprecated.  Use tauv_msgs-msg:p_roll instead.")
  (p_roll m))

(cl:ensure-generic-function 'p_pitch-val :lambda-list '(m))
(cl:defmethod p_pitch-val ((m <ControllerCmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:p_pitch-val is deprecated.  Use tauv_msgs-msg:p_pitch instead.")
  (p_pitch m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ControllerCmd>) ostream)
  "Serializes a message object of type '<ControllerCmd>"
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'a_x))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'a_y))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'a_z))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'a_yaw))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'p_roll))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'p_pitch))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ControllerCmd>) istream)
  "Deserializes a message object of type '<ControllerCmd>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'a_x) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'a_y) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'a_z) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'a_yaw) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'p_roll) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'p_pitch) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ControllerCmd>)))
  "Returns string type for a message object of type '<ControllerCmd>"
  "tauv_msgs/ControllerCmd")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ControllerCmd)))
  "Returns string type for a message object of type 'ControllerCmd"
  "tauv_msgs/ControllerCmd")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ControllerCmd>)))
  "Returns md5sum for a message object of type '<ControllerCmd>"
  "c0612f34c73db057150c241e2726f1e9")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ControllerCmd)))
  "Returns md5sum for a message object of type 'ControllerCmd"
  "c0612f34c73db057150c241e2726f1e9")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ControllerCmd>)))
  "Returns full string definition for message of type '<ControllerCmd>"
  (cl:format cl:nil "float32 a_x  # Acceleration in fixed frame x direction~%float32 a_y  # Acceleration in fixed frame y direction~%float32 a_z  # Acceleration in fixed frame z direction~%float32 a_yaw  # Acceleration in fixed frame yaw direction~%float32 p_roll  # roll target~%float32 p_pitch  # pitch target~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ControllerCmd)))
  "Returns full string definition for message of type 'ControllerCmd"
  (cl:format cl:nil "float32 a_x  # Acceleration in fixed frame x direction~%float32 a_y  # Acceleration in fixed frame y direction~%float32 a_z  # Acceleration in fixed frame z direction~%float32 a_yaw  # Acceleration in fixed frame yaw direction~%float32 p_roll  # roll target~%float32 p_pitch  # pitch target~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ControllerCmd>))
  (cl:+ 0
     4
     4
     4
     4
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ControllerCmd>))
  "Converts a ROS message object to a list"
  (cl:list 'ControllerCmd
    (cl:cons ':a_x (a_x msg))
    (cl:cons ':a_y (a_y msg))
    (cl:cons ':a_z (a_z msg))
    (cl:cons ':a_yaw (a_yaw msg))
    (cl:cons ':p_roll (p_roll msg))
    (cl:cons ':p_pitch (p_pitch msg))
))
