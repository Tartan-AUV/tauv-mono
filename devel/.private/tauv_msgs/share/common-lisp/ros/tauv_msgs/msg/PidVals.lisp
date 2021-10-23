; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-msg)


;//! \htmlinclude PidVals.msg.html

(cl:defclass <PidVals> (roslisp-msg-protocol:ros-message)
  ((a_p
    :reader a_p
    :initarg :a_p
    :type cl:float
    :initform 0.0)
   (a_i
    :reader a_i
    :initarg :a_i
    :type cl:float
    :initform 0.0)
   (a_d
    :reader a_d
    :initarg :a_d
    :type cl:float
    :initform 0.0)
   (a_sat
    :reader a_sat
    :initarg :a_sat
    :type cl:float
    :initform 0.0)
   (l_p
    :reader l_p
    :initarg :l_p
    :type cl:float
    :initform 0.0)
   (l_i
    :reader l_i
    :initarg :l_i
    :type cl:float
    :initform 0.0)
   (l_d
    :reader l_d
    :initarg :l_d
    :type cl:float
    :initform 0.0)
   (l_sat
    :reader l_sat
    :initarg :l_sat
    :type cl:float
    :initform 0.0))
)

(cl:defclass PidVals (<PidVals>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <PidVals>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'PidVals)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-msg:<PidVals> is deprecated: use tauv_msgs-msg:PidVals instead.")))

(cl:ensure-generic-function 'a_p-val :lambda-list '(m))
(cl:defmethod a_p-val ((m <PidVals>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:a_p-val is deprecated.  Use tauv_msgs-msg:a_p instead.")
  (a_p m))

(cl:ensure-generic-function 'a_i-val :lambda-list '(m))
(cl:defmethod a_i-val ((m <PidVals>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:a_i-val is deprecated.  Use tauv_msgs-msg:a_i instead.")
  (a_i m))

(cl:ensure-generic-function 'a_d-val :lambda-list '(m))
(cl:defmethod a_d-val ((m <PidVals>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:a_d-val is deprecated.  Use tauv_msgs-msg:a_d instead.")
  (a_d m))

(cl:ensure-generic-function 'a_sat-val :lambda-list '(m))
(cl:defmethod a_sat-val ((m <PidVals>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:a_sat-val is deprecated.  Use tauv_msgs-msg:a_sat instead.")
  (a_sat m))

(cl:ensure-generic-function 'l_p-val :lambda-list '(m))
(cl:defmethod l_p-val ((m <PidVals>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:l_p-val is deprecated.  Use tauv_msgs-msg:l_p instead.")
  (l_p m))

(cl:ensure-generic-function 'l_i-val :lambda-list '(m))
(cl:defmethod l_i-val ((m <PidVals>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:l_i-val is deprecated.  Use tauv_msgs-msg:l_i instead.")
  (l_i m))

(cl:ensure-generic-function 'l_d-val :lambda-list '(m))
(cl:defmethod l_d-val ((m <PidVals>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:l_d-val is deprecated.  Use tauv_msgs-msg:l_d instead.")
  (l_d m))

(cl:ensure-generic-function 'l_sat-val :lambda-list '(m))
(cl:defmethod l_sat-val ((m <PidVals>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:l_sat-val is deprecated.  Use tauv_msgs-msg:l_sat instead.")
  (l_sat m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <PidVals>) ostream)
  "Serializes a message object of type '<PidVals>"
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'a_p))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'a_i))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'a_d))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'a_sat))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'l_p))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'l_i))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'l_d))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'l_sat))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <PidVals>) istream)
  "Deserializes a message object of type '<PidVals>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'a_p) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'a_i) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'a_d) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'a_sat) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'l_p) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'l_i) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'l_d) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'l_sat) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<PidVals>)))
  "Returns string type for a message object of type '<PidVals>"
  "tauv_msgs/PidVals")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'PidVals)))
  "Returns string type for a message object of type 'PidVals"
  "tauv_msgs/PidVals")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<PidVals>)))
  "Returns md5sum for a message object of type '<PidVals>"
  "d4db47770a0caf47edbb925bd3a9269a")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'PidVals)))
  "Returns md5sum for a message object of type 'PidVals"
  "d4db47770a0caf47edbb925bd3a9269a")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<PidVals>)))
  "Returns full string definition for message of type '<PidVals>"
  (cl:format cl:nil "float32 a_p~%float32 a_i~%float32 a_d~%float32 a_sat~%float32 l_p~%float32 l_i~%float32 l_d~%float32 l_sat~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'PidVals)))
  "Returns full string definition for message of type 'PidVals"
  (cl:format cl:nil "float32 a_p~%float32 a_i~%float32 a_d~%float32 a_sat~%float32 l_p~%float32 l_i~%float32 l_d~%float32 l_sat~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <PidVals>))
  (cl:+ 0
     4
     4
     4
     4
     4
     4
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <PidVals>))
  "Converts a ROS message object to a list"
  (cl:list 'PidVals
    (cl:cons ':a_p (a_p msg))
    (cl:cons ':a_i (a_i msg))
    (cl:cons ':a_d (a_d msg))
    (cl:cons ':a_sat (a_sat msg))
    (cl:cons ':l_p (l_p msg))
    (cl:cons ':l_i (l_i msg))
    (cl:cons ':l_d (l_d msg))
    (cl:cons ':l_sat (l_sat msg))
))
