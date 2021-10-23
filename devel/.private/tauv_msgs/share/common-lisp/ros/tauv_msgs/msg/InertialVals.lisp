; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-msg)


;//! \htmlinclude InertialVals.msg.html

(cl:defclass <InertialVals> (roslisp-msg-protocol:ros-message)
  ((mass
    :reader mass
    :initarg :mass
    :type cl:float
    :initform 0.0)
   (buoyancy
    :reader buoyancy
    :initarg :buoyancy
    :type cl:float
    :initform 0.0)
   (ixx
    :reader ixx
    :initarg :ixx
    :type cl:float
    :initform 0.0)
   (iyy
    :reader iyy
    :initarg :iyy
    :type cl:float
    :initform 0.0)
   (izz
    :reader izz
    :initarg :izz
    :type cl:float
    :initform 0.0))
)

(cl:defclass InertialVals (<InertialVals>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <InertialVals>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'InertialVals)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-msg:<InertialVals> is deprecated: use tauv_msgs-msg:InertialVals instead.")))

(cl:ensure-generic-function 'mass-val :lambda-list '(m))
(cl:defmethod mass-val ((m <InertialVals>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:mass-val is deprecated.  Use tauv_msgs-msg:mass instead.")
  (mass m))

(cl:ensure-generic-function 'buoyancy-val :lambda-list '(m))
(cl:defmethod buoyancy-val ((m <InertialVals>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:buoyancy-val is deprecated.  Use tauv_msgs-msg:buoyancy instead.")
  (buoyancy m))

(cl:ensure-generic-function 'ixx-val :lambda-list '(m))
(cl:defmethod ixx-val ((m <InertialVals>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:ixx-val is deprecated.  Use tauv_msgs-msg:ixx instead.")
  (ixx m))

(cl:ensure-generic-function 'iyy-val :lambda-list '(m))
(cl:defmethod iyy-val ((m <InertialVals>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:iyy-val is deprecated.  Use tauv_msgs-msg:iyy instead.")
  (iyy m))

(cl:ensure-generic-function 'izz-val :lambda-list '(m))
(cl:defmethod izz-val ((m <InertialVals>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:izz-val is deprecated.  Use tauv_msgs-msg:izz instead.")
  (izz m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <InertialVals>) ostream)
  "Serializes a message object of type '<InertialVals>"
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'mass))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'buoyancy))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'ixx))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'iyy))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'izz))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <InertialVals>) istream)
  "Deserializes a message object of type '<InertialVals>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'mass) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'buoyancy) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'ixx) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'iyy) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'izz) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<InertialVals>)))
  "Returns string type for a message object of type '<InertialVals>"
  "tauv_msgs/InertialVals")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'InertialVals)))
  "Returns string type for a message object of type 'InertialVals"
  "tauv_msgs/InertialVals")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<InertialVals>)))
  "Returns md5sum for a message object of type '<InertialVals>"
  "dc905a6a26bfe30465ae55cdfc3db94e")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'InertialVals)))
  "Returns md5sum for a message object of type 'InertialVals"
  "dc905a6a26bfe30465ae55cdfc3db94e")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<InertialVals>)))
  "Returns full string definition for message of type '<InertialVals>"
  (cl:format cl:nil "float32 mass~%float32 buoyancy~%float32 ixx~%float32 iyy~%float32 izz~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'InertialVals)))
  "Returns full string definition for message of type 'InertialVals"
  (cl:format cl:nil "float32 mass~%float32 buoyancy~%float32 ixx~%float32 iyy~%float32 izz~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <InertialVals>))
  (cl:+ 0
     4
     4
     4
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <InertialVals>))
  "Converts a ROS message object to a list"
  (cl:list 'InertialVals
    (cl:cons ':mass (mass msg))
    (cl:cons ':buoyancy (buoyancy msg))
    (cl:cons ':ixx (ixx msg))
    (cl:cons ':iyy (iyy msg))
    (cl:cons ':izz (izz msg))
))
