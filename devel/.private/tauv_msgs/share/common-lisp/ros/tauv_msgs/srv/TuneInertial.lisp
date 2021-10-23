; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-srv)


;//! \htmlinclude TuneInertial-request.msg.html

(cl:defclass <TuneInertial-request> (roslisp-msg-protocol:ros-message)
  ((vals
    :reader vals
    :initarg :vals
    :type tauv_msgs-msg:InertialVals
    :initform (cl:make-instance 'tauv_msgs-msg:InertialVals)))
)

(cl:defclass TuneInertial-request (<TuneInertial-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <TuneInertial-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'TuneInertial-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-srv:<TuneInertial-request> is deprecated: use tauv_msgs-srv:TuneInertial-request instead.")))

(cl:ensure-generic-function 'vals-val :lambda-list '(m))
(cl:defmethod vals-val ((m <TuneInertial-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-srv:vals-val is deprecated.  Use tauv_msgs-srv:vals instead.")
  (vals m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <TuneInertial-request>) ostream)
  "Serializes a message object of type '<TuneInertial-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'vals) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <TuneInertial-request>) istream)
  "Deserializes a message object of type '<TuneInertial-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'vals) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<TuneInertial-request>)))
  "Returns string type for a service object of type '<TuneInertial-request>"
  "tauv_msgs/TuneInertialRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'TuneInertial-request)))
  "Returns string type for a service object of type 'TuneInertial-request"
  "tauv_msgs/TuneInertialRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<TuneInertial-request>)))
  "Returns md5sum for a message object of type '<TuneInertial-request>"
  "b07a11376c5b2ae9ec7888a1a74564d4")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'TuneInertial-request)))
  "Returns md5sum for a message object of type 'TuneInertial-request"
  "b07a11376c5b2ae9ec7888a1a74564d4")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<TuneInertial-request>)))
  "Returns full string definition for message of type '<TuneInertial-request>"
  (cl:format cl:nil "tauv_msgs/InertialVals vals~%~%================================================================================~%MSG: tauv_msgs/InertialVals~%float32 mass~%float32 buoyancy~%float32 ixx~%float32 iyy~%float32 izz~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'TuneInertial-request)))
  "Returns full string definition for message of type 'TuneInertial-request"
  (cl:format cl:nil "tauv_msgs/InertialVals vals~%~%================================================================================~%MSG: tauv_msgs/InertialVals~%float32 mass~%float32 buoyancy~%float32 ixx~%float32 iyy~%float32 izz~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <TuneInertial-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'vals))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <TuneInertial-request>))
  "Converts a ROS message object to a list"
  (cl:list 'TuneInertial-request
    (cl:cons ':vals (vals msg))
))
;//! \htmlinclude TuneInertial-response.msg.html

(cl:defclass <TuneInertial-response> (roslisp-msg-protocol:ros-message)
  ((success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass TuneInertial-response (<TuneInertial-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <TuneInertial-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'TuneInertial-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-srv:<TuneInertial-response> is deprecated: use tauv_msgs-srv:TuneInertial-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <TuneInertial-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-srv:success-val is deprecated.  Use tauv_msgs-srv:success instead.")
  (success m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <TuneInertial-response>) ostream)
  "Serializes a message object of type '<TuneInertial-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <TuneInertial-response>) istream)
  "Deserializes a message object of type '<TuneInertial-response>"
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<TuneInertial-response>)))
  "Returns string type for a service object of type '<TuneInertial-response>"
  "tauv_msgs/TuneInertialResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'TuneInertial-response)))
  "Returns string type for a service object of type 'TuneInertial-response"
  "tauv_msgs/TuneInertialResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<TuneInertial-response>)))
  "Returns md5sum for a message object of type '<TuneInertial-response>"
  "b07a11376c5b2ae9ec7888a1a74564d4")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'TuneInertial-response)))
  "Returns md5sum for a message object of type 'TuneInertial-response"
  "b07a11376c5b2ae9ec7888a1a74564d4")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<TuneInertial-response>)))
  "Returns full string definition for message of type '<TuneInertial-response>"
  (cl:format cl:nil "bool success~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'TuneInertial-response)))
  "Returns full string definition for message of type 'TuneInertial-response"
  (cl:format cl:nil "bool success~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <TuneInertial-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <TuneInertial-response>))
  "Converts a ROS message object to a list"
  (cl:list 'TuneInertial-response
    (cl:cons ':success (success msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'TuneInertial)))
  'TuneInertial-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'TuneInertial)))
  'TuneInertial-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'TuneInertial)))
  "Returns string type for a service object of type '<TuneInertial>"
  "tauv_msgs/TuneInertial")