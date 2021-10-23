; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-srv)


;//! \htmlinclude TunePid-request.msg.html

(cl:defclass <TunePid-request> (roslisp-msg-protocol:ros-message)
  ((vals
    :reader vals
    :initarg :vals
    :type tauv_msgs-msg:PidVals
    :initform (cl:make-instance 'tauv_msgs-msg:PidVals)))
)

(cl:defclass TunePid-request (<TunePid-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <TunePid-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'TunePid-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-srv:<TunePid-request> is deprecated: use tauv_msgs-srv:TunePid-request instead.")))

(cl:ensure-generic-function 'vals-val :lambda-list '(m))
(cl:defmethod vals-val ((m <TunePid-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-srv:vals-val is deprecated.  Use tauv_msgs-srv:vals instead.")
  (vals m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <TunePid-request>) ostream)
  "Serializes a message object of type '<TunePid-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'vals) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <TunePid-request>) istream)
  "Deserializes a message object of type '<TunePid-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'vals) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<TunePid-request>)))
  "Returns string type for a service object of type '<TunePid-request>"
  "tauv_msgs/TunePidRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'TunePid-request)))
  "Returns string type for a service object of type 'TunePid-request"
  "tauv_msgs/TunePidRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<TunePid-request>)))
  "Returns md5sum for a message object of type '<TunePid-request>"
  "c7f7e64a8df4cb6364bb9c28032e070e")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'TunePid-request)))
  "Returns md5sum for a message object of type 'TunePid-request"
  "c7f7e64a8df4cb6364bb9c28032e070e")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<TunePid-request>)))
  "Returns full string definition for message of type '<TunePid-request>"
  (cl:format cl:nil "tauv_msgs/PidVals vals~%~%================================================================================~%MSG: tauv_msgs/PidVals~%float32 a_p~%float32 a_i~%float32 a_d~%float32 a_sat~%float32 l_p~%float32 l_i~%float32 l_d~%float32 l_sat~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'TunePid-request)))
  "Returns full string definition for message of type 'TunePid-request"
  (cl:format cl:nil "tauv_msgs/PidVals vals~%~%================================================================================~%MSG: tauv_msgs/PidVals~%float32 a_p~%float32 a_i~%float32 a_d~%float32 a_sat~%float32 l_p~%float32 l_i~%float32 l_d~%float32 l_sat~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <TunePid-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'vals))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <TunePid-request>))
  "Converts a ROS message object to a list"
  (cl:list 'TunePid-request
    (cl:cons ':vals (vals msg))
))
;//! \htmlinclude TunePid-response.msg.html

(cl:defclass <TunePid-response> (roslisp-msg-protocol:ros-message)
  ((success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass TunePid-response (<TunePid-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <TunePid-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'TunePid-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-srv:<TunePid-response> is deprecated: use tauv_msgs-srv:TunePid-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <TunePid-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-srv:success-val is deprecated.  Use tauv_msgs-srv:success instead.")
  (success m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <TunePid-response>) ostream)
  "Serializes a message object of type '<TunePid-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <TunePid-response>) istream)
  "Deserializes a message object of type '<TunePid-response>"
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<TunePid-response>)))
  "Returns string type for a service object of type '<TunePid-response>"
  "tauv_msgs/TunePidResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'TunePid-response)))
  "Returns string type for a service object of type 'TunePid-response"
  "tauv_msgs/TunePidResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<TunePid-response>)))
  "Returns md5sum for a message object of type '<TunePid-response>"
  "c7f7e64a8df4cb6364bb9c28032e070e")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'TunePid-response)))
  "Returns md5sum for a message object of type 'TunePid-response"
  "c7f7e64a8df4cb6364bb9c28032e070e")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<TunePid-response>)))
  "Returns full string definition for message of type '<TunePid-response>"
  (cl:format cl:nil "bool success~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'TunePid-response)))
  "Returns full string definition for message of type 'TunePid-response"
  (cl:format cl:nil "bool success~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <TunePid-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <TunePid-response>))
  "Converts a ROS message object to a list"
  (cl:list 'TunePid-response
    (cl:cons ':success (success msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'TunePid)))
  'TunePid-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'TunePid)))
  'TunePid-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'TunePid)))
  "Returns string type for a service object of type '<TunePid>"
  "tauv_msgs/TunePid")