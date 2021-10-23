; Auto-generated. Do not edit!


(cl:in-package tauv_joy-srv)


;//! \htmlinclude JoyConnect-request.msg.html

(cl:defclass <JoyConnect-request> (roslisp-msg-protocol:ros-message)
  ((dev
    :reader dev
    :initarg :dev
    :type cl:string
    :initform ""))
)

(cl:defclass JoyConnect-request (<JoyConnect-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <JoyConnect-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'JoyConnect-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_joy-srv:<JoyConnect-request> is deprecated: use tauv_joy-srv:JoyConnect-request instead.")))

(cl:ensure-generic-function 'dev-val :lambda-list '(m))
(cl:defmethod dev-val ((m <JoyConnect-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_joy-srv:dev-val is deprecated.  Use tauv_joy-srv:dev instead.")
  (dev m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <JoyConnect-request>) ostream)
  "Serializes a message object of type '<JoyConnect-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'dev))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'dev))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <JoyConnect-request>) istream)
  "Deserializes a message object of type '<JoyConnect-request>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'dev) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'dev) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<JoyConnect-request>)))
  "Returns string type for a service object of type '<JoyConnect-request>"
  "tauv_joy/JoyConnectRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'JoyConnect-request)))
  "Returns string type for a service object of type 'JoyConnect-request"
  "tauv_joy/JoyConnectRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<JoyConnect-request>)))
  "Returns md5sum for a message object of type '<JoyConnect-request>"
  "4ac0afce82f389644128ac3ab46cf672")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'JoyConnect-request)))
  "Returns md5sum for a message object of type 'JoyConnect-request"
  "4ac0afce82f389644128ac3ab46cf672")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<JoyConnect-request>)))
  "Returns full string definition for message of type '<JoyConnect-request>"
  (cl:format cl:nil "string dev # device file path~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'JoyConnect-request)))
  "Returns full string definition for message of type 'JoyConnect-request"
  (cl:format cl:nil "string dev # device file path~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <JoyConnect-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'dev))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <JoyConnect-request>))
  "Converts a ROS message object to a list"
  (cl:list 'JoyConnect-request
    (cl:cons ':dev (dev msg))
))
;//! \htmlinclude JoyConnect-response.msg.html

(cl:defclass <JoyConnect-response> (roslisp-msg-protocol:ros-message)
  ((success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil)
   (message
    :reader message
    :initarg :message
    :type cl:string
    :initform ""))
)

(cl:defclass JoyConnect-response (<JoyConnect-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <JoyConnect-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'JoyConnect-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_joy-srv:<JoyConnect-response> is deprecated: use tauv_joy-srv:JoyConnect-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <JoyConnect-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_joy-srv:success-val is deprecated.  Use tauv_joy-srv:success instead.")
  (success m))

(cl:ensure-generic-function 'message-val :lambda-list '(m))
(cl:defmethod message-val ((m <JoyConnect-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_joy-srv:message-val is deprecated.  Use tauv_joy-srv:message instead.")
  (message m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <JoyConnect-response>) ostream)
  "Serializes a message object of type '<JoyConnect-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'message))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'message))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <JoyConnect-response>) istream)
  "Deserializes a message object of type '<JoyConnect-response>"
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'message) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'message) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<JoyConnect-response>)))
  "Returns string type for a service object of type '<JoyConnect-response>"
  "tauv_joy/JoyConnectResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'JoyConnect-response)))
  "Returns string type for a service object of type 'JoyConnect-response"
  "tauv_joy/JoyConnectResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<JoyConnect-response>)))
  "Returns md5sum for a message object of type '<JoyConnect-response>"
  "4ac0afce82f389644128ac3ab46cf672")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'JoyConnect-response)))
  "Returns md5sum for a message object of type 'JoyConnect-response"
  "4ac0afce82f389644128ac3ab46cf672")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<JoyConnect-response>)))
  "Returns full string definition for message of type '<JoyConnect-response>"
  (cl:format cl:nil "bool success   # indicate successful run of triggered service~%string message # informational, e.g. for error messages~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'JoyConnect-response)))
  "Returns full string definition for message of type 'JoyConnect-response"
  (cl:format cl:nil "bool success   # indicate successful run of triggered service~%string message # informational, e.g. for error messages~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <JoyConnect-response>))
  (cl:+ 0
     1
     4 (cl:length (cl:slot-value msg 'message))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <JoyConnect-response>))
  "Converts a ROS message object to a list"
  (cl:list 'JoyConnect-response
    (cl:cons ':success (success msg))
    (cl:cons ':message (message msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'JoyConnect)))
  'JoyConnect-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'JoyConnect)))
  'JoyConnect-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'JoyConnect)))
  "Returns string type for a service object of type '<JoyConnect>"
  "tauv_joy/JoyConnect")