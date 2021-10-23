; Auto-generated. Do not edit!


(cl:in-package tauv_common-srv)


;//! \htmlinclude RegisterObjectDetections-request.msg.html

(cl:defclass <RegisterObjectDetections-request> (roslisp-msg-protocol:ros-message)
  ((objdets
    :reader objdets
    :initarg :objdets
    :type (cl:vector tauv_msgs-msg:BucketDetection)
   :initform (cl:make-array 0 :element-type 'tauv_msgs-msg:BucketDetection :initial-element (cl:make-instance 'tauv_msgs-msg:BucketDetection)))
   (detector_tag
    :reader detector_tag
    :initarg :detector_tag
    :type cl:string
    :initform ""))
)

(cl:defclass RegisterObjectDetections-request (<RegisterObjectDetections-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <RegisterObjectDetections-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'RegisterObjectDetections-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_common-srv:<RegisterObjectDetections-request> is deprecated: use tauv_common-srv:RegisterObjectDetections-request instead.")))

(cl:ensure-generic-function 'objdets-val :lambda-list '(m))
(cl:defmethod objdets-val ((m <RegisterObjectDetections-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_common-srv:objdets-val is deprecated.  Use tauv_common-srv:objdets instead.")
  (objdets m))

(cl:ensure-generic-function 'detector_tag-val :lambda-list '(m))
(cl:defmethod detector_tag-val ((m <RegisterObjectDetections-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_common-srv:detector_tag-val is deprecated.  Use tauv_common-srv:detector_tag instead.")
  (detector_tag m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <RegisterObjectDetections-request>) ostream)
  "Serializes a message object of type '<RegisterObjectDetections-request>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'objdets))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'objdets))
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'detector_tag))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'detector_tag))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <RegisterObjectDetections-request>) istream)
  "Deserializes a message object of type '<RegisterObjectDetections-request>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'objdets) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'objdets)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'tauv_msgs-msg:BucketDetection))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'detector_tag) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'detector_tag) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<RegisterObjectDetections-request>)))
  "Returns string type for a service object of type '<RegisterObjectDetections-request>"
  "tauv_common/RegisterObjectDetectionsRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'RegisterObjectDetections-request)))
  "Returns string type for a service object of type 'RegisterObjectDetections-request"
  "tauv_common/RegisterObjectDetectionsRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<RegisterObjectDetections-request>)))
  "Returns md5sum for a message object of type '<RegisterObjectDetections-request>"
  "5cc8e5672e1e74a20b716902a91f95c7")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'RegisterObjectDetections-request)))
  "Returns md5sum for a message object of type 'RegisterObjectDetections-request"
  "5cc8e5672e1e74a20b716902a91f95c7")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<RegisterObjectDetections-request>)))
  "Returns full string definition for message of type '<RegisterObjectDetections-request>"
  (cl:format cl:nil "tauv_msgs/BucketDetection[] objdets~%string detector_tag~%~%================================================================================~%MSG: tauv_msgs/BucketDetection~%Header header~%geometry_msgs/Point position~%float32 length~%float32 width~%float32 height~%geometry_msgs/Vector3 normal~%sensor_msgs/Image image~%vision_msgs/BoundingBox2D bbox_2d~%jsk_recognition_msgs/BoundingBox bbox_3d~%string tag~%uint32 detection_number~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%================================================================================~%MSG: sensor_msgs/Image~%# This message contains an uncompressed image~%# (0, 0) is at top-left corner of image~%#~%~%Header header        # Header timestamp should be acquisition time of image~%                     # Header frame_id should be optical frame of camera~%                     # origin of frame should be optical center of camera~%                     # +x should point to the right in the image~%                     # +y should point down in the image~%                     # +z should point into to plane of the image~%                     # If the frame_id here and the frame_id of the CameraInfo~%                     # message associated with the image conflict~%                     # the behavior is undefined~%~%uint32 height         # image height, that is, number of rows~%uint32 width          # image width, that is, number of columns~%~%# The legal values for encoding are in file src/image_encodings.cpp~%# If you want to standardize a new string format, join~%# ros-users@lists.sourceforge.net and send an email proposing a new encoding.~%~%string encoding       # Encoding of pixels -- channel meaning, ordering, size~%                      # taken from the list of strings in include/sensor_msgs/image_encodings.h~%~%uint8 is_bigendian    # is this data bigendian?~%uint32 step           # Full row length in bytes~%uint8[] data          # actual matrix data, size is (step * rows)~%~%================================================================================~%MSG: vision_msgs/BoundingBox2D~%# A 2D bounding box that can be rotated about its center.~%# All dimensions are in pixels, but represented using floating-point~%#   values to allow sub-pixel precision. If an exact pixel crop is required~%#   for a rotated bounding box, it can be calculated using Bresenham's line~%#   algorithm.~%~%# The 2D position (in pixels) and orientation of the bounding box center.~%geometry_msgs/Pose2D center~%~%# The size (in pixels) of the bounding box surrounding the object relative~%#   to the pose of its center.~%float64 size_x~%float64 size_y~%~%================================================================================~%MSG: geometry_msgs/Pose2D~%# Deprecated~%# Please use the full 3D pose.~%~%# In general our recommendation is to use a full 3D representation of everything and for 2D specific applications make the appropriate projections into the plane for their calculations but optimally will preserve the 3D information during processing.~%~%# If we have parallel copies of 2D datatypes every UI and other pipeline will end up needing to have dual interfaces to plot everything. And you will end up with not being able to use 3D tools for 2D use cases even if they're completely valid, as you'd have to reimplement it with different inputs and outputs. It's not particularly hard to plot the 2D pose or compute the yaw error for the Pose message and there are already tools and libraries that can do this for you.~%~%~%# This expresses a position and orientation on a 2D manifold.~%~%float64 x~%float64 y~%float64 theta~%~%================================================================================~%MSG: jsk_recognition_msgs/BoundingBox~%# BoundingBox represents a oriented bounding box.~%Header header~%geometry_msgs/Pose pose~%geometry_msgs/Vector3 dimensions  # size of bounding box (x, y, z)~%# You can use this field to hold value such as likelihood~%float32 value~%uint32 label~%~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'RegisterObjectDetections-request)))
  "Returns full string definition for message of type 'RegisterObjectDetections-request"
  (cl:format cl:nil "tauv_msgs/BucketDetection[] objdets~%string detector_tag~%~%================================================================================~%MSG: tauv_msgs/BucketDetection~%Header header~%geometry_msgs/Point position~%float32 length~%float32 width~%float32 height~%geometry_msgs/Vector3 normal~%sensor_msgs/Image image~%vision_msgs/BoundingBox2D bbox_2d~%jsk_recognition_msgs/BoundingBox bbox_3d~%string tag~%uint32 detection_number~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%================================================================================~%MSG: sensor_msgs/Image~%# This message contains an uncompressed image~%# (0, 0) is at top-left corner of image~%#~%~%Header header        # Header timestamp should be acquisition time of image~%                     # Header frame_id should be optical frame of camera~%                     # origin of frame should be optical center of camera~%                     # +x should point to the right in the image~%                     # +y should point down in the image~%                     # +z should point into to plane of the image~%                     # If the frame_id here and the frame_id of the CameraInfo~%                     # message associated with the image conflict~%                     # the behavior is undefined~%~%uint32 height         # image height, that is, number of rows~%uint32 width          # image width, that is, number of columns~%~%# The legal values for encoding are in file src/image_encodings.cpp~%# If you want to standardize a new string format, join~%# ros-users@lists.sourceforge.net and send an email proposing a new encoding.~%~%string encoding       # Encoding of pixels -- channel meaning, ordering, size~%                      # taken from the list of strings in include/sensor_msgs/image_encodings.h~%~%uint8 is_bigendian    # is this data bigendian?~%uint32 step           # Full row length in bytes~%uint8[] data          # actual matrix data, size is (step * rows)~%~%================================================================================~%MSG: vision_msgs/BoundingBox2D~%# A 2D bounding box that can be rotated about its center.~%# All dimensions are in pixels, but represented using floating-point~%#   values to allow sub-pixel precision. If an exact pixel crop is required~%#   for a rotated bounding box, it can be calculated using Bresenham's line~%#   algorithm.~%~%# The 2D position (in pixels) and orientation of the bounding box center.~%geometry_msgs/Pose2D center~%~%# The size (in pixels) of the bounding box surrounding the object relative~%#   to the pose of its center.~%float64 size_x~%float64 size_y~%~%================================================================================~%MSG: geometry_msgs/Pose2D~%# Deprecated~%# Please use the full 3D pose.~%~%# In general our recommendation is to use a full 3D representation of everything and for 2D specific applications make the appropriate projections into the plane for their calculations but optimally will preserve the 3D information during processing.~%~%# If we have parallel copies of 2D datatypes every UI and other pipeline will end up needing to have dual interfaces to plot everything. And you will end up with not being able to use 3D tools for 2D use cases even if they're completely valid, as you'd have to reimplement it with different inputs and outputs. It's not particularly hard to plot the 2D pose or compute the yaw error for the Pose message and there are already tools and libraries that can do this for you.~%~%~%# This expresses a position and orientation on a 2D manifold.~%~%float64 x~%float64 y~%float64 theta~%~%================================================================================~%MSG: jsk_recognition_msgs/BoundingBox~%# BoundingBox represents a oriented bounding box.~%Header header~%geometry_msgs/Pose pose~%geometry_msgs/Vector3 dimensions  # size of bounding box (x, y, z)~%# You can use this field to hold value such as likelihood~%float32 value~%uint32 label~%~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <RegisterObjectDetections-request>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'objdets) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
     4 (cl:length (cl:slot-value msg 'detector_tag))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <RegisterObjectDetections-request>))
  "Converts a ROS message object to a list"
  (cl:list 'RegisterObjectDetections-request
    (cl:cons ':objdets (objdets msg))
    (cl:cons ':detector_tag (detector_tag msg))
))
;//! \htmlinclude RegisterObjectDetections-response.msg.html

(cl:defclass <RegisterObjectDetections-response> (roslisp-msg-protocol:ros-message)
  ((success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass RegisterObjectDetections-response (<RegisterObjectDetections-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <RegisterObjectDetections-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'RegisterObjectDetections-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_common-srv:<RegisterObjectDetections-response> is deprecated: use tauv_common-srv:RegisterObjectDetections-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <RegisterObjectDetections-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_common-srv:success-val is deprecated.  Use tauv_common-srv:success instead.")
  (success m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <RegisterObjectDetections-response>) ostream)
  "Serializes a message object of type '<RegisterObjectDetections-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <RegisterObjectDetections-response>) istream)
  "Deserializes a message object of type '<RegisterObjectDetections-response>"
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<RegisterObjectDetections-response>)))
  "Returns string type for a service object of type '<RegisterObjectDetections-response>"
  "tauv_common/RegisterObjectDetectionsResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'RegisterObjectDetections-response)))
  "Returns string type for a service object of type 'RegisterObjectDetections-response"
  "tauv_common/RegisterObjectDetectionsResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<RegisterObjectDetections-response>)))
  "Returns md5sum for a message object of type '<RegisterObjectDetections-response>"
  "5cc8e5672e1e74a20b716902a91f95c7")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'RegisterObjectDetections-response)))
  "Returns md5sum for a message object of type 'RegisterObjectDetections-response"
  "5cc8e5672e1e74a20b716902a91f95c7")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<RegisterObjectDetections-response>)))
  "Returns full string definition for message of type '<RegisterObjectDetections-response>"
  (cl:format cl:nil "bool success~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'RegisterObjectDetections-response)))
  "Returns full string definition for message of type 'RegisterObjectDetections-response"
  (cl:format cl:nil "bool success~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <RegisterObjectDetections-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <RegisterObjectDetections-response>))
  "Converts a ROS message object to a list"
  (cl:list 'RegisterObjectDetections-response
    (cl:cons ':success (success msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'RegisterObjectDetections)))
  'RegisterObjectDetections-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'RegisterObjectDetections)))
  'RegisterObjectDetections-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'RegisterObjectDetections)))
  "Returns string type for a service object of type '<RegisterObjectDetections>"
  "tauv_common/RegisterObjectDetections")