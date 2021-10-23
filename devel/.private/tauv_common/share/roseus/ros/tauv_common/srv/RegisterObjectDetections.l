;; Auto-generated. Do not edit!


(when (boundp 'tauv_common::RegisterObjectDetections)
  (if (not (find-package "TAUV_COMMON"))
    (make-package "TAUV_COMMON"))
  (shadow 'RegisterObjectDetections (find-package "TAUV_COMMON")))
(unless (find-package "TAUV_COMMON::REGISTEROBJECTDETECTIONS")
  (make-package "TAUV_COMMON::REGISTEROBJECTDETECTIONS"))
(unless (find-package "TAUV_COMMON::REGISTEROBJECTDETECTIONSREQUEST")
  (make-package "TAUV_COMMON::REGISTEROBJECTDETECTIONSREQUEST"))
(unless (find-package "TAUV_COMMON::REGISTEROBJECTDETECTIONSRESPONSE")
  (make-package "TAUV_COMMON::REGISTEROBJECTDETECTIONSRESPONSE"))

(in-package "ROS")

(if (not (find-package "TAUV_MSGS"))
  (ros::roseus-add-msgs "tauv_msgs"))




(defclass tauv_common::RegisterObjectDetectionsRequest
  :super ros::object
  :slots (_objdets _detector_tag ))

(defmethod tauv_common::RegisterObjectDetectionsRequest
  (:init
   (&key
    ((:objdets __objdets) ())
    ((:detector_tag __detector_tag) "")
    )
   (send-super :init)
   (setq _objdets __objdets)
   (setq _detector_tag (string __detector_tag))
   self)
  (:objdets
   (&rest __objdets)
   (if (keywordp (car __objdets))
       (send* _objdets __objdets)
     (progn
       (if __objdets (setq _objdets (car __objdets)))
       _objdets)))
  (:detector_tag
   (&optional __detector_tag)
   (if __detector_tag (setq _detector_tag __detector_tag)) _detector_tag)
  (:serialization-length
   ()
   (+
    ;; tauv_msgs/BucketDetection[] _objdets
    (apply #'+ (send-all _objdets :serialization-length)) 4
    ;; string _detector_tag
    4 (length _detector_tag)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; tauv_msgs/BucketDetection[] _objdets
     (write-long (length _objdets) s)
     (dolist (elem _objdets)
       (send elem :serialize s)
       )
     ;; string _detector_tag
       (write-long (length _detector_tag) s) (princ _detector_tag s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; tauv_msgs/BucketDetection[] _objdets
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _objdets (let (r) (dotimes (i n) (push (instance tauv_msgs::BucketDetection :init) r)) r))
     (dolist (elem- _objdets)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;; string _detector_tag
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _detector_tag (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass tauv_common::RegisterObjectDetectionsResponse
  :super ros::object
  :slots (_success ))

(defmethod tauv_common::RegisterObjectDetectionsResponse
  (:init
   (&key
    ((:success __success) nil)
    )
   (send-super :init)
   (setq _success __success)
   self)
  (:success
   (&optional (__success :null))
   (if (not (eq __success :null)) (setq _success __success)) _success)
  (:serialization-length
   ()
   (+
    ;; bool _success
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _success
       (if _success (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _success
     (setq _success (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(defclass tauv_common::RegisterObjectDetections
  :super ros::object
  :slots ())

(setf (get tauv_common::RegisterObjectDetections :md5sum-) "5cc8e5672e1e74a20b716902a91f95c7")
(setf (get tauv_common::RegisterObjectDetections :datatype-) "tauv_common/RegisterObjectDetections")
(setf (get tauv_common::RegisterObjectDetections :request) tauv_common::RegisterObjectDetectionsRequest)
(setf (get tauv_common::RegisterObjectDetections :response) tauv_common::RegisterObjectDetectionsResponse)

(defmethod tauv_common::RegisterObjectDetectionsRequest
  (:response () (instance tauv_common::RegisterObjectDetectionsResponse :init)))

(setf (get tauv_common::RegisterObjectDetectionsRequest :md5sum-) "5cc8e5672e1e74a20b716902a91f95c7")
(setf (get tauv_common::RegisterObjectDetectionsRequest :datatype-) "tauv_common/RegisterObjectDetectionsRequest")
(setf (get tauv_common::RegisterObjectDetectionsRequest :definition-)
      "tauv_msgs/BucketDetection[] objdets
string detector_tag

================================================================================
MSG: tauv_msgs/BucketDetection
Header header
geometry_msgs/Point position
float32 length
float32 width
float32 height
geometry_msgs/Vector3 normal
sensor_msgs/Image image
vision_msgs/BoundingBox2D bbox_2d
jsk_recognition_msgs/BoundingBox bbox_3d
string tag
uint32 detection_number
================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Vector3
# This represents a vector in free space. 
# It is only meant to represent a direction. Therefore, it does not
# make sense to apply a translation to it (e.g., when applying a 
# generic rigid transformation to a Vector3, tf2 will only apply the
# rotation). If you want your data to be translatable too, use the
# geometry_msgs/Point message instead.

float64 x
float64 y
float64 z
================================================================================
MSG: sensor_msgs/Image
# This message contains an uncompressed image
# (0, 0) is at top-left corner of image
#

Header header        # Header timestamp should be acquisition time of image
                     # Header frame_id should be optical frame of camera
                     # origin of frame should be optical center of camera
                     # +x should point to the right in the image
                     # +y should point down in the image
                     # +z should point into to plane of the image
                     # If the frame_id here and the frame_id of the CameraInfo
                     # message associated with the image conflict
                     # the behavior is undefined

uint32 height         # image height, that is, number of rows
uint32 width          # image width, that is, number of columns

# The legal values for encoding are in file src/image_encodings.cpp
# If you want to standardize a new string format, join
# ros-users@lists.sourceforge.net and send an email proposing a new encoding.

string encoding       # Encoding of pixels -- channel meaning, ordering, size
                      # taken from the list of strings in include/sensor_msgs/image_encodings.h

uint8 is_bigendian    # is this data bigendian?
uint32 step           # Full row length in bytes
uint8[] data          # actual matrix data, size is (step * rows)

================================================================================
MSG: vision_msgs/BoundingBox2D
# A 2D bounding box that can be rotated about its center.
# All dimensions are in pixels, but represented using floating-point
#   values to allow sub-pixel precision. If an exact pixel crop is required
#   for a rotated bounding box, it can be calculated using Bresenham's line
#   algorithm.

# The 2D position (in pixels) and orientation of the bounding box center.
geometry_msgs/Pose2D center

# The size (in pixels) of the bounding box surrounding the object relative
#   to the pose of its center.
float64 size_x
float64 size_y

================================================================================
MSG: geometry_msgs/Pose2D
# Deprecated
# Please use the full 3D pose.

# In general our recommendation is to use a full 3D representation of everything and for 2D specific applications make the appropriate projections into the plane for their calculations but optimally will preserve the 3D information during processing.

# If we have parallel copies of 2D datatypes every UI and other pipeline will end up needing to have dual interfaces to plot everything. And you will end up with not being able to use 3D tools for 2D use cases even if they're completely valid, as you'd have to reimplement it with different inputs and outputs. It's not particularly hard to plot the 2D pose or compute the yaw error for the Pose message and there are already tools and libraries that can do this for you.


# This expresses a position and orientation on a 2D manifold.

float64 x
float64 y
float64 theta

================================================================================
MSG: jsk_recognition_msgs/BoundingBox
# BoundingBox represents a oriented bounding box.
Header header
geometry_msgs/Pose pose
geometry_msgs/Vector3 dimensions  # size of bounding box (x, y, z)
# You can use this field to hold value such as likelihood
float32 value
uint32 label

================================================================================
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation. 
Point position
Quaternion orientation

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w
---
bool success
")

(setf (get tauv_common::RegisterObjectDetectionsResponse :md5sum-) "5cc8e5672e1e74a20b716902a91f95c7")
(setf (get tauv_common::RegisterObjectDetectionsResponse :datatype-) "tauv_common/RegisterObjectDetectionsResponse")
(setf (get tauv_common::RegisterObjectDetectionsResponse :definition-)
      "tauv_msgs/BucketDetection[] objdets
string detector_tag

================================================================================
MSG: tauv_msgs/BucketDetection
Header header
geometry_msgs/Point position
float32 length
float32 width
float32 height
geometry_msgs/Vector3 normal
sensor_msgs/Image image
vision_msgs/BoundingBox2D bbox_2d
jsk_recognition_msgs/BoundingBox bbox_3d
string tag
uint32 detection_number
================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Vector3
# This represents a vector in free space. 
# It is only meant to represent a direction. Therefore, it does not
# make sense to apply a translation to it (e.g., when applying a 
# generic rigid transformation to a Vector3, tf2 will only apply the
# rotation). If you want your data to be translatable too, use the
# geometry_msgs/Point message instead.

float64 x
float64 y
float64 z
================================================================================
MSG: sensor_msgs/Image
# This message contains an uncompressed image
# (0, 0) is at top-left corner of image
#

Header header        # Header timestamp should be acquisition time of image
                     # Header frame_id should be optical frame of camera
                     # origin of frame should be optical center of camera
                     # +x should point to the right in the image
                     # +y should point down in the image
                     # +z should point into to plane of the image
                     # If the frame_id here and the frame_id of the CameraInfo
                     # message associated with the image conflict
                     # the behavior is undefined

uint32 height         # image height, that is, number of rows
uint32 width          # image width, that is, number of columns

# The legal values for encoding are in file src/image_encodings.cpp
# If you want to standardize a new string format, join
# ros-users@lists.sourceforge.net and send an email proposing a new encoding.

string encoding       # Encoding of pixels -- channel meaning, ordering, size
                      # taken from the list of strings in include/sensor_msgs/image_encodings.h

uint8 is_bigendian    # is this data bigendian?
uint32 step           # Full row length in bytes
uint8[] data          # actual matrix data, size is (step * rows)

================================================================================
MSG: vision_msgs/BoundingBox2D
# A 2D bounding box that can be rotated about its center.
# All dimensions are in pixels, but represented using floating-point
#   values to allow sub-pixel precision. If an exact pixel crop is required
#   for a rotated bounding box, it can be calculated using Bresenham's line
#   algorithm.

# The 2D position (in pixels) and orientation of the bounding box center.
geometry_msgs/Pose2D center

# The size (in pixels) of the bounding box surrounding the object relative
#   to the pose of its center.
float64 size_x
float64 size_y

================================================================================
MSG: geometry_msgs/Pose2D
# Deprecated
# Please use the full 3D pose.

# In general our recommendation is to use a full 3D representation of everything and for 2D specific applications make the appropriate projections into the plane for their calculations but optimally will preserve the 3D information during processing.

# If we have parallel copies of 2D datatypes every UI and other pipeline will end up needing to have dual interfaces to plot everything. And you will end up with not being able to use 3D tools for 2D use cases even if they're completely valid, as you'd have to reimplement it with different inputs and outputs. It's not particularly hard to plot the 2D pose or compute the yaw error for the Pose message and there are already tools and libraries that can do this for you.


# This expresses a position and orientation on a 2D manifold.

float64 x
float64 y
float64 theta

================================================================================
MSG: jsk_recognition_msgs/BoundingBox
# BoundingBox represents a oriented bounding box.
Header header
geometry_msgs/Pose pose
geometry_msgs/Vector3 dimensions  # size of bounding box (x, y, z)
# You can use this field to hold value such as likelihood
float32 value
uint32 label

================================================================================
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation. 
Point position
Quaternion orientation

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w
---
bool success
")



(provide :tauv_common/RegisterObjectDetections "5cc8e5672e1e74a20b716902a91f95c7")


