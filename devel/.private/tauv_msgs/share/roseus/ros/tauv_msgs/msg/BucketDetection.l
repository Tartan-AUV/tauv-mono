;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::BucketDetection)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'BucketDetection (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::BUCKETDETECTION")
  (make-package "TAUV_MSGS::BUCKETDETECTION"))

(in-package "ROS")
;;//! \htmlinclude BucketDetection.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))
(if (not (find-package "JSK_RECOGNITION_MSGS"))
  (ros::roseus-add-msgs "jsk_recognition_msgs"))
(if (not (find-package "SENSOR_MSGS"))
  (ros::roseus-add-msgs "sensor_msgs"))
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))
(if (not (find-package "VISION_MSGS"))
  (ros::roseus-add-msgs "vision_msgs"))


(defclass tauv_msgs::BucketDetection
  :super ros::object
  :slots (_header _position _length _width _height _normal _image _bbox_2d _bbox_3d _tag _detection_number ))

(defmethod tauv_msgs::BucketDetection
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:position __position) (instance geometry_msgs::Point :init))
    ((:length __length) 0.0)
    ((:width __width) 0.0)
    ((:height __height) 0.0)
    ((:normal __normal) (instance geometry_msgs::Vector3 :init))
    ((:image __image) (instance sensor_msgs::Image :init))
    ((:bbox_2d __bbox_2d) (instance vision_msgs::BoundingBox2D :init))
    ((:bbox_3d __bbox_3d) (instance jsk_recognition_msgs::BoundingBox :init))
    ((:tag __tag) "")
    ((:detection_number __detection_number) 0)
    )
   (send-super :init)
   (setq _header __header)
   (setq _position __position)
   (setq _length (float __length))
   (setq _width (float __width))
   (setq _height (float __height))
   (setq _normal __normal)
   (setq _image __image)
   (setq _bbox_2d __bbox_2d)
   (setq _bbox_3d __bbox_3d)
   (setq _tag (string __tag))
   (setq _detection_number (round __detection_number))
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:position
   (&rest __position)
   (if (keywordp (car __position))
       (send* _position __position)
     (progn
       (if __position (setq _position (car __position)))
       _position)))
  (:length
   (&optional __length)
   (if __length (setq _length __length)) _length)
  (:width
   (&optional __width)
   (if __width (setq _width __width)) _width)
  (:height
   (&optional __height)
   (if __height (setq _height __height)) _height)
  (:normal
   (&rest __normal)
   (if (keywordp (car __normal))
       (send* _normal __normal)
     (progn
       (if __normal (setq _normal (car __normal)))
       _normal)))
  (:image
   (&rest __image)
   (if (keywordp (car __image))
       (send* _image __image)
     (progn
       (if __image (setq _image (car __image)))
       _image)))
  (:bbox_2d
   (&rest __bbox_2d)
   (if (keywordp (car __bbox_2d))
       (send* _bbox_2d __bbox_2d)
     (progn
       (if __bbox_2d (setq _bbox_2d (car __bbox_2d)))
       _bbox_2d)))
  (:bbox_3d
   (&rest __bbox_3d)
   (if (keywordp (car __bbox_3d))
       (send* _bbox_3d __bbox_3d)
     (progn
       (if __bbox_3d (setq _bbox_3d (car __bbox_3d)))
       _bbox_3d)))
  (:tag
   (&optional __tag)
   (if __tag (setq _tag __tag)) _tag)
  (:detection_number
   (&optional __detection_number)
   (if __detection_number (setq _detection_number __detection_number)) _detection_number)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; geometry_msgs/Point _position
    (send _position :serialization-length)
    ;; float32 _length
    4
    ;; float32 _width
    4
    ;; float32 _height
    4
    ;; geometry_msgs/Vector3 _normal
    (send _normal :serialization-length)
    ;; sensor_msgs/Image _image
    (send _image :serialization-length)
    ;; vision_msgs/BoundingBox2D _bbox_2d
    (send _bbox_2d :serialization-length)
    ;; jsk_recognition_msgs/BoundingBox _bbox_3d
    (send _bbox_3d :serialization-length)
    ;; string _tag
    4 (length _tag)
    ;; uint32 _detection_number
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; geometry_msgs/Point _position
       (send _position :serialize s)
     ;; float32 _length
       (sys::poke _length (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _width
       (sys::poke _width (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _height
       (sys::poke _height (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; geometry_msgs/Vector3 _normal
       (send _normal :serialize s)
     ;; sensor_msgs/Image _image
       (send _image :serialize s)
     ;; vision_msgs/BoundingBox2D _bbox_2d
       (send _bbox_2d :serialize s)
     ;; jsk_recognition_msgs/BoundingBox _bbox_3d
       (send _bbox_3d :serialize s)
     ;; string _tag
       (write-long (length _tag) s) (princ _tag s)
     ;; uint32 _detection_number
       (write-long _detection_number s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; geometry_msgs/Point _position
     (send _position :deserialize buf ptr-) (incf ptr- (send _position :serialization-length))
   ;; float32 _length
     (setq _length (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _width
     (setq _width (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _height
     (setq _height (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; geometry_msgs/Vector3 _normal
     (send _normal :deserialize buf ptr-) (incf ptr- (send _normal :serialization-length))
   ;; sensor_msgs/Image _image
     (send _image :deserialize buf ptr-) (incf ptr- (send _image :serialization-length))
   ;; vision_msgs/BoundingBox2D _bbox_2d
     (send _bbox_2d :deserialize buf ptr-) (incf ptr- (send _bbox_2d :serialization-length))
   ;; jsk_recognition_msgs/BoundingBox _bbox_3d
     (send _bbox_3d :deserialize buf ptr-) (incf ptr- (send _bbox_3d :serialization-length))
   ;; string _tag
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _tag (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; uint32 _detection_number
     (setq _detection_number (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get tauv_msgs::BucketDetection :md5sum-) "41448fa5d4f902dafc6c1063c8e50467")
(setf (get tauv_msgs::BucketDetection :datatype-) "tauv_msgs/BucketDetection")
(setf (get tauv_msgs::BucketDetection :definition-)
      "Header header
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

")



(provide :tauv_msgs/BucketDetection "41448fa5d4f902dafc6c1063c8e50467")


