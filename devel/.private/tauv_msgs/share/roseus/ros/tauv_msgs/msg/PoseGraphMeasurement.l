;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::PoseGraphMeasurement)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'PoseGraphMeasurement (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::POSEGRAPHMEASUREMENT")
  (make-package "TAUV_MSGS::POSEGRAPHMEASUREMENT"))

(in-package "ROS")
;;//! \htmlinclude PoseGraphMeasurement.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass tauv_msgs::PoseGraphMeasurement
  :super ros::object
  :slots (_header _landmark_id _position ))

(defmethod tauv_msgs::PoseGraphMeasurement
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:landmark_id __landmark_id) 0)
    ((:position __position) (instance geometry_msgs::Point :init))
    )
   (send-super :init)
   (setq _header __header)
   (setq _landmark_id (round __landmark_id))
   (setq _position __position)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:landmark_id
   (&optional __landmark_id)
   (if __landmark_id (setq _landmark_id __landmark_id)) _landmark_id)
  (:position
   (&rest __position)
   (if (keywordp (car __position))
       (send* _position __position)
     (progn
       (if __position (setq _position (car __position)))
       _position)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; uint32 _landmark_id
    4
    ;; geometry_msgs/Point _position
    (send _position :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; uint32 _landmark_id
       (write-long _landmark_id s)
     ;; geometry_msgs/Point _position
       (send _position :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; uint32 _landmark_id
     (setq _landmark_id (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; geometry_msgs/Point _position
     (send _position :deserialize buf ptr-) (incf ptr- (send _position :serialization-length))
   ;;
   self)
  )

(setf (get tauv_msgs::PoseGraphMeasurement :md5sum-) "7e27f94f75bfdd40c9e3cb1c46d27f36")
(setf (get tauv_msgs::PoseGraphMeasurement :datatype-) "tauv_msgs/PoseGraphMeasurement")
(setf (get tauv_msgs::PoseGraphMeasurement :definition-)
      "Header header
uint32 landmark_id
geometry_msgs/Point position


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

")



(provide :tauv_msgs/PoseGraphMeasurement "7e27f94f75bfdd40c9e3cb1c46d27f36")


