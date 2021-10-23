;; Auto-generated. Do not edit!


(when (boundp 'tauv_common::RegisterMeasurement)
  (if (not (find-package "TAUV_COMMON"))
    (make-package "TAUV_COMMON"))
  (shadow 'RegisterMeasurement (find-package "TAUV_COMMON")))
(unless (find-package "TAUV_COMMON::REGISTERMEASUREMENT")
  (make-package "TAUV_COMMON::REGISTERMEASUREMENT"))
(unless (find-package "TAUV_COMMON::REGISTERMEASUREMENTREQUEST")
  (make-package "TAUV_COMMON::REGISTERMEASUREMENTREQUEST"))
(unless (find-package "TAUV_COMMON::REGISTERMEASUREMENTRESPONSE")
  (make-package "TAUV_COMMON::REGISTERMEASUREMENTRESPONSE"))

(in-package "ROS")

(if (not (find-package "TAUV_MSGS"))
  (ros::roseus-add-msgs "tauv_msgs"))




(defclass tauv_common::RegisterMeasurementRequest
  :super ros::object
  :slots (_pg_meas ))

(defmethod tauv_common::RegisterMeasurementRequest
  (:init
   (&key
    ((:pg_meas __pg_meas) (instance tauv_msgs::PoseGraphMeasurement :init))
    )
   (send-super :init)
   (setq _pg_meas __pg_meas)
   self)
  (:pg_meas
   (&rest __pg_meas)
   (if (keywordp (car __pg_meas))
       (send* _pg_meas __pg_meas)
     (progn
       (if __pg_meas (setq _pg_meas (car __pg_meas)))
       _pg_meas)))
  (:serialization-length
   ()
   (+
    ;; tauv_msgs/PoseGraphMeasurement _pg_meas
    (send _pg_meas :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; tauv_msgs/PoseGraphMeasurement _pg_meas
       (send _pg_meas :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; tauv_msgs/PoseGraphMeasurement _pg_meas
     (send _pg_meas :deserialize buf ptr-) (incf ptr- (send _pg_meas :serialization-length))
   ;;
   self)
  )

(defclass tauv_common::RegisterMeasurementResponse
  :super ros::object
  :slots (_success ))

(defmethod tauv_common::RegisterMeasurementResponse
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

(defclass tauv_common::RegisterMeasurement
  :super ros::object
  :slots ())

(setf (get tauv_common::RegisterMeasurement :md5sum-) "f2167c58c6c958d121ce25746a34db61")
(setf (get tauv_common::RegisterMeasurement :datatype-) "tauv_common/RegisterMeasurement")
(setf (get tauv_common::RegisterMeasurement :request) tauv_common::RegisterMeasurementRequest)
(setf (get tauv_common::RegisterMeasurement :response) tauv_common::RegisterMeasurementResponse)

(defmethod tauv_common::RegisterMeasurementRequest
  (:response () (instance tauv_common::RegisterMeasurementResponse :init)))

(setf (get tauv_common::RegisterMeasurementRequest :md5sum-) "f2167c58c6c958d121ce25746a34db61")
(setf (get tauv_common::RegisterMeasurementRequest :datatype-) "tauv_common/RegisterMeasurementRequest")
(setf (get tauv_common::RegisterMeasurementRequest :definition-)
      "tauv_msgs/PoseGraphMeasurement pg_meas

================================================================================
MSG: tauv_msgs/PoseGraphMeasurement
Header header
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
---
bool success
")

(setf (get tauv_common::RegisterMeasurementResponse :md5sum-) "f2167c58c6c958d121ce25746a34db61")
(setf (get tauv_common::RegisterMeasurementResponse :datatype-) "tauv_common/RegisterMeasurementResponse")
(setf (get tauv_common::RegisterMeasurementResponse :definition-)
      "tauv_msgs/PoseGraphMeasurement pg_meas

================================================================================
MSG: tauv_msgs/PoseGraphMeasurement
Header header
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
---
bool success
")



(provide :tauv_common/RegisterMeasurement "f2167c58c6c958d121ce25746a34db61")


