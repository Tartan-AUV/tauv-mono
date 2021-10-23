;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::TuneInertial)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'TuneInertial (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::TUNEINERTIAL")
  (make-package "TAUV_MSGS::TUNEINERTIAL"))
(unless (find-package "TAUV_MSGS::TUNEINERTIALREQUEST")
  (make-package "TAUV_MSGS::TUNEINERTIALREQUEST"))
(unless (find-package "TAUV_MSGS::TUNEINERTIALRESPONSE")
  (make-package "TAUV_MSGS::TUNEINERTIALRESPONSE"))

(in-package "ROS")





(defclass tauv_msgs::TuneInertialRequest
  :super ros::object
  :slots (_vals ))

(defmethod tauv_msgs::TuneInertialRequest
  (:init
   (&key
    ((:vals __vals) (instance tauv_msgs::InertialVals :init))
    )
   (send-super :init)
   (setq _vals __vals)
   self)
  (:vals
   (&rest __vals)
   (if (keywordp (car __vals))
       (send* _vals __vals)
     (progn
       (if __vals (setq _vals (car __vals)))
       _vals)))
  (:serialization-length
   ()
   (+
    ;; tauv_msgs/InertialVals _vals
    (send _vals :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; tauv_msgs/InertialVals _vals
       (send _vals :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; tauv_msgs/InertialVals _vals
     (send _vals :deserialize buf ptr-) (incf ptr- (send _vals :serialization-length))
   ;;
   self)
  )

(defclass tauv_msgs::TuneInertialResponse
  :super ros::object
  :slots (_success ))

(defmethod tauv_msgs::TuneInertialResponse
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

(defclass tauv_msgs::TuneInertial
  :super ros::object
  :slots ())

(setf (get tauv_msgs::TuneInertial :md5sum-) "b07a11376c5b2ae9ec7888a1a74564d4")
(setf (get tauv_msgs::TuneInertial :datatype-) "tauv_msgs/TuneInertial")
(setf (get tauv_msgs::TuneInertial :request) tauv_msgs::TuneInertialRequest)
(setf (get tauv_msgs::TuneInertial :response) tauv_msgs::TuneInertialResponse)

(defmethod tauv_msgs::TuneInertialRequest
  (:response () (instance tauv_msgs::TuneInertialResponse :init)))

(setf (get tauv_msgs::TuneInertialRequest :md5sum-) "b07a11376c5b2ae9ec7888a1a74564d4")
(setf (get tauv_msgs::TuneInertialRequest :datatype-) "tauv_msgs/TuneInertialRequest")
(setf (get tauv_msgs::TuneInertialRequest :definition-)
      "tauv_msgs/InertialVals vals

================================================================================
MSG: tauv_msgs/InertialVals
float32 mass
float32 buoyancy
float32 ixx
float32 iyy
---
bool success
")

(setf (get tauv_msgs::TuneInertialResponse :md5sum-) "b07a11376c5b2ae9ec7888a1a74564d4")
(setf (get tauv_msgs::TuneInertialResponse :datatype-) "tauv_msgs/TuneInertialResponse")
(setf (get tauv_msgs::TuneInertialResponse :definition-)
      "tauv_msgs/InertialVals vals

================================================================================
MSG: tauv_msgs/InertialVals
float32 mass
float32 buoyancy
float32 ixx
float32 iyy
---
bool success
")



(provide :tauv_msgs/TuneInertial "b07a11376c5b2ae9ec7888a1a74564d4")


