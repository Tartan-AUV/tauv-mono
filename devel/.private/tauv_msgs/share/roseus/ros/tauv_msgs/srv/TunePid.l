;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::TunePid)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'TunePid (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::TUNEPID")
  (make-package "TAUV_MSGS::TUNEPID"))
(unless (find-package "TAUV_MSGS::TUNEPIDREQUEST")
  (make-package "TAUV_MSGS::TUNEPIDREQUEST"))
(unless (find-package "TAUV_MSGS::TUNEPIDRESPONSE")
  (make-package "TAUV_MSGS::TUNEPIDRESPONSE"))

(in-package "ROS")





(defclass tauv_msgs::TunePidRequest
  :super ros::object
  :slots (_vals ))

(defmethod tauv_msgs::TunePidRequest
  (:init
   (&key
    ((:vals __vals) (instance tauv_msgs::PidVals :init))
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
    ;; tauv_msgs/PidVals _vals
    (send _vals :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; tauv_msgs/PidVals _vals
       (send _vals :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; tauv_msgs/PidVals _vals
     (send _vals :deserialize buf ptr-) (incf ptr- (send _vals :serialization-length))
   ;;
   self)
  )

(defclass tauv_msgs::TunePidResponse
  :super ros::object
  :slots (_success ))

(defmethod tauv_msgs::TunePidResponse
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

(defclass tauv_msgs::TunePid
  :super ros::object
  :slots ())

(setf (get tauv_msgs::TunePid :md5sum-) "c7f7e64a8df4cb6364bb9c28032e070e")
(setf (get tauv_msgs::TunePid :datatype-) "tauv_msgs/TunePid")
(setf (get tauv_msgs::TunePid :request) tauv_msgs::TunePidRequest)
(setf (get tauv_msgs::TunePid :response) tauv_msgs::TunePidResponse)

(defmethod tauv_msgs::TunePidRequest
  (:response () (instance tauv_msgs::TunePidResponse :init)))

(setf (get tauv_msgs::TunePidRequest :md5sum-) "c7f7e64a8df4cb6364bb9c28032e070e")
(setf (get tauv_msgs::TunePidRequest :datatype-) "tauv_msgs/TunePidRequest")
(setf (get tauv_msgs::TunePidRequest :definition-)
      "tauv_msgs/PidVals vals

================================================================================
MSG: tauv_msgs/PidVals
float32 a_p
float32 a_i
float32 a_d
float32 a_sat
float32 l_p
float32 l_i
float32 l_d
---
bool success
")

(setf (get tauv_msgs::TunePidResponse :md5sum-) "c7f7e64a8df4cb6364bb9c28032e070e")
(setf (get tauv_msgs::TunePidResponse :datatype-) "tauv_msgs/TunePidResponse")
(setf (get tauv_msgs::TunePidResponse :definition-)
      "tauv_msgs/PidVals vals

================================================================================
MSG: tauv_msgs/PidVals
float32 a_p
float32 a_i
float32 a_d
float32 a_sat
float32 l_p
float32 l_i
float32 l_d
---
bool success
")



(provide :tauv_msgs/TunePid "c7f7e64a8df4cb6364bb9c28032e070e")


