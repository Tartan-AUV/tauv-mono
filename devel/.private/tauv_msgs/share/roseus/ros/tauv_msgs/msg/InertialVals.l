;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::InertialVals)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'InertialVals (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::INERTIALVALS")
  (make-package "TAUV_MSGS::INERTIALVALS"))

(in-package "ROS")
;;//! \htmlinclude InertialVals.msg.html


(defclass tauv_msgs::InertialVals
  :super ros::object
  :slots (_mass _buoyancy _ixx _iyy _izz ))

(defmethod tauv_msgs::InertialVals
  (:init
   (&key
    ((:mass __mass) 0.0)
    ((:buoyancy __buoyancy) 0.0)
    ((:ixx __ixx) 0.0)
    ((:iyy __iyy) 0.0)
    ((:izz __izz) 0.0)
    )
   (send-super :init)
   (setq _mass (float __mass))
   (setq _buoyancy (float __buoyancy))
   (setq _ixx (float __ixx))
   (setq _iyy (float __iyy))
   (setq _izz (float __izz))
   self)
  (:mass
   (&optional __mass)
   (if __mass (setq _mass __mass)) _mass)
  (:buoyancy
   (&optional __buoyancy)
   (if __buoyancy (setq _buoyancy __buoyancy)) _buoyancy)
  (:ixx
   (&optional __ixx)
   (if __ixx (setq _ixx __ixx)) _ixx)
  (:iyy
   (&optional __iyy)
   (if __iyy (setq _iyy __iyy)) _iyy)
  (:izz
   (&optional __izz)
   (if __izz (setq _izz __izz)) _izz)
  (:serialization-length
   ()
   (+
    ;; float32 _mass
    4
    ;; float32 _buoyancy
    4
    ;; float32 _ixx
    4
    ;; float32 _iyy
    4
    ;; float32 _izz
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _mass
       (sys::poke _mass (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _buoyancy
       (sys::poke _buoyancy (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _ixx
       (sys::poke _ixx (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _iyy
       (sys::poke _iyy (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _izz
       (sys::poke _izz (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _mass
     (setq _mass (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _buoyancy
     (setq _buoyancy (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _ixx
     (setq _ixx (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _iyy
     (setq _iyy (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _izz
     (setq _izz (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get tauv_msgs::InertialVals :md5sum-) "dc905a6a26bfe30465ae55cdfc3db94e")
(setf (get tauv_msgs::InertialVals :datatype-) "tauv_msgs/InertialVals")
(setf (get tauv_msgs::InertialVals :definition-)
      "float32 mass
float32 buoyancy
float32 ixx
float32 iyy
float32 izz
")



(provide :tauv_msgs/InertialVals "dc905a6a26bfe30465ae55cdfc3db94e")


