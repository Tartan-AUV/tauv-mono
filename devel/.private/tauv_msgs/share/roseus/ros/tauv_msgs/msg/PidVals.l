;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::PidVals)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'PidVals (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::PIDVALS")
  (make-package "TAUV_MSGS::PIDVALS"))

(in-package "ROS")
;;//! \htmlinclude PidVals.msg.html


(defclass tauv_msgs::PidVals
  :super ros::object
  :slots (_a_p _a_i _a_d _a_sat _l_p _l_i _l_d _l_sat ))

(defmethod tauv_msgs::PidVals
  (:init
   (&key
    ((:a_p __a_p) 0.0)
    ((:a_i __a_i) 0.0)
    ((:a_d __a_d) 0.0)
    ((:a_sat __a_sat) 0.0)
    ((:l_p __l_p) 0.0)
    ((:l_i __l_i) 0.0)
    ((:l_d __l_d) 0.0)
    ((:l_sat __l_sat) 0.0)
    )
   (send-super :init)
   (setq _a_p (float __a_p))
   (setq _a_i (float __a_i))
   (setq _a_d (float __a_d))
   (setq _a_sat (float __a_sat))
   (setq _l_p (float __l_p))
   (setq _l_i (float __l_i))
   (setq _l_d (float __l_d))
   (setq _l_sat (float __l_sat))
   self)
  (:a_p
   (&optional __a_p)
   (if __a_p (setq _a_p __a_p)) _a_p)
  (:a_i
   (&optional __a_i)
   (if __a_i (setq _a_i __a_i)) _a_i)
  (:a_d
   (&optional __a_d)
   (if __a_d (setq _a_d __a_d)) _a_d)
  (:a_sat
   (&optional __a_sat)
   (if __a_sat (setq _a_sat __a_sat)) _a_sat)
  (:l_p
   (&optional __l_p)
   (if __l_p (setq _l_p __l_p)) _l_p)
  (:l_i
   (&optional __l_i)
   (if __l_i (setq _l_i __l_i)) _l_i)
  (:l_d
   (&optional __l_d)
   (if __l_d (setq _l_d __l_d)) _l_d)
  (:l_sat
   (&optional __l_sat)
   (if __l_sat (setq _l_sat __l_sat)) _l_sat)
  (:serialization-length
   ()
   (+
    ;; float32 _a_p
    4
    ;; float32 _a_i
    4
    ;; float32 _a_d
    4
    ;; float32 _a_sat
    4
    ;; float32 _l_p
    4
    ;; float32 _l_i
    4
    ;; float32 _l_d
    4
    ;; float32 _l_sat
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _a_p
       (sys::poke _a_p (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _a_i
       (sys::poke _a_i (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _a_d
       (sys::poke _a_d (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _a_sat
       (sys::poke _a_sat (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _l_p
       (sys::poke _l_p (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _l_i
       (sys::poke _l_i (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _l_d
       (sys::poke _l_d (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _l_sat
       (sys::poke _l_sat (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _a_p
     (setq _a_p (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _a_i
     (setq _a_i (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _a_d
     (setq _a_d (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _a_sat
     (setq _a_sat (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _l_p
     (setq _l_p (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _l_i
     (setq _l_i (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _l_d
     (setq _l_d (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _l_sat
     (setq _l_sat (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get tauv_msgs::PidVals :md5sum-) "d4db47770a0caf47edbb925bd3a9269a")
(setf (get tauv_msgs::PidVals :datatype-) "tauv_msgs/PidVals")
(setf (get tauv_msgs::PidVals :definition-)
      "float32 a_p
float32 a_i
float32 a_d
float32 a_sat
float32 l_p
float32 l_i
float32 l_d
float32 l_sat
")



(provide :tauv_msgs/PidVals "d4db47770a0caf47edbb925bd3a9269a")


