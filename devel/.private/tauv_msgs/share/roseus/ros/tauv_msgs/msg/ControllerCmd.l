;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::ControllerCmd)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'ControllerCmd (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::CONTROLLERCMD")
  (make-package "TAUV_MSGS::CONTROLLERCMD"))

(in-package "ROS")
;;//! \htmlinclude ControllerCmd.msg.html


(defclass tauv_msgs::ControllerCmd
  :super ros::object
  :slots (_a_x _a_y _a_z _a_yaw _p_roll _p_pitch ))

(defmethod tauv_msgs::ControllerCmd
  (:init
   (&key
    ((:a_x __a_x) 0.0)
    ((:a_y __a_y) 0.0)
    ((:a_z __a_z) 0.0)
    ((:a_yaw __a_yaw) 0.0)
    ((:p_roll __p_roll) 0.0)
    ((:p_pitch __p_pitch) 0.0)
    )
   (send-super :init)
   (setq _a_x (float __a_x))
   (setq _a_y (float __a_y))
   (setq _a_z (float __a_z))
   (setq _a_yaw (float __a_yaw))
   (setq _p_roll (float __p_roll))
   (setq _p_pitch (float __p_pitch))
   self)
  (:a_x
   (&optional __a_x)
   (if __a_x (setq _a_x __a_x)) _a_x)
  (:a_y
   (&optional __a_y)
   (if __a_y (setq _a_y __a_y)) _a_y)
  (:a_z
   (&optional __a_z)
   (if __a_z (setq _a_z __a_z)) _a_z)
  (:a_yaw
   (&optional __a_yaw)
   (if __a_yaw (setq _a_yaw __a_yaw)) _a_yaw)
  (:p_roll
   (&optional __p_roll)
   (if __p_roll (setq _p_roll __p_roll)) _p_roll)
  (:p_pitch
   (&optional __p_pitch)
   (if __p_pitch (setq _p_pitch __p_pitch)) _p_pitch)
  (:serialization-length
   ()
   (+
    ;; float32 _a_x
    4
    ;; float32 _a_y
    4
    ;; float32 _a_z
    4
    ;; float32 _a_yaw
    4
    ;; float32 _p_roll
    4
    ;; float32 _p_pitch
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _a_x
       (sys::poke _a_x (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _a_y
       (sys::poke _a_y (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _a_z
       (sys::poke _a_z (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _a_yaw
       (sys::poke _a_yaw (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _p_roll
       (sys::poke _p_roll (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _p_pitch
       (sys::poke _p_pitch (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _a_x
     (setq _a_x (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _a_y
     (setq _a_y (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _a_z
     (setq _a_z (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _a_yaw
     (setq _a_yaw (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _p_roll
     (setq _p_roll (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _p_pitch
     (setq _p_pitch (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get tauv_msgs::ControllerCmd :md5sum-) "c0612f34c73db057150c241e2726f1e9")
(setf (get tauv_msgs::ControllerCmd :datatype-) "tauv_msgs/ControllerCmd")
(setf (get tauv_msgs::ControllerCmd :definition-)
      "float32 a_x  # Acceleration in fixed frame x direction
float32 a_y  # Acceleration in fixed frame y direction
float32 a_z  # Acceleration in fixed frame z direction
float32 a_yaw  # Acceleration in fixed frame yaw direction
float32 p_roll  # roll target
float32 p_pitch  # pitch target
")



(provide :tauv_msgs/ControllerCmd "c0612f34c73db057150c241e2726f1e9")


