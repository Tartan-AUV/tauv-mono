;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::SonarPulse)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'SonarPulse (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::SONARPULSE")
  (make-package "TAUV_MSGS::SONARPULSE"))

(in-package "ROS")
;;//! \htmlinclude SonarPulse.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass tauv_msgs::SonarPulse
  :super ros::object
  :slots (_header _mode _gain_setting _angle _transmit_duration _sample_period _transmit_frequency _number_of_samples _data_length _data ))

(defmethod tauv_msgs::SonarPulse
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:mode __mode) 0)
    ((:gain_setting __gain_setting) 0)
    ((:angle __angle) 0.0)
    ((:transmit_duration __transmit_duration) 0.0)
    ((:sample_period __sample_period) 0.0)
    ((:transmit_frequency __transmit_frequency) 0.0)
    ((:number_of_samples __number_of_samples) 0)
    ((:data_length __data_length) 0)
    ((:data __data) (make-array 0 :initial-element 0 :element-type :char))
    )
   (send-super :init)
   (setq _header __header)
   (setq _mode (round __mode))
   (setq _gain_setting (round __gain_setting))
   (setq _angle (float __angle))
   (setq _transmit_duration (float __transmit_duration))
   (setq _sample_period (float __sample_period))
   (setq _transmit_frequency (float __transmit_frequency))
   (setq _number_of_samples (round __number_of_samples))
   (setq _data_length (round __data_length))
   (setq _data __data)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:mode
   (&optional __mode)
   (if __mode (setq _mode __mode)) _mode)
  (:gain_setting
   (&optional __gain_setting)
   (if __gain_setting (setq _gain_setting __gain_setting)) _gain_setting)
  (:angle
   (&optional __angle)
   (if __angle (setq _angle __angle)) _angle)
  (:transmit_duration
   (&optional __transmit_duration)
   (if __transmit_duration (setq _transmit_duration __transmit_duration)) _transmit_duration)
  (:sample_period
   (&optional __sample_period)
   (if __sample_period (setq _sample_period __sample_period)) _sample_period)
  (:transmit_frequency
   (&optional __transmit_frequency)
   (if __transmit_frequency (setq _transmit_frequency __transmit_frequency)) _transmit_frequency)
  (:number_of_samples
   (&optional __number_of_samples)
   (if __number_of_samples (setq _number_of_samples __number_of_samples)) _number_of_samples)
  (:data_length
   (&optional __data_length)
   (if __data_length (setq _data_length __data_length)) _data_length)
  (:data
   (&optional __data)
   (if __data (setq _data __data)) _data)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; uint8 _mode
    1
    ;; uint8 _gain_setting
    1
    ;; float32 _angle
    4
    ;; float32 _transmit_duration
    4
    ;; float32 _sample_period
    4
    ;; float32 _transmit_frequency
    4
    ;; uint16 _number_of_samples
    2
    ;; uint16 _data_length
    2
    ;; uint8[] _data
    (* 1    (length _data)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; uint8 _mode
       (write-byte _mode s)
     ;; uint8 _gain_setting
       (write-byte _gain_setting s)
     ;; float32 _angle
       (sys::poke _angle (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _transmit_duration
       (sys::poke _transmit_duration (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _sample_period
       (sys::poke _sample_period (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _transmit_frequency
       (sys::poke _transmit_frequency (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; uint16 _number_of_samples
       (write-word _number_of_samples s)
     ;; uint16 _data_length
       (write-word _data_length s)
     ;; uint8[] _data
     (write-long (length _data) s)
     (princ _data s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; uint8 _mode
     (setq _mode (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint8 _gain_setting
     (setq _gain_setting (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; float32 _angle
     (setq _angle (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _transmit_duration
     (setq _transmit_duration (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _sample_period
     (setq _sample_period (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _transmit_frequency
     (setq _transmit_frequency (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; uint16 _number_of_samples
     (setq _number_of_samples (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; uint16 _data_length
     (setq _data_length (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; uint8[] _data
   (let ((n (sys::peek buf ptr- :integer))) (incf ptr- 4)
     (setq _data (make-array n :element-type :char))
     (replace _data buf :start2 ptr-) (incf ptr- n))
   ;;
   self)
  )

(setf (get tauv_msgs::SonarPulse :md5sum-) "5a4534993e7634b0f0c3a4e8eec771be")
(setf (get tauv_msgs::SonarPulse :datatype-) "tauv_msgs/SonarPulse")
(setf (get tauv_msgs::SonarPulse :definition-)
      "Header header

uint8 mode
uint8 gain_setting
float32 angle
float32 transmit_duration
float32 sample_period
float32 transmit_frequency
uint16 number_of_samples

uint16 data_length
uint8[] data
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

")



(provide :tauv_msgs/SonarPulse "5a4534993e7634b0f0c3a4e8eec771be")


