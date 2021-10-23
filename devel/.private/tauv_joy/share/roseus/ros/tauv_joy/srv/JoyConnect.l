;; Auto-generated. Do not edit!


(when (boundp 'tauv_joy::JoyConnect)
  (if (not (find-package "TAUV_JOY"))
    (make-package "TAUV_JOY"))
  (shadow 'JoyConnect (find-package "TAUV_JOY")))
(unless (find-package "TAUV_JOY::JOYCONNECT")
  (make-package "TAUV_JOY::JOYCONNECT"))
(unless (find-package "TAUV_JOY::JOYCONNECTREQUEST")
  (make-package "TAUV_JOY::JOYCONNECTREQUEST"))
(unless (find-package "TAUV_JOY::JOYCONNECTRESPONSE")
  (make-package "TAUV_JOY::JOYCONNECTRESPONSE"))

(in-package "ROS")





(defclass tauv_joy::JoyConnectRequest
  :super ros::object
  :slots (_dev ))

(defmethod tauv_joy::JoyConnectRequest
  (:init
   (&key
    ((:dev __dev) "")
    )
   (send-super :init)
   (setq _dev (string __dev))
   self)
  (:dev
   (&optional __dev)
   (if __dev (setq _dev __dev)) _dev)
  (:serialization-length
   ()
   (+
    ;; string _dev
    4 (length _dev)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _dev
       (write-long (length _dev) s) (princ _dev s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _dev
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _dev (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass tauv_joy::JoyConnectResponse
  :super ros::object
  :slots (_success _message ))

(defmethod tauv_joy::JoyConnectResponse
  (:init
   (&key
    ((:success __success) nil)
    ((:message __message) "")
    )
   (send-super :init)
   (setq _success __success)
   (setq _message (string __message))
   self)
  (:success
   (&optional (__success :null))
   (if (not (eq __success :null)) (setq _success __success)) _success)
  (:message
   (&optional __message)
   (if __message (setq _message __message)) _message)
  (:serialization-length
   ()
   (+
    ;; bool _success
    1
    ;; string _message
    4 (length _message)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _success
       (if _success (write-byte -1 s) (write-byte 0 s))
     ;; string _message
       (write-long (length _message) s) (princ _message s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _success
     (setq _success (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; string _message
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _message (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass tauv_joy::JoyConnect
  :super ros::object
  :slots ())

(setf (get tauv_joy::JoyConnect :md5sum-) "4ac0afce82f389644128ac3ab46cf672")
(setf (get tauv_joy::JoyConnect :datatype-) "tauv_joy/JoyConnect")
(setf (get tauv_joy::JoyConnect :request) tauv_joy::JoyConnectRequest)
(setf (get tauv_joy::JoyConnect :response) tauv_joy::JoyConnectResponse)

(defmethod tauv_joy::JoyConnectRequest
  (:response () (instance tauv_joy::JoyConnectResponse :init)))

(setf (get tauv_joy::JoyConnectRequest :md5sum-) "4ac0afce82f389644128ac3ab46cf672")
(setf (get tauv_joy::JoyConnectRequest :datatype-) "tauv_joy/JoyConnectRequest")
(setf (get tauv_joy::JoyConnectRequest :definition-)
      "string dev # device file path
---
bool success   # indicate successful run of triggered service
string message # informational, e.g. for error messages
")

(setf (get tauv_joy::JoyConnectResponse :md5sum-) "4ac0afce82f389644128ac3ab46cf672")
(setf (get tauv_joy::JoyConnectResponse :datatype-) "tauv_joy/JoyConnectResponse")
(setf (get tauv_joy::JoyConnectResponse :definition-)
      "string dev # device file path
---
bool success   # indicate successful run of triggered service
string message # informational, e.g. for error messages
")



(provide :tauv_joy/JoyConnect "4ac0afce82f389644128ac3ab46cf672")


