;; Auto-generated. Do not edit!


(when (boundp 'tauv_common::GetThrusterManagerConfig)
  (if (not (find-package "TAUV_COMMON"))
    (make-package "TAUV_COMMON"))
  (shadow 'GetThrusterManagerConfig (find-package "TAUV_COMMON")))
(unless (find-package "TAUV_COMMON::GETTHRUSTERMANAGERCONFIG")
  (make-package "TAUV_COMMON::GETTHRUSTERMANAGERCONFIG"))
(unless (find-package "TAUV_COMMON::GETTHRUSTERMANAGERCONFIGREQUEST")
  (make-package "TAUV_COMMON::GETTHRUSTERMANAGERCONFIGREQUEST"))
(unless (find-package "TAUV_COMMON::GETTHRUSTERMANAGERCONFIGRESPONSE")
  (make-package "TAUV_COMMON::GETTHRUSTERMANAGERCONFIGRESPONSE"))

(in-package "ROS")





(defclass tauv_common::GetThrusterManagerConfigRequest
  :super ros::object
  :slots ())

(defmethod tauv_common::GetThrusterManagerConfigRequest
  (:init
   (&key
    )
   (send-super :init)
   self)
  (:serialization-length
   ()
   (+
    0
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;;
   self)
  )

(defclass tauv_common::GetThrusterManagerConfigResponse
  :super ros::object
  :slots (_tf_prefix _base_link _thruster_frame_base _thruster_topic_prefix _thruster_topic_suffix _timeout _max_thrust _n_thrusters _allocation_matrix ))

(defmethod tauv_common::GetThrusterManagerConfigResponse
  (:init
   (&key
    ((:tf_prefix __tf_prefix) "")
    ((:base_link __base_link) "")
    ((:thruster_frame_base __thruster_frame_base) "")
    ((:thruster_topic_prefix __thruster_topic_prefix) "")
    ((:thruster_topic_suffix __thruster_topic_suffix) "")
    ((:timeout __timeout) 0.0)
    ((:max_thrust __max_thrust) 0.0)
    ((:n_thrusters __n_thrusters) 0)
    ((:allocation_matrix __allocation_matrix) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _tf_prefix (string __tf_prefix))
   (setq _base_link (string __base_link))
   (setq _thruster_frame_base (string __thruster_frame_base))
   (setq _thruster_topic_prefix (string __thruster_topic_prefix))
   (setq _thruster_topic_suffix (string __thruster_topic_suffix))
   (setq _timeout (float __timeout))
   (setq _max_thrust (float __max_thrust))
   (setq _n_thrusters (round __n_thrusters))
   (setq _allocation_matrix __allocation_matrix)
   self)
  (:tf_prefix
   (&optional __tf_prefix)
   (if __tf_prefix (setq _tf_prefix __tf_prefix)) _tf_prefix)
  (:base_link
   (&optional __base_link)
   (if __base_link (setq _base_link __base_link)) _base_link)
  (:thruster_frame_base
   (&optional __thruster_frame_base)
   (if __thruster_frame_base (setq _thruster_frame_base __thruster_frame_base)) _thruster_frame_base)
  (:thruster_topic_prefix
   (&optional __thruster_topic_prefix)
   (if __thruster_topic_prefix (setq _thruster_topic_prefix __thruster_topic_prefix)) _thruster_topic_prefix)
  (:thruster_topic_suffix
   (&optional __thruster_topic_suffix)
   (if __thruster_topic_suffix (setq _thruster_topic_suffix __thruster_topic_suffix)) _thruster_topic_suffix)
  (:timeout
   (&optional __timeout)
   (if __timeout (setq _timeout __timeout)) _timeout)
  (:max_thrust
   (&optional __max_thrust)
   (if __max_thrust (setq _max_thrust __max_thrust)) _max_thrust)
  (:n_thrusters
   (&optional __n_thrusters)
   (if __n_thrusters (setq _n_thrusters __n_thrusters)) _n_thrusters)
  (:allocation_matrix
   (&optional __allocation_matrix)
   (if __allocation_matrix (setq _allocation_matrix __allocation_matrix)) _allocation_matrix)
  (:serialization-length
   ()
   (+
    ;; string _tf_prefix
    4 (length _tf_prefix)
    ;; string _base_link
    4 (length _base_link)
    ;; string _thruster_frame_base
    4 (length _thruster_frame_base)
    ;; string _thruster_topic_prefix
    4 (length _thruster_topic_prefix)
    ;; string _thruster_topic_suffix
    4 (length _thruster_topic_suffix)
    ;; float64 _timeout
    8
    ;; float64 _max_thrust
    8
    ;; int32 _n_thrusters
    4
    ;; float64[] _allocation_matrix
    (* 8    (length _allocation_matrix)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _tf_prefix
       (write-long (length _tf_prefix) s) (princ _tf_prefix s)
     ;; string _base_link
       (write-long (length _base_link) s) (princ _base_link s)
     ;; string _thruster_frame_base
       (write-long (length _thruster_frame_base) s) (princ _thruster_frame_base s)
     ;; string _thruster_topic_prefix
       (write-long (length _thruster_topic_prefix) s) (princ _thruster_topic_prefix s)
     ;; string _thruster_topic_suffix
       (write-long (length _thruster_topic_suffix) s) (princ _thruster_topic_suffix s)
     ;; float64 _timeout
       (sys::poke _timeout (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _max_thrust
       (sys::poke _max_thrust (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; int32 _n_thrusters
       (write-long _n_thrusters s)
     ;; float64[] _allocation_matrix
     (write-long (length _allocation_matrix) s)
     (dotimes (i (length _allocation_matrix))
       (sys::poke (elt _allocation_matrix i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _tf_prefix
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _tf_prefix (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string _base_link
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _base_link (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string _thruster_frame_base
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _thruster_frame_base (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string _thruster_topic_prefix
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _thruster_topic_prefix (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string _thruster_topic_suffix
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _thruster_topic_suffix (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; float64 _timeout
     (setq _timeout (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _max_thrust
     (setq _max_thrust (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; int32 _n_thrusters
     (setq _n_thrusters (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; float64[] _allocation_matrix
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _allocation_matrix (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _allocation_matrix i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;;
   self)
  )

(defclass tauv_common::GetThrusterManagerConfig
  :super ros::object
  :slots ())

(setf (get tauv_common::GetThrusterManagerConfig :md5sum-) "b5a2d9d3bb510dd91fdb03f95e95b8de")
(setf (get tauv_common::GetThrusterManagerConfig :datatype-) "tauv_common/GetThrusterManagerConfig")
(setf (get tauv_common::GetThrusterManagerConfig :request) tauv_common::GetThrusterManagerConfigRequest)
(setf (get tauv_common::GetThrusterManagerConfig :response) tauv_common::GetThrusterManagerConfigResponse)

(defmethod tauv_common::GetThrusterManagerConfigRequest
  (:response () (instance tauv_common::GetThrusterManagerConfigResponse :init)))

(setf (get tauv_common::GetThrusterManagerConfigRequest :md5sum-) "b5a2d9d3bb510dd91fdb03f95e95b8de")
(setf (get tauv_common::GetThrusterManagerConfigRequest :datatype-) "tauv_common/GetThrusterManagerConfigRequest")
(setf (get tauv_common::GetThrusterManagerConfigRequest :definition-)
      "# Copyright (c) 2016 The UUV Simulator Authors.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

---
string tf_prefix
string base_link
string thruster_frame_base
string thruster_topic_prefix
string thruster_topic_suffix
float64 timeout
float64 max_thrust
int32 n_thrusters
float64[] allocation_matrix

")

(setf (get tauv_common::GetThrusterManagerConfigResponse :md5sum-) "b5a2d9d3bb510dd91fdb03f95e95b8de")
(setf (get tauv_common::GetThrusterManagerConfigResponse :datatype-) "tauv_common/GetThrusterManagerConfigResponse")
(setf (get tauv_common::GetThrusterManagerConfigResponse :definition-)
      "# Copyright (c) 2016 The UUV Simulator Authors.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

---
string tf_prefix
string base_link
string thruster_frame_base
string thruster_topic_prefix
string thruster_topic_suffix
float64 timeout
float64 max_thrust
int32 n_thrusters
float64[] allocation_matrix

")



(provide :tauv_common/GetThrusterManagerConfig "b5a2d9d3bb510dd91fdb03f95e95b8de")


