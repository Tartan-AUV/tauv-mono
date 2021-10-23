;; Auto-generated. Do not edit!


(when (boundp 'tauv_common::ThrusterManagerInfo)
  (if (not (find-package "TAUV_COMMON"))
    (make-package "TAUV_COMMON"))
  (shadow 'ThrusterManagerInfo (find-package "TAUV_COMMON")))
(unless (find-package "TAUV_COMMON::THRUSTERMANAGERINFO")
  (make-package "TAUV_COMMON::THRUSTERMANAGERINFO"))
(unless (find-package "TAUV_COMMON::THRUSTERMANAGERINFOREQUEST")
  (make-package "TAUV_COMMON::THRUSTERMANAGERINFOREQUEST"))
(unless (find-package "TAUV_COMMON::THRUSTERMANAGERINFORESPONSE")
  (make-package "TAUV_COMMON::THRUSTERMANAGERINFORESPONSE"))

(in-package "ROS")





(defclass tauv_common::ThrusterManagerInfoRequest
  :super ros::object
  :slots ())

(defmethod tauv_common::ThrusterManagerInfoRequest
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

(defclass tauv_common::ThrusterManagerInfoResponse
  :super ros::object
  :slots (_n_thrusters _allocation_matrix _reference_frame ))

(defmethod tauv_common::ThrusterManagerInfoResponse
  (:init
   (&key
    ((:n_thrusters __n_thrusters) 0)
    ((:allocation_matrix __allocation_matrix) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:reference_frame __reference_frame) "")
    )
   (send-super :init)
   (setq _n_thrusters (round __n_thrusters))
   (setq _allocation_matrix __allocation_matrix)
   (setq _reference_frame (string __reference_frame))
   self)
  (:n_thrusters
   (&optional __n_thrusters)
   (if __n_thrusters (setq _n_thrusters __n_thrusters)) _n_thrusters)
  (:allocation_matrix
   (&optional __allocation_matrix)
   (if __allocation_matrix (setq _allocation_matrix __allocation_matrix)) _allocation_matrix)
  (:reference_frame
   (&optional __reference_frame)
   (if __reference_frame (setq _reference_frame __reference_frame)) _reference_frame)
  (:serialization-length
   ()
   (+
    ;; int32 _n_thrusters
    4
    ;; float64[] _allocation_matrix
    (* 8    (length _allocation_matrix)) 4
    ;; string _reference_frame
    4 (length _reference_frame)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int32 _n_thrusters
       (write-long _n_thrusters s)
     ;; float64[] _allocation_matrix
     (write-long (length _allocation_matrix) s)
     (dotimes (i (length _allocation_matrix))
       (sys::poke (elt _allocation_matrix i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; string _reference_frame
       (write-long (length _reference_frame) s) (princ _reference_frame s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int32 _n_thrusters
     (setq _n_thrusters (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; float64[] _allocation_matrix
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _allocation_matrix (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _allocation_matrix i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;; string _reference_frame
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _reference_frame (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass tauv_common::ThrusterManagerInfo
  :super ros::object
  :slots ())

(setf (get tauv_common::ThrusterManagerInfo :md5sum-) "66fb8ab2f9c5649d97263c955edb636e")
(setf (get tauv_common::ThrusterManagerInfo :datatype-) "tauv_common/ThrusterManagerInfo")
(setf (get tauv_common::ThrusterManagerInfo :request) tauv_common::ThrusterManagerInfoRequest)
(setf (get tauv_common::ThrusterManagerInfo :response) tauv_common::ThrusterManagerInfoResponse)

(defmethod tauv_common::ThrusterManagerInfoRequest
  (:response () (instance tauv_common::ThrusterManagerInfoResponse :init)))

(setf (get tauv_common::ThrusterManagerInfoRequest :md5sum-) "66fb8ab2f9c5649d97263c955edb636e")
(setf (get tauv_common::ThrusterManagerInfoRequest :datatype-) "tauv_common/ThrusterManagerInfoRequest")
(setf (get tauv_common::ThrusterManagerInfoRequest :definition-)
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
int32 n_thrusters
float64[] allocation_matrix
string reference_frame

")

(setf (get tauv_common::ThrusterManagerInfoResponse :md5sum-) "66fb8ab2f9c5649d97263c955edb636e")
(setf (get tauv_common::ThrusterManagerInfoResponse :datatype-) "tauv_common/ThrusterManagerInfoResponse")
(setf (get tauv_common::ThrusterManagerInfoResponse :definition-)
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
int32 n_thrusters
float64[] allocation_matrix
string reference_frame

")



(provide :tauv_common/ThrusterManagerInfo "66fb8ab2f9c5649d97263c955edb636e")


