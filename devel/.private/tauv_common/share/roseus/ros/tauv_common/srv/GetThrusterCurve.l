;; Auto-generated. Do not edit!


(when (boundp 'tauv_common::GetThrusterCurve)
  (if (not (find-package "TAUV_COMMON"))
    (make-package "TAUV_COMMON"))
  (shadow 'GetThrusterCurve (find-package "TAUV_COMMON")))
(unless (find-package "TAUV_COMMON::GETTHRUSTERCURVE")
  (make-package "TAUV_COMMON::GETTHRUSTERCURVE"))
(unless (find-package "TAUV_COMMON::GETTHRUSTERCURVEREQUEST")
  (make-package "TAUV_COMMON::GETTHRUSTERCURVEREQUEST"))
(unless (find-package "TAUV_COMMON::GETTHRUSTERCURVERESPONSE")
  (make-package "TAUV_COMMON::GETTHRUSTERCURVERESPONSE"))

(in-package "ROS")





(defclass tauv_common::GetThrusterCurveRequest
  :super ros::object
  :slots (_min _max _n_points ))

(defmethod tauv_common::GetThrusterCurveRequest
  (:init
   (&key
    ((:min __min) 0.0)
    ((:max __max) 0.0)
    ((:n_points __n_points) 0)
    )
   (send-super :init)
   (setq _min (float __min))
   (setq _max (float __max))
   (setq _n_points (round __n_points))
   self)
  (:min
   (&optional __min)
   (if __min (setq _min __min)) _min)
  (:max
   (&optional __max)
   (if __max (setq _max __max)) _max)
  (:n_points
   (&optional __n_points)
   (if __n_points (setq _n_points __n_points)) _n_points)
  (:serialization-length
   ()
   (+
    ;; float64 _min
    8
    ;; float64 _max
    8
    ;; int32 _n_points
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float64 _min
       (sys::poke _min (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _max
       (sys::poke _max (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; int32 _n_points
       (write-long _n_points s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float64 _min
     (setq _min (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _max
     (setq _max (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; int32 _n_points
     (setq _n_points (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;;
   self)
  )

(defclass tauv_common::GetThrusterCurveResponse
  :super ros::object
  :slots (_input _thrust ))

(defmethod tauv_common::GetThrusterCurveResponse
  (:init
   (&key
    ((:input __input) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:thrust __thrust) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _input __input)
   (setq _thrust __thrust)
   self)
  (:input
   (&optional __input)
   (if __input (setq _input __input)) _input)
  (:thrust
   (&optional __thrust)
   (if __thrust (setq _thrust __thrust)) _thrust)
  (:serialization-length
   ()
   (+
    ;; float64[] _input
    (* 8    (length _input)) 4
    ;; float64[] _thrust
    (* 8    (length _thrust)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float64[] _input
     (write-long (length _input) s)
     (dotimes (i (length _input))
       (sys::poke (elt _input i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64[] _thrust
     (write-long (length _thrust) s)
     (dotimes (i (length _thrust))
       (sys::poke (elt _thrust i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float64[] _input
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _input (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _input i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;; float64[] _thrust
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _thrust (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _thrust i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;;
   self)
  )

(defclass tauv_common::GetThrusterCurve
  :super ros::object
  :slots ())

(setf (get tauv_common::GetThrusterCurve :md5sum-) "93e0d8a78977b3619bdc09f290ef57fa")
(setf (get tauv_common::GetThrusterCurve :datatype-) "tauv_common/GetThrusterCurve")
(setf (get tauv_common::GetThrusterCurve :request) tauv_common::GetThrusterCurveRequest)
(setf (get tauv_common::GetThrusterCurve :response) tauv_common::GetThrusterCurveResponse)

(defmethod tauv_common::GetThrusterCurveRequest
  (:response () (instance tauv_common::GetThrusterCurveResponse :init)))

(setf (get tauv_common::GetThrusterCurveRequest :md5sum-) "93e0d8a78977b3619bdc09f290ef57fa")
(setf (get tauv_common::GetThrusterCurveRequest :datatype-) "tauv_common/GetThrusterCurveRequest")
(setf (get tauv_common::GetThrusterCurveRequest :definition-)
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

float64 min
float64 max
int32 n_points
---
float64[] input
float64[] thrust

")

(setf (get tauv_common::GetThrusterCurveResponse :md5sum-) "93e0d8a78977b3619bdc09f290ef57fa")
(setf (get tauv_common::GetThrusterCurveResponse :datatype-) "tauv_common/GetThrusterCurveResponse")
(setf (get tauv_common::GetThrusterCurveResponse :definition-)
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

float64 min
float64 max
int32 n_points
---
float64[] input
float64[] thrust

")



(provide :tauv_common/GetThrusterCurve "93e0d8a78977b3619bdc09f290ef57fa")


