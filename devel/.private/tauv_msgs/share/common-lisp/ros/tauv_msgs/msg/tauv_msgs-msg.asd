
(cl:in-package :asdf)

(defsystem "tauv_msgs-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :geometry_msgs-msg
               :jsk_recognition_msgs-msg
               :sensor_msgs-msg
               :std_msgs-msg
               :vision_msgs-msg
)
  :components ((:file "_package")
    (:file "BucketDetection" :depends-on ("_package_BucketDetection"))
    (:file "_package_BucketDetection" :depends-on ("_package"))
    (:file "BucketList" :depends-on ("_package_BucketList"))
    (:file "_package_BucketList" :depends-on ("_package"))
    (:file "ControllerCmd" :depends-on ("_package_ControllerCmd"))
    (:file "_package_ControllerCmd" :depends-on ("_package"))
    (:file "FluidDepth" :depends-on ("_package_FluidDepth"))
    (:file "_package_FluidDepth" :depends-on ("_package"))
    (:file "InertialVals" :depends-on ("_package_InertialVals"))
    (:file "_package_InertialVals" :depends-on ("_package"))
    (:file "MpcRefTraj" :depends-on ("_package_MpcRefTraj"))
    (:file "_package_MpcRefTraj" :depends-on ("_package"))
    (:file "PidVals" :depends-on ("_package_PidVals"))
    (:file "_package_PidVals" :depends-on ("_package"))
    (:file "PoseGraphMeasurement" :depends-on ("_package_PoseGraphMeasurement"))
    (:file "_package_PoseGraphMeasurement" :depends-on ("_package"))
    (:file "SonarPulse" :depends-on ("_package_SonarPulse"))
    (:file "_package_SonarPulse" :depends-on ("_package"))
  ))