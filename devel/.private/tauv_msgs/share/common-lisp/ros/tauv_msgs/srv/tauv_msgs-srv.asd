
(cl:in-package :asdf)

(defsystem "tauv_msgs-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :geometry_msgs-msg
               :std_msgs-msg
               :tauv_msgs-msg
)
  :components ((:file "_package")
    (:file "GetTraj" :depends-on ("_package_GetTraj"))
    (:file "_package_GetTraj" :depends-on ("_package"))
    (:file "TuneInertial" :depends-on ("_package_TuneInertial"))
    (:file "_package_TuneInertial" :depends-on ("_package"))
    (:file "TunePid" :depends-on ("_package_TunePid"))
    (:file "_package_TunePid" :depends-on ("_package"))
  ))