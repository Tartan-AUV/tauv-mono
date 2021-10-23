
(cl:in-package :asdf)

(defsystem "tauv_joy-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "JoyConnect" :depends-on ("_package_JoyConnect"))
    (:file "_package_JoyConnect" :depends-on ("_package"))
  ))