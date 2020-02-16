import os
import rospy
import rospkg

from qt_gui.plugin import Plugin
from python_qt_binding import loadUi
from python_qt_binding.QtWidgets import QWidget

from tauv_msgs.msg import InertialVals, PidVals
from tauv_msgs.srv import TuneInertial, TuneInertialResponse, TunePid, TunePidResponse

from enum import Enum
import json


class PIDValue(Enum):
    Pl = 0
    Il = 1
    Dl = 2
    Sl = 3
    Pa = 4
    Ia = 5
    Da = 6
    Sa = 7


class InertialValue(Enum):
    Mass = 0
    Buoyancy = 1
    Ixx = 2
    Iyy = 3
    Izz = 4


class Tuner(Enum):
    Pid1 = 0
    Pid2 = 1
    Inertial = 2


class PidTuner(Plugin):

    def __init__(self, context):
        super(PidTuner, self).__init__(context)
        # Give QObjects reasonable names
        self.setObjectName('PidTuner')

        # Process standalone plugin command-line arguments
        from argparse import ArgumentParser
        parser = ArgumentParser()
        # Add argument(s) to the parser.
        parser.add_argument("-q", "--quiet", action="store_true",
                            dest="quiet",
                            help="Put plugin in silent mode")
        args, unknowns = parser.parse_known_args(context.argv())
        if not args.quiet:
            print 'arguments: ', args
            print 'unknowns: ', unknowns

        # Create QWidget
        self._widget = QWidget()
        # Get path to UI file which should be in the "resource" folder of this package
        ui_file = os.path.join(rospkg.RosPack().get_path('rqt_pid_tuner'), 'resource', 'pid_tuner.ui')
        # Extend the widget with all attributes and children from UI file
        loadUi(ui_file, self._widget)
        # Give QObjects reasonable names
        self._widget.setObjectName('PidTunerUI')
        # Show _widget.windowTitle on left-top of each plugin (when
        # it's set in _widget). This is useful when you open multiple
        # plugins at once. Also if you open multiple instances of your
        # plugin at once, these lines add number to make it easy to
        # tell from pane to pane.

        self.vals = {
            Tuner.Pid1: {
                PIDValue.Pl: 0.0,
                PIDValue.Il: 0.0,
                PIDValue.Dl: 0.0,
                PIDValue.Sl: 0.0,
                PIDValue.Pa: 0.0,
                PIDValue.Ia: 0.0,
                PIDValue.Da: 0.0,
                PIDValue.Sa: 0.0,
                "service": "",
                "topic": ""
            },
            Tuner.Pid2: {
                PIDValue.Pl: 0.0,
                PIDValue.Il: 0.0,
                PIDValue.Dl: 0.0,
                PIDValue.Sl: 0.0,
                PIDValue.Pa: 0.0,
                PIDValue.Ia: 0.0,
                PIDValue.Da: 0.0,
                PIDValue.Sa: 0.0,
                "service": "",
                "topic": ""
            },
            Tuner.Inertial: {
                InertialValue.Mass: 0.0,
                InertialValue.Buoyancy: 0.0,
                InertialValue.Ixx: 0.0,
                InertialValue.Iyy: 0.0,
                InertialValue.Izz: 0.0,
                "service": "",
                "topic": ""
            }
        }

        self.cachedVals = {
            Tuner.Pid1: {
                PIDValue.Pl: 0.0,
                PIDValue.Il: 0.0,
                PIDValue.Dl: 0.0,
                PIDValue.Sl: 0.0,
                PIDValue.Pa: 0.0,
                PIDValue.Ia: 0.0,
                PIDValue.Da: 0.0,
                PIDValue.Sa: 0.0
            },
            Tuner.Pid2: {
                PIDValue.Pl: 0.0,
                PIDValue.Il: 0.0,
                PIDValue.Dl: 0.0,
                PIDValue.Sl: 0.0,
                PIDValue.Pa: 0.0,
                PIDValue.Ia: 0.0,
                PIDValue.Da: 0.0,
                PIDValue.Sa: 0.0
            },
            Tuner.Inertial: {
                InertialValue.Mass: 0.0,
                InertialValue.Buoyancy: 0.0,
                InertialValue.Ixx: 0.0,
                InertialValue.Iyy: 0.0,
                InertialValue.Izz: 0.0
            }
        }

        self.sub_t1 = None
        self.sub_t2 = None
        self.sub_t3 = None

        self._widget.t1_lp.textEdited.connect(lambda t: self.on_pid_val_changed(t, Tuner.Pid1, PIDValue.Pl))
        self._widget.t1_li.textEdited.connect(lambda t: self.on_pid_val_changed(t, Tuner.Pid1, PIDValue.Il))
        self._widget.t1_ld.textEdited.connect(lambda t: self.on_pid_val_changed(t, Tuner.Pid1, PIDValue.Dl))
        self._widget.t1_ls.textEdited.connect(lambda t: self.on_pid_val_changed(t, Tuner.Pid1, PIDValue.Sl))
        self._widget.t1_ap.textEdited.connect(lambda t: self.on_pid_val_changed(t, Tuner.Pid1, PIDValue.Pa))
        self._widget.t1_ai.textEdited.connect(lambda t: self.on_pid_val_changed(t, Tuner.Pid1, PIDValue.Ia))
        self._widget.t1_ad.textEdited.connect(lambda t: self.on_pid_val_changed(t, Tuner.Pid1, PIDValue.Da))
        self._widget.t1_as.textEdited.connect(lambda t: self.on_pid_val_changed(t, Tuner.Pid1, PIDValue.Sa))
        self._widget.t1_srv.editingFinished.connect(lambda: self.on_srv_name_changed(self._widget.t1_srv, Tuner.Pid1))
        self._widget.t1_tpc.editingFinished.connect(lambda: self.on_topic_name_changed(self._widget.t1_tpc, Tuner.Pid1))
        self._widget.t1_read.pressed.connect(lambda: self.on_reload_values(Tuner.Pid1))
        self._widget.t1_send.pressed.connect(lambda: self.on_send_values(Tuner.Pid1))

        self._widget.t2_lp.textEdited.connect(lambda t: self.on_pid_val_changed(t, Tuner.Pid2, PIDValue.Pl))
        self._widget.t2_li.textEdited.connect(lambda t: self.on_pid_val_changed(t, Tuner.Pid2, PIDValue.Il))
        self._widget.t2_ld.textEdited.connect(lambda t: self.on_pid_val_changed(t, Tuner.Pid2, PIDValue.Dl))
        self._widget.t2_ls.textEdited.connect(lambda t: self.on_pid_val_changed(t, Tuner.Pid2, PIDValue.Sl))
        self._widget.t2_ap.textEdited.connect(lambda t: self.on_pid_val_changed(t, Tuner.Pid2, PIDValue.Pa))
        self._widget.t2_ai.textEdited.connect(lambda t: self.on_pid_val_changed(t, Tuner.Pid2, PIDValue.Ia))
        self._widget.t2_ad.textEdited.connect(lambda t: self.on_pid_val_changed(t, Tuner.Pid2, PIDValue.Da))
        self._widget.t2_as.textEdited.connect(lambda t: self.on_pid_val_changed(t, Tuner.Pid2, PIDValue.Sa))
        self._widget.t2_srv.editingFinished.connect(lambda: self.on_srv_name_changed(self._widget.t2_srv, Tuner.Pid2))
        self._widget.t2_tpc.editingFinished.connect(lambda: self.on_topic_name_changed(self._widget.t2_tpc, Tuner.Pid2))
        self._widget.t2_read.pressed.connect(lambda: self.on_reload_values(Tuner.Pid2))
        self._widget.t2_send.pressed.connect(lambda: self.on_send_values(Tuner.Pid2))

        self._widget.t3_m.textEdited.connect(lambda t: self.on_pid_val_changed(t, Tuner.Inertial, InertialValue.Mass))
        self._widget.t3_b.textEdited.connect(
            lambda t: self.on_pid_val_changed(t, Tuner.Inertial, InertialValue.Buoyancy))
        self._widget.t3_ixx.textEdited.connect(lambda t: self.on_pid_val_changed(t, Tuner.Inertial, InertialValue.Ixx))
        self._widget.t3_iyy.textEdited.connect(lambda t: self.on_pid_val_changed(t, Tuner.Inertial, InertialValue.Iyy))
        self._widget.t3_izz.textEdited.connect(lambda t: self.on_pid_val_changed(t, Tuner.Inertial, InertialValue.Izz))
        self._widget.t3_srv.editingFinished.connect(
            lambda: self.on_srv_name_changed(self._widget.t3_srv, Tuner.Inertial))
        self._widget.t3_tpc.editingFinished.connect(
            lambda: self.on_topic_name_changed(self._widget.t3_tpc, Tuner.Inertial))
        self._widget.t3_read.pressed.connect(lambda: self.on_reload_values(Tuner.Inertial))
        self._widget.t3_send.pressed.connect(lambda: self.on_send_values(Tuner.Inertial))

        if context.serial_number() > 1:
            self._widget.setWindowTitle(self._widget.windowTitle() + (' (%d)' % context.serial_number()))
        # Add widget to the user interface
        context.add_widget(self._widget)

    def on_pid_val_changed(self, text, tuner, pidvaluetype):
        try:
            self.vals[tuner][pidvaluetype] = float(text)
        except ValueError as ve:
            pass
            self.update_text()

    def on_srv_name_changed(self, source, tuner):
        self.vals[tuner]["service"] = source.text()

    def on_topic_name_changed(self, source, tuner):
        self.vals[tuner]["topic"] = source.text()
        self.resub()

    def update_text(self):
        self._widget.t1_lp.setText(str(self.vals[Tuner.Pid1][PIDValue.Pl]))
        self._widget.t1_li.setText(str(self.vals[Tuner.Pid1][PIDValue.Il]))
        self._widget.t1_ld.setText(str(self.vals[Tuner.Pid1][PIDValue.Dl]))
        self._widget.t1_ls.setText(str(self.vals[Tuner.Pid1][PIDValue.Sl]))
        self._widget.t1_ap.setText(str(self.vals[Tuner.Pid1][PIDValue.Pa]))
        self._widget.t1_ai.setText(str(self.vals[Tuner.Pid1][PIDValue.Ia]))
        self._widget.t1_ad.setText(str(self.vals[Tuner.Pid1][PIDValue.Da]))
        self._widget.t1_as.setText(str(self.vals[Tuner.Pid1][PIDValue.Sa]))
        self._widget.t1_srv.setText(str(self.vals[Tuner.Pid1]["service"]))
        self._widget.t1_tpc.setText(str(self.vals[Tuner.Pid1]["topic"]))
        self._widget.t2_lp.setText(str(self.vals[Tuner.Pid2][PIDValue.Pl]))
        self._widget.t2_li.setText(str(self.vals[Tuner.Pid2][PIDValue.Il]))
        self._widget.t2_ld.setText(str(self.vals[Tuner.Pid2][PIDValue.Dl]))
        self._widget.t2_ls.setText(str(self.vals[Tuner.Pid2][PIDValue.Sl]))
        self._widget.t2_ap.setText(str(self.vals[Tuner.Pid2][PIDValue.Pa]))
        self._widget.t2_ai.setText(str(self.vals[Tuner.Pid2][PIDValue.Ia]))
        self._widget.t2_ad.setText(str(self.vals[Tuner.Pid2][PIDValue.Da]))
        self._widget.t2_as.setText(str(self.vals[Tuner.Pid2][PIDValue.Sa]))
        self._widget.t2_srv.setText(str(self.vals[Tuner.Pid2]["service"]))
        self._widget.t2_tpc.setText(str(self.vals[Tuner.Pid2]["topic"]))
        self._widget.t3_m.setText(str(self.vals[Tuner.Inertial][InertialValue.Mass]))
        self._widget.t3_b.setText(str(self.vals[Tuner.Inertial][InertialValue.Buoyancy]))
        self._widget.t3_ixx.setText(str(self.vals[Tuner.Inertial][InertialValue.Ixx]))
        self._widget.t3_iyy.setText(str(self.vals[Tuner.Inertial][InertialValue.Iyy]))
        self._widget.t3_izz.setText(str(self.vals[Tuner.Inertial][InertialValue.Izz]))
        self._widget.t3_srv.setText(str(self.vals[Tuner.Inertial]["service"]))
        self._widget.t3_tpc.setText(str(self.vals[Tuner.Inertial]["topic"]))

    def on_reload_values(self, tuner):
        if tuner == Tuner.Pid1:
            self.on_topic_name_changed(self._widget.t1_tpc, tuner)
        if tuner == Tuner.Pid2:
            self.on_topic_name_changed(self._widget.t2_tpc, tuner)
        if tuner == Tuner.Inertial:
            self.on_topic_name_changed(self._widget.t3_tpc, tuner)

        if tuner != Tuner.Inertial:
            self.vals[tuner][PIDValue.Pl] = self.cachedVals[tuner][PIDValue.Pl]
            self.vals[tuner][PIDValue.Il] = self.cachedVals[tuner][PIDValue.Il]
            self.vals[tuner][PIDValue.Dl] = self.cachedVals[tuner][PIDValue.Dl]
            self.vals[tuner][PIDValue.Sl] = self.cachedVals[tuner][PIDValue.Sl]
            self.vals[tuner][PIDValue.Pa] = self.cachedVals[tuner][PIDValue.Pa]
            self.vals[tuner][PIDValue.Ia] = self.cachedVals[tuner][PIDValue.Ia]
            self.vals[tuner][PIDValue.Da] = self.cachedVals[tuner][PIDValue.Da]
            self.vals[tuner][PIDValue.Sa] = self.cachedVals[tuner][PIDValue.Sa]
        else:
            self.vals[tuner][InertialValue.Mass] = self.cachedVals[tuner][InertialValue.Mass]
            self.vals[tuner][InertialValue.Buoyancy] = self.cachedVals[tuner][InertialValue.Buoyancy]
            self.vals[tuner][InertialValue.Iyy] = self.cachedVals[tuner][InertialValue.Ixx]
            self.vals[tuner][InertialValue.Ixx] = self.cachedVals[tuner][InertialValue.Iyy]
            self.vals[tuner][InertialValue.Izz] = self.cachedVals[tuner][InertialValue.Izz]
        self.update_text()

    def on_send_values(self, tuner):
        if tuner == Tuner.Pid1:
            self.on_srv_name_changed(self._widget.t1_srv, tuner)
        if tuner == Tuner.Pid2:
            self.on_srv_name_changed(self._widget.t2_srv, tuner)
        if tuner == Tuner.Inertial:
            self.on_srv_name_changed(self._widget.t3_srv, tuner)

        if tuner != Tuner.Inertial:
            try:
                tunesrv = rospy.ServiceProxy(self.vals[tuner]["service"], TunePid)
                tp = TunePid()
                tp.l_p = self.vals[tuner][PIDValue.Pl]
                tp.l_i = self.vals[tuner][PIDValue.Il]
                tp.l_d = self.vals[tuner][PIDValue.Dl]
                tp.l_sat = self.vals[tuner][PIDValue.Sl]
                tp.a_p = self.vals[tuner][PIDValue.Pa]
                tp.a_i = self.vals[tuner][PIDValue.Ia]
                tp.a_d = self.vals[tuner][PIDValue.Da]
                tp.a_sat = self.vals[tuner][PIDValue.Sa]
                tunesrv(tp)
            except rospy.ServiceException, e:
                print "Service call failed: %s" % e
        else:
            try:
                tunesrv = rospy.ServiceProxy(self.vals[tuner]["service"], TuneInertial)
                ti = TuneInertial()
                ti.mass = self.vals[tuner][InertialValue.Mass]
                ti.buoyancy = self.vals[tuner][InertialValue.Buoyancy]
                ti.ixx = self.vals[tuner][InertialValue.Ixx]
                ti.iyy = self.vals[tuner][InertialValue.Iyy]
                ti.izz = self.vals[tuner][InertialValue.Izz]
                succ = tunesrv(ti)
            except rospy.ServiceException, e:
                print "Service call failed: %s" % e

    def resub(self):
        if self.sub_t1 is not None:
            self.sub_t1.unregister()
        if self.sub_t2 is not None:
            self.sub_t2.unregister()
        if self.sub_t3 is not None:
            self.sub_t3.unregister()

        try:
            self.sub_t1 = rospy.Subscriber(self.vals[Tuner.Pid1]["topic"], PidVals,
                                           lambda msg: self.pidvals_callback(msg, Tuner.Pid1))
        except ValueError:
            pass

        try:
            self.sub_t2 = rospy.Subscriber(self.vals[Tuner.Pid2]["topic"], PidVals,
                                           lambda msg: self.pidvals_callback(msg, Tuner.Pid2))
        except ValueError:
            pass

        try:
            self.sub_t3 = rospy.Subscriber(self.vals[Tuner.Inertial]["topic"], InertialVals, self.inertialvals_callback)
        except ValueError:
            pass

    def pidvals_callback(self, msg, tuner):
        self.cachedVals[tuner][PIDValue.Pl] = msg.l_p
        self.cachedVals[tuner][PIDValue.Il] = msg.l_i
        self.cachedVals[tuner][PIDValue.Dl] = msg.l_d
        self.cachedVals[tuner][PIDValue.Sl] = msg.l_sat
        self.cachedVals[tuner][PIDValue.Pa] = msg.a_p
        self.cachedVals[tuner][PIDValue.Ia] = msg.a_i
        self.cachedVals[tuner][PIDValue.Da] = msg.a_d
        self.cachedVals[tuner][PIDValue.Sa] = msg.a_sat

    def inertialvals_callback(self, msg):
        self.cachedVals[Tuner.Inertial][InertialValue.Mass] = msg.mass
        self.cachedVals[Tuner.Inertial][InertialValue.Buoyancy] = msg.buoyancy
        self.cachedVals[Tuner.Inertial][InertialValue.Ixx] = msg.ixx
        self.cachedVals[Tuner.Inertial][InertialValue.Iyy] = msg.iyy
        self.cachedVals[Tuner.Inertial][InertialValue.Izz] = msg.izz
        print(self.cachedVals)

    def shutdown_plugin(self):
        # TODO unregister all publishers here
        pass

    def save_settings(self, plugin_settings, instance_settings):
        # TODO save intrinsic configuration, usually using:
        # instance_settings.set_value(k, v)
        instance_settings.set_value("t1_lp", self.vals[Tuner.Pid1][PIDValue.Pl])
        instance_settings.set_value("t1_li", self.vals[Tuner.Pid1][PIDValue.Il])
        instance_settings.set_value("t1_ld", self.vals[Tuner.Pid1][PIDValue.Dl])
        instance_settings.set_value("t1_ls", self.vals[Tuner.Pid1][PIDValue.Sl])
        instance_settings.set_value("t1_ap", self.vals[Tuner.Pid1][PIDValue.Pa])
        instance_settings.set_value("t1_ai", self.vals[Tuner.Pid1][PIDValue.Ia])
        instance_settings.set_value("t1_ad", self.vals[Tuner.Pid1][PIDValue.Da])
        instance_settings.set_value("t1_as", self.vals[Tuner.Pid1][PIDValue.Sa])
        instance_settings.set_value("t2_lp", self.vals[Tuner.Pid2][PIDValue.Pl])
        instance_settings.set_value("t2_li", self.vals[Tuner.Pid2][PIDValue.Il])
        instance_settings.set_value("t2_ld", self.vals[Tuner.Pid2][PIDValue.Dl])
        instance_settings.set_value("t2_ls", self.vals[Tuner.Pid2][PIDValue.Sl])
        instance_settings.set_value("t2_ap", self.vals[Tuner.Pid2][PIDValue.Pa])
        instance_settings.set_value("t2_ai", self.vals[Tuner.Pid2][PIDValue.Ia])
        instance_settings.set_value("t2_ad", self.vals[Tuner.Pid2][PIDValue.Da])
        instance_settings.set_value("t2_as", self.vals[Tuner.Pid2][PIDValue.Sa])
        instance_settings.set_value("t3_m", self.vals[Tuner.Inertial][InertialValue.Mass])
        instance_settings.set_value("t3_b", self.vals[Tuner.Inertial][InertialValue.Buoyancy])
        instance_settings.set_value("t3_ixx", self.vals[Tuner.Inertial][InertialValue.Ixx])
        instance_settings.set_value("t3_iyy", self.vals[Tuner.Inertial][InertialValue.Iyy])
        instance_settings.set_value("t3_izz", self.vals[Tuner.Inertial][InertialValue.Izz])

        instance_settings.set_value("t1_srv", self.vals[Tuner.Pid1]["service"])
        instance_settings.set_value("t2_srv", self.vals[Tuner.Pid2]["service"])
        instance_settings.set_value("t3_srv", self.vals[Tuner.Inertial]["service"])
        instance_settings.set_value("t1_tpc", self.vals[Tuner.Pid1]["topic"])
        instance_settings.set_value("t2_tpc", self.vals[Tuner.Pid2]["topic"])
        instance_settings.set_value("t3_tpc", self.vals[Tuner.Inertial]["topic"])

    def restore_settings(self, plugin_settings, instance_settings):
        # TODO restore intrinsic configuration, usually using:
        self.vals[Tuner.Pid1][PIDValue.Pl] = instance_settings.value("t1_lp")
        self.vals[Tuner.Pid1][PIDValue.Il] = instance_settings.value("t1_li")
        self.vals[Tuner.Pid1][PIDValue.Dl] = instance_settings.value("t1_ld")
        self.vals[Tuner.Pid1][PIDValue.Sl] = instance_settings.value("t1_ls")
        self.vals[Tuner.Pid1][PIDValue.Pa] = instance_settings.value("t1_ap")
        self.vals[Tuner.Pid1][PIDValue.Ia] = instance_settings.value("t1_ai")
        self.vals[Tuner.Pid1][PIDValue.Da] = instance_settings.value("t1_ad")
        self.vals[Tuner.Pid1][PIDValue.Sa] = instance_settings.value("t1_as")
        self.vals[Tuner.Pid2][PIDValue.Pl] = instance_settings.value("t2_lp")
        self.vals[Tuner.Pid2][PIDValue.Il] = instance_settings.value("t2_li")
        self.vals[Tuner.Pid2][PIDValue.Dl] = instance_settings.value("t2_ld")
        self.vals[Tuner.Pid2][PIDValue.Sl] = instance_settings.value("t2_ls")
        self.vals[Tuner.Pid2][PIDValue.Pa] = instance_settings.value("t2_ap")
        self.vals[Tuner.Pid2][PIDValue.Ia] = instance_settings.value("t2_ai")
        self.vals[Tuner.Pid2][PIDValue.Da] = instance_settings.value("t2_ad")
        self.vals[Tuner.Pid2][PIDValue.Sa] = instance_settings.value("t2_as")
        self.vals[Tuner.Inertial][InertialValue.Mass] = instance_settings.value("t3_m")
        self.vals[Tuner.Inertial][InertialValue.Buoyancy] = instance_settings.value("t3_b")
        self.vals[Tuner.Inertial][InertialValue.Ixx] = instance_settings.value("t3_ixx")
        self.vals[Tuner.Inertial][InertialValue.Iyy] = instance_settings.value("t3_iyy")
        self.vals[Tuner.Inertial][InertialValue.Izz] = instance_settings.value("t3_izz")

        self.vals[Tuner.Pid1]["service"] = instance_settings.value("t1_srv")
        self.vals[Tuner.Pid2]["service"] = instance_settings.value("t2_srv")
        self.vals[Tuner.Inertial]["service"] = instance_settings.value("t3_srv")
        self.vals[Tuner.Pid1]["topic"] = instance_settings.value("t1_tpc")
        self.vals[Tuner.Pid2]["topic"] = instance_settings.value("t2_tpc")
        self.vals[Tuner.Inertial]["topic"] = instance_settings.value("t3_tpc")
        self.update_text()
        self.resub()

    # def trigger_configuration(self):
    # Comment in to signal that the plugin has a way to configure
    # This will enable a setting button (gear icon) in each dock widget title bar
    # Usually used to open a modal configuration dialog
