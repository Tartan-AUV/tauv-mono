import os
import rospy
import rospkg

from collections import namedtuple
from qt_gui.plugin import Plugin
from python_qt_binding import loadUi
from python_qt_binding.QtWidgets import QWidget
from python_qt_binding.QtCore import Signal, QObject, Slot, Qt, QTimer
from python_qt_binding.QtGui import QColor

from data_plot import DataPlot
from response_model import ResponseModel, ResponseType
import json
from enum import Enum


class ResponseMode(Enum):
    pos_lin = 1
    pos_ang = 2
    vel_lin = 3
    vel_ang = 4
    acc_lin = 5
    acc_ang = 6


ModeResponses = {ResponseMode.pos_lin: [ResponseType.pos_x, ResponseType.pos_y, ResponseType.pos_z,
                                        ResponseType.cmd_pos_x, ResponseType.cmd_pos_y, ResponseType.cmd_pos_z],
                 ResponseMode.pos_ang: [ResponseType.pos_wx, ResponseType.pos_wy, ResponseType.pos_wz,
                                        ResponseType.cmd_pos_wx, ResponseType.cmd_pos_wy, ResponseType.cmd_pos_wz],
                 ResponseMode.vel_lin: [ResponseType.vel_x, ResponseType.vel_y, ResponseType.vel_z,
                                        ResponseType.cmd_vel_x, ResponseType.cmd_vel_y, ResponseType.cmd_vel_z],
                 ResponseMode.vel_ang: [ResponseType.vel_wx, ResponseType.vel_wy, ResponseType.vel_wz,
                                        ResponseType.cmd_vel_wx, ResponseType.cmd_vel_wy, ResponseType.cmd_vel_wz],
                 ResponseMode.acc_lin: [ResponseType.acc_x, ResponseType.acc_y, ResponseType.acc_z,
                                        ResponseType.cmd_acc_x, ResponseType.cmd_acc_y, ResponseType.cmd_acc_z],
                 ResponseMode.acc_ang: [ResponseType.acc_wx, ResponseType.acc_wy, ResponseType.acc_wz,
                                        ResponseType.cmd_acc_wx, ResponseType.cmd_acc_wy, ResponseType.cmd_acc_wz]}


class ResponseVis(Plugin):

    def __init__(self, context):
        super(ResponseVis, self).__init__(context)
        # Give QObjects reasonable names
        self.setObjectName('JoyVis')

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
        self._ui = QWidget()
        # Get path to UI file which should be in the "resource" folder of this package
        ui_file = os.path.join(rospkg.RosPack().get_path('rqt_response_vis'), 'resource', 'plot.ui')
        # Extend the widget with all attributes and children from UI file
        loadUi(ui_file, self._ui)

        self._plotWidget = DataPlot()
        self._redraw_interval = 1.0/60.0

        self._ui.plot_frame.layout().addWidget(self._plotWidget)

        self._ui.resp_acc_ang.toggled.connect(self.on_response_changed)
        self._ui.resp_acc_lin.toggled.connect(self.on_response_changed)
        self._ui.resp_vel_ang.toggled.connect(self.on_response_changed)
        self._ui.resp_vel_lin.toggled.connect(self.on_response_changed)
        self._ui.resp_pos_ang.toggled.connect(self.on_response_changed)
        self._ui.resp_pos_lin.toggled.connect(self.on_response_changed)

        self._ui.slider_zoom_ms.valueChanged.connect(self.on_width_slider_changed)
        self._ui.check_pause.toggled.connect(self.on_pause_button_clicked)

        self.horizon = 5.0
        self._plotWidget.set_x_width(self.horizon)

        # Give QObjects reasonable names
        self._ui.setObjectName('ResponseVis')
        # Show _widget.windowTitle on left-top of each plugin (when
        # it's set in _widget). This is useful when you open multiple
        # plugins at once. Also if you open multiple instances of your
        # plugin at once, these lines add number to make it easy to
        # tell from pane to pane.
        if context.serial_number() > 1:
            self._ui.setWindowTitle(self._ui.windowTitle() + (' (%d)' % context.serial_number()))
        # Add widget to the user interface
        context.add_widget(self._ui)

        self._response_model = ResponseModel()

        self.topics = {"odom": "/gnc/odom",
                       "cmd_acc": "/gnc/cmd_accel",
                       "cmd_vel": "/gnc/cmd_vel",
                       "cmd_pos": "/gnc/cmd_pose"}
        self._response_model.declare_subscribers(self.topics)

        self.mode = None
        self._update_plot_timer = QTimer(self)
        self._update_plot_timer.timeout.connect(self.update_plot)

        self._ui.show()

        self.on_response_changed()
        self.enable_timer()

    def shutdown_plugin(self):
        print("shutting down joy_vis plugin")

    def save_settings(self, plugin_settings, instance_settings):
        # instance_settings.set_value(k, v)
        instance_settings.set_value("topics", json.dumps(self.topics))
        pass

    def restore_settings(self, plugin_settings, instance_settings):
        # v = instance_settings.value(k)
        self.topics = json.loads(instance_settings.value("topics"))
        # self._response_model.declare_subscribers(self.topics)
        pass

    @Slot()
    def update_plot(self):
        if self._plotWidget is not None and self.mode is not None:
            self._plotWidget.set_autoscale(x=None, y=DataPlot.SCALE_ALL)

            for i in range(6):
                x, y = self._response_model.get_data(ModeResponses[self.mode][i], self._redraw_interval*10)
                self._plotWidget.update_values(str(i), x, y)

            self._plotWidget.redraw()

    @Slot()
    def on_response_changed(self):
        if self._ui.resp_pos_lin.isChecked():
            self.mode = ResponseMode.pos_lin

        if self._ui.resp_pos_ang.isChecked():
            self.mode = ResponseMode.pos_ang

        if self._ui.resp_vel_lin.isChecked():
            self.mode = ResponseMode.vel_lin

        if self._ui.resp_vel_ang.isChecked():
            self.mode = ResponseMode.vel_ang

        if self._ui.resp_acc_lin.isChecked():
            self.mode = ResponseMode.acc_lin

        if self._ui.resp_acc_ang.isChecked():
            self.mode = ResponseMode.acc_ang

        if self.mode is None:
            return

        self._plotWidget.remove_curve(0)
        self._plotWidget.remove_curve(1)
        self._plotWidget.remove_curve(2)
        self._plotWidget.remove_curve(3)
        self._plotWidget.remove_curve(4)
        self._plotWidget.remove_curve(5)
        x, y = self._response_model.get_data(ModeResponses[self.mode][0])
        self._plotWidget.add_curve(str(0), ModeResponses[self.mode][0].name, x, y, curve_color=QColor(255, 0, 0))
        x, y = self._response_model.get_data(ModeResponses[self.mode][1])
        self._plotWidget.add_curve(str(1), ModeResponses[self.mode][1].name, x, y, curve_color=QColor(0, 255, 0))
        x, y = self._response_model.get_data(ModeResponses[self.mode][2])
        self._plotWidget.add_curve(str(2), ModeResponses[self.mode][2].name, x, y, curve_color=QColor(0, 0, 255))
        x, y = self._response_model.get_data(ModeResponses[self.mode][3])
        self._plotWidget.add_curve(str(3), ModeResponses[self.mode][3].name, x, y, curve_color=QColor(255, 127, 127), dashed=True)
        x, y = self._response_model.get_data(ModeResponses[self.mode][4])
        self._plotWidget.add_curve(str(4), ModeResponses[self.mode][4].name, x, y, curve_color=QColor(127, 255, 127), dashed=True)
        x, y = self._response_model.get_data(ModeResponses[self.mode][5])
        self._plotWidget.add_curve(str(5), ModeResponses[self.mode][5].name, x, y, curve_color=QColor(127, 127, 255), dashed=True)
        self._plotWidget.redraw()

    @Slot()
    def on_pause_button_clicked(self):
        self.enable_timer(not self._ui.check_pause.isChecked())

    @Slot()
    def on_width_slider_changed(self):
        val = self._ui.slider_zoom_ms.value() / 1000.0
        self._plotWidget.set_x_width(val)
        self._plotWidget.redraw()

    def enable_timer(self, enabled=True):
        if enabled:
            self._update_plot_timer.start(self._redraw_interval)
        else:
            self._update_plot_timer.stop()

    # def trigger_configuration(self):
    # Comment in to signal that the plugin has a way to configure
    # This will enable a setting button (gear icon) in each dock widget title bar
    # Usually used to open a modal configuration dialog
