import os
import rospy
import rospkg

from qt_gui.plugin import Plugin
from python_qt_binding import loadUi
from python_qt_binding.QtWidgets import QWidget
from python_qt_binding.QtCore import Signal, QObject, Slot, Qt

from joy_model import JoyModel
from axis_widget import AxisWidget


class JoyVis(Plugin):

    def __init__(self, context):
        super(JoyVis, self).__init__(context)
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
        self._widget = QWidget()
        # Get path to UI file which should be in the "resource" folder of this package
        ui_file = os.path.join(rospkg.RosPack().get_path('rqt_joy_vis'), 'resource', 'joy_vis.ui')
        # Extend the widget with all attributes and children from UI file
        loadUi(ui_file, self._widget)

        self._leftStickWidget = AxisWidget()
        self._rightStickWidget = AxisWidget()
        self._dPadStickWidget = AxisWidget()

        self._widget.axis_frame.layout().addWidget(self._leftStickWidget)
        self._widget.axis_frame.layout().addWidget(self._rightStickWidget)
        self._widget.axis_frame.layout().addWidget(self._dPadStickWidget)

        # Give QObjects reasonable names
        self._widget.setObjectName('JoyVis')
        # Show _widget.windowTitle on left-top of each plugin (when
        # it's set in _widget). This is useful when you open multiple
        # plugins at once. Also if you open multiple instances of your
        # plugin at once, these lines add number to make it easy to
        # tell from pane to pane.
        if context.serial_number() > 1:
            self._widget.setWindowTitle(self._widget.windowTitle() + (' (%d)' % context.serial_number()))
        # Add widget to the user interface
        context.add_widget(self._widget)

        self._joyModel = JoyModel()

        self._widget.refresh_joysticks.clicked.connect(self.refresh_pressed)
        self._widget.connect_joystick.clicked.connect(self._joyModel.doConnect)
        self._widget.disconnect_joystick.clicked.connect(self._joyModel.doDisconnect)
        self._widget.joysticks_list.currentIndexChanged.connect(self.updateDevSelection)

        self._joyModel.axisSignal0.connect(self._leftStickWidget.setX)
        self._joyModel.axisSignal1.connect(self._leftStickWidget.setY)
        self._joyModel.axisSignal3.connect(self._rightStickWidget.setX)
        self._joyModel.axisSignal4.connect(self._rightStickWidget.setY)
        self._joyModel.axisSignal6.connect(self._dPadStickWidget.setX)
        self._joyModel.axisSignal7.connect(self._dPadStickWidget.setY)

        self._widget.left_trigger.setRange(-100, 100)
        self._widget.right_trigger.setRange(-100, 100)
        self._joyModel.axisSignal2.connect(self.trigger_remap_left)
        self._joyModel.axisSignal5.connect(self.trigger_remap_right)

        self._joyModel.buttonSignal0.connect(self._widget.btn_1.setChecked)
        self._joyModel.buttonSignal1.connect(self._widget.btn_2.setChecked)
        self._joyModel.buttonSignal2.connect(self._widget.btn_3.setChecked)
        self._joyModel.buttonSignal3.connect(self._widget.btn_4.setChecked)
        self._joyModel.buttonSignal4.connect(self._widget.btn_5.setChecked)
        self._joyModel.buttonSignal5.connect(self._widget.btn_6.setChecked)
        self._joyModel.buttonSignal6.connect(self._widget.btn_7.setChecked)
        self._joyModel.buttonSignal7.connect(self._widget.btn_8.setChecked)
        self._joyModel.buttonSignal8.connect(self._widget.btn_9.setChecked)
        self._joyModel.buttonSignal9.connect(self._widget.btn_10.setChecked)
        self._joyModel.buttonSignal10.connect(self._widget.btn_11.setChecked)

        self._widget.btn_1.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._widget.btn_2.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._widget.btn_3.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._widget.btn_4.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._widget.btn_5.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._widget.btn_6.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._widget.btn_7.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._widget.btn_8.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._widget.btn_9.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._widget.btn_10.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._widget.btn_11.setAttribute(Qt.WA_TransparentForMouseEvents)

        self.refresh_pressed()
        self.updateDevSelection()
        self._joyModel.doConnect()


    @Slot(float)
    def trigger_remap_left(self, val):
        self._widget.left_trigger.setValue(val*100)

    @Slot(float)
    def trigger_remap_right(self, val):
        self._widget.right_trigger.setValue(val*100)

    @Slot()
    def refresh_pressed(self):
        devs = self._joyModel.getDevList()
        self._widget.joysticks_list.clear()
        self._widget.joysticks_list.insertItems(0,devs)

    @Slot()
    def updateDevSelection(self):
        self._joyModel.changeDev(self._widget.joysticks_list.currentText())

    def shutdown_plugin(self):
        print("shutting down joy_vis plugin")
        del self._joyModel

    def save_settings(self, plugin_settings, instance_settings):
        # TODO save intrinsic configuration, usually using:
        # instance_settings.set_value(k, v)
        pass

    def restore_settings(self, plugin_settings, instance_settings):
        # TODO restore intrinsic configuration, usually using:
        # v = instance_settings.value(k)
        pass

    # def trigger_configuration(self):
    # Comment in to signal that the plugin has a way to configure
    # This will enable a setting button (gear icon) in each dock widget title bar
    # Usually used to open a modal configuration dialog
