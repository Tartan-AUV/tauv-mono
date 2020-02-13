import os
import rospy
import rospkg

from qt_gui.plugin import Plugin
from python_qt_binding import loadUi
from python_qt_binding.QtWidgets import QWidget
from python_qt_binding.QtCore import Signal, QObject, Slot, Qt

from response_model import ResponseModel


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

        self._leftStickWidget = AxisWidget()
        self._rightStickWidget = AxisWidget()
        self._dPadStickWidget = AxisWidget()

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
