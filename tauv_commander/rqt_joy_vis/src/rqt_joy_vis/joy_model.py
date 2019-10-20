import os
import rospy
import rospkg

from qt_gui.plugin import Plugin
from python_qt_binding import loadUi
from python_qt_binding.QtWidgets import QWidget
from python_qt_binding.QtCore import Signal, QObject



class JoyModel(QObject):

    def __init__(self):
        self.buttonSignals = [Signal() for _ in range(12)]
        self.axesSignals = [Signal() for _ in range(8)]
        self._is_connected = False

    def __del__(self):

    @Slot()
    def doConnect(self):
        if self._is_connected:
            self.doDisconnect()

    @Slot()
    def doDisconnect(self):
