import os
import rospy
import rospkg

from python_qt_binding.QtCore import Signal, QObject, Slot
from joy.srv import JoyConnect
from std_srvs.srv import Trigger
from sensor_msgs.msg import Joy
import glob


class JoyModel(QObject):
    buttonSignal0 = Signal(bool)
    buttonSignal1 = Signal(bool)
    buttonSignal2 = Signal(bool)
    buttonSignal3 = Signal(bool)
    buttonSignal4 = Signal(bool)
    buttonSignal5 = Signal(bool)
    buttonSignal6 = Signal(bool)
    buttonSignal7 = Signal(bool)
    buttonSignal8 = Signal(bool)
    buttonSignal9 = Signal(bool)
    buttonSignal10 = Signal(bool)
    axisSignal0 = Signal(float)
    axisSignal1 = Signal(float)
    axisSignal2 = Signal(float)
    axisSignal3 = Signal(float)
    axisSignal4 = Signal(float)
    axisSignal5 = Signal(float)
    axisSignal6 = Signal(float)
    axisSignal7 = Signal(float)

    def __init__(self, parent=None):
        super(JoyModel, self).__init__(parent)
        self._is_connected = False

        self._joySubscriber = rospy.Subscriber('joy', Joy, self.joyCallback)
        self._connectionService = rospy.ServiceProxy('joy_node/connect', JoyConnect)
        self._disconnectionService = rospy.ServiceProxy('joy_node/disconnect', Trigger)
        self._shutdownService = rospy.ServiceProxy('joy_node/shutdown', Trigger)

        self._devname = ""
        self._devpath = "/dev/input/"

    def __del__(self):
        print("deleting joystick visualizer.")
        self.doDisconnect()
        self._joySubscriber.unregister()
        self._connectionService.close()
        self._disconnectionService.close()
        self._shutdownService.close()

    @Slot()
    def doConnect(self):
        if self._is_connected:
            self.doDisconnect()
        try:
            rospy.wait_for_service("joy_node/connect", 0.2)
        except rospy.ROSException, e:
            rospy.logerr("Cannot connect Joystick! No Joystick Node Running: %s", e)
            return
        #
        # jc = JoyConnect()
        # jc.dev = "asdf"#str(self._devpath + self._devname)
        self._connectionService(self._devpath + self._devname)

    @Slot()
    def doDisconnect(self):
        try:
            rospy.wait_for_service("joy_node/disconnect", 0.2)
        except rospy.ROSException, e:
            rospy.logerr("Cannot disconnect Joystick! No Joystick Node Running: %s", e)
            return

        self._disconnectionService()

    def doShutdown(self):
        try:
            rospy.wait_for_service("joy_node/shutdown", 0.2)
        except rospy.ROSException, e:
            rospy.logerr("Cannot shutdown joystick node! No Joystick Node Running: %s", e)
            return

        self._shutdownService()

    def getDevList(self):
        return [os.path.basename(x) for x in glob.glob(self._devpath + 'js*')]

    def changeDev(self, devname):
        self._devname = devname

    def joyCallback(self, msg):
        self.buttonSignal0.emit(msg.buttons[0] != 0)
        self.buttonSignal1.emit(msg.buttons[1] != 0)
        self.buttonSignal2.emit(msg.buttons[2] != 0)
        self.buttonSignal3.emit(msg.buttons[3] != 0)
        self.buttonSignal4.emit(msg.buttons[4] != 0)
        self.buttonSignal5.emit(msg.buttons[5] != 0)
        self.buttonSignal6.emit(msg.buttons[6] != 0)
        self.buttonSignal7.emit(msg.buttons[7] != 0)
        self.buttonSignal8.emit(msg.buttons[8] != 0)
        self.buttonSignal9.emit(msg.buttons[9] != 0)
        self.buttonSignal10.emit(msg.buttons[10] != 0)
        self.axisSignal0.emit(-msg.axes[0])
        self.axisSignal1.emit(-msg.axes[1])
        self.axisSignal2.emit(-msg.axes[2])
        self.axisSignal3.emit(-msg.axes[3])
        self.axisSignal4.emit(-msg.axes[4])
        self.axisSignal5.emit(-msg.axes[5])
        self.axisSignal6.emit(-msg.axes[6])
        self.axisSignal7.emit(-msg.axes[7])
