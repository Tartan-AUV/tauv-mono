from maestro import Maestro
import rospy

class ActuatorController:
    def __init__(self):
        maestro_tty = rospy.getParam('/albatross/maestro')
        self.maestro = Maestro(ttyStr=maestro_tty)
        self.channels = range(0,num_thrusters)
    def start(self):
