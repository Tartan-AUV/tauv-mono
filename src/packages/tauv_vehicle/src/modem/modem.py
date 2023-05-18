import rospy
from tauv_msgs.msg import ModemFrame

class ModemNode:
    def __init__(self):
        pass

    def start(self):
        while not rospy.is_shutdown():
            pass


def main():
    rospy.init_node('modem')
    n = ModemNode()
    n.start()