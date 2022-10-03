import rospy

class Acoustics():

    def __init__(self):
        pass

    def start(self):
        rospy.spin()

def main():
    rospy.init_node('acoustics')
    a = Acoustics()
    a.start()