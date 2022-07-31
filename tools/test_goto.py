import rospy
from motion.motion_utils import MotionUtils
rospy.init_node('test')

mu = MotionUtils()

p, t = mu.get_robot_state()

print(f"Pose: {p}, Twist: {t}")

mu.goto([0,0,0.5], v=0.1)

