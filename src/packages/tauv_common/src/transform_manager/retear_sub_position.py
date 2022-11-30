#!/usr/bin/env python3

import rospy
import tf2_ros
import geometry_msgs.msg

class RetearSubPosition():
    def __init__(self):
        self._NODE_NAME = "retear_sub_position"
        self._NODE_NAME_FMT = "[{}]".format(self._NODE_NAME)

        self._WORLD_FRAME_ID = "world_ned"
        self._ODOM_FRAME_ID = "odom_ned"
    
    def handle_retear(self, msg):
        br = tf2_ros.TransformBroadcaster()
        t = geometry_msgs.msg.TransformStamped()

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self._WORLD_FRAME_ID
        t.child_frame_id = self._ODOM_FRAME_ID

        t.transform.translation.x = msg.position.x
        t.transform.translation.y = msg.position.y
        t.transform.translation.z = msg.position.z

        t.transform.orientation.x = msg.orientation.x
        t.transform.orientation.y = msg.orientation.y
        t.transform.orientation.z = msg.orientation.z
        t.transform.orientation.w = msg.orientation.w

        try:
            br.sendTransform(t)
            return True; # return success code
        except Exception as e:
            rospy.logerr("{} Could not send {} -> {} tf transform. Got execption:\n{}"
                .format(self._NODE_NAME_FMT, self._WORLD_FRAME_ID, self._ODOM_FRAME_ID, e))
            return False; # return success code

    def start(self):
        rospy.init_node(self._NODE_NAME)
        s = rospy.Service(self._NODE_NAME, RetearSubPosition, self.handle_retear)
        rospy.spin()

if __name__ == "__main__":
    r = RetearSubPosition()
    r.start()