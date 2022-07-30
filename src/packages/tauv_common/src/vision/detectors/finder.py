#!/usr/bin/env python3

import rospy
from tauv_msgs.msg import BucketList, BucketDetection, Pose
import geometry_msgs.msg

class Finder():
    def __init__(self):
        # rospy.init_node('bucket_finder')
        self.bucket_list = []
        self.bucket = rospy.Subscriber("bucket_list", BucketList, self.update)
        # rospy.spin()

    #return the first bucket detection with the key matching name
    def find_by_tag(self, tag):
        for entry in self.bucket_list:
            if entry.tag == tag:
                return entry
        return None

    def update(self, lst):
        self.bucket_list = lst.bucket_list


    def printer(self):
        entry = self.find_by_tag("badge")
        if(entry!=None):
            rospy.loginfo(f"badge position: {entry.position}")