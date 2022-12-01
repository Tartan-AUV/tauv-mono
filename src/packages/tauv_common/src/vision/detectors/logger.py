#!/usr/bin/env python3

import rospy
from tauv_msgs.msg import FeatureDetections, FeatureDetection
from tauv_msgs.srv import MapFind, MapFindClosest
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
import numpy as np

def getColor(tag, trans=False):
    color = [0,0,0,0]

    if(tag=="phone"):
        color = [1,0,0,1]

    if(tag=="badge"):
        color = [0,1,0,1]

    if(tag=="notebook"):
        color = [0,0,1,1]

    if(not trans):
        color[3] = 0.75

    return color

def makeMarker(id, detection, color, scale = 0.5, shape = Marker.SPHERE, time = rospy.Duration(1.0)):
    marker = Marker()
    marker.header.frame_id = "odom_ned"
    marker.header.stamp = rospy.Time()
    marker.ns = "my_namespace"
    marker.id = id
    marker.type = shape
    marker.action = Marker.ADD
    marker.pose.position.x = detection.position.x
    marker.pose.position.y = detection.position.y
    marker.pose.position.z = detection.position.z
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    marker.scale.x = scale
    marker.scale.y = scale
    marker.scale.z = scale
    marker.color.a = color[3]
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.lifetime = time
    
    return marker


class Logger():
    def __init__(self):
        rospy.init_node('logger')
        rospy.Subscriber("/global_map/transform_detections", FeatureDetections,
                                                self.publish)

        rospy.Subscriber("/global_map/find", FeatureDetections,
                                        self.visualize)

        rospy.wait_for_service("/global_map/find")
        self.find = rospy.ServiceProxy("/global_map/find", MapFind)

        self.viz = rospy.Publisher("/visualization_marker_array", MarkerArray, queue_size=100)
        
        rospy.Timer(rospy.Duration(0.5), self.visualize)

        self.ind = 10000

    def visualize(self, time):
        #buckets = bucketList.bucket_list
        buckets1 = self.find("badge").detections
        buckets2 = self.find("phone").detections
        buckets3 = self.find("notebook").detections

        markers = []
        for ind in range(len(buckets1)):
            det = buckets1[ind]
            markers.append(makeMarker(ind, det, getColor(det.tag), 1, Marker.CUBE, rospy.Duration(5.0)))

        for ind in range(len(buckets2)):
            det = buckets2[ind]
            markers.append(makeMarker(ind+len(buckets1), det, getColor(det.tag), 1, Marker.CUBE, rospy.Duration(5.0)))

        for ind in range(len(buckets3)):
            det = buckets3[ind]
            markers.append(makeMarker(ind+len(buckets1)+len(buckets2), det, getColor(det.tag), 1, Marker.CUBE, rospy.Duration(5.0)))

        OBJ = MarkerArray()
        OBJ.markers = markers

        self.viz.publish(OBJ)

    def publish(self, objects):
        detections = objects.detections
        markers = []
        for detection in detections:
            markers.append(makeMarker(self.ind, detection, getColor(detection.tag, True)))
            self.ind+=1

        markersPub = MarkerArray()
        markersPub.markers = markers
        self.viz.publish(markersPub)

s = Logger()
rospy.spin()