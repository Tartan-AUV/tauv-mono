#!/usr/bin/env python
import rospy
import roslib
import time
import cv2 as cv2
from pynput import keyboard
import sys
import atexit
import curses

from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Wrench
from cv_bridge import CvBridge, CvBridgeError
from uuv_gazebo_ros_plugins_msgs.msg import FloatStamped
from sensor_msgs.msg import Image
import numpy as np
import matplotlib.pyplot as plt

class UserControl:
    def __init__(self):
        self.gain = [30, 30, 60, 20, 20, 8]       
        self.pub = rospy.Publisher("/manta/thruster_manager/input", Wrench, queue_size=1)
        listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        listener.start()
        self.keys = dict()
        self.vector = [0, 0, 0, 0, 0, 0]
        

    def on_press(self, key):
        k = str(key)
        k = k.strip("u\'")
        self.keys[k] = True

    def on_release(self, key):
        k = str(key)
        k = k.strip("u\'")
        self.keys[k] = False

    def is_pressed(self, key):
        if key in self.keys and self.keys[key] == True:
            return True
        return False

    def get_input(self):
        self.vector = [0, 0, 0, 0, 0, 0]
        if self.is_pressed("w"):
            self.vector[0] = self.gain[0]
        if self.is_pressed("s"):
            self.vector[0] = -self.gain[0]
        if self.is_pressed("a"):
            self.vector[1] = self.gain[1]
        if self.is_pressed("d"):
            self.vector[1] = -self.gain[1]
        if self.is_pressed("r"):
            self.vector[2] = self.gain[2]
        if self.is_pressed("f"):
            self.vector[2] = -self.gain[2]
        if self.is_pressed("Key.down"):
            self.vector[4] = -self.gain[4]
        if self.is_pressed("Key.up"):
            self.vector[4] = self.gain[4]
        if self.is_pressed("Key.left"):
            self.vector[5] = self.gain[5]
        if self.is_pressed("Key.right"):
            self.vector[5] = -self.gain[5]
        if self.is_pressed("e"):
            self.vector[3] = self.gain[3]
        if self.is_pressed("q"):
            self.vector[3] = -self.gain[3]        

    def send_thrust(self):
        command = self.vector
        msg = Wrench()
        msg.force.x = command[0]
        msg.force.y = command[1]
        msg.force.z = command[2]
        msg.torque.x = command[3]
        msg.torque.y = command[4]
        msg.torque.z = command[5]

        self.pub.publish(msg)

    def spin(self):
        self.get_input()
        self.send_thrust()
        time.sleep(0.01)

def shutdownHandler():
    curses.endwin()

def main():
    stdscr = curses.initscr()
    curses.noecho()
    atexit.register(shutdownHandler)
    rospy.init_node('user_control', anonymous=True)
    uc = UserControl()
    while(not rospy.is_shutdown()):
        uc.spin()

if __name__ == "__main__":
    main()