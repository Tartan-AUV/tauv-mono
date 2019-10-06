#!/usr/bin/env python
from controllers import UserController
import sys
import atexit
import curses

def shutdownHandler():
    curses.endwin()

if __name__ == "__main__":
    stdscr = curses.initscr()
    curses.noecho()
    atexit.register(shutdownHandler)
    uc = UserController()
    while(not rospy.is_shutdown()):
        uc.spin()