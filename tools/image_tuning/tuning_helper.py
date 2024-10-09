#command-line script for finding HSV filtering parameters in images

#TO RUN: python3 "./tuning_helper.py" relative_path_to_file
#where relative_path_to_file is a string path to a .jpg/.png/.mp4 file relative to current folder
#other formats not accepted (e.g. .jpeg) and the script automatically deduces whether file is image or video without other command-line arguments

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider, Button
import matplotlib.colors as mcolors
import sys

class Frame():
    def __init__(self, path, img=True):
        #distinguish between video and img
        self.image = img
        self.path = path

        #figure setup
        self.fig, self.axs = plt.subplots(1,1)
        self.fig.subplots_adjust(bottom=0.4)
        self.fig.suptitle('HSV Tuner', fontsize=20)
        
        #video initial frame setup
        if(not self.image):
            self.cap = cv.VideoCapture(self.path)
            if not self.cap.isOpened():
                sys.exit(0)
            ret, frame = self.cap.read()
            if not ret:
                sys.exit(0)
            self.__setup_frame(frame)
        else: #image frame setup
            init_frame = np.array(cv.imread(self.path))
            self.__setup_frame(init_frame)
        
        #display first frame 
        self.im = self.axs.imshow(self.display)

        #create HSV range sliders
        axH = self.fig.add_axes([0.20, 0.3, 0.60, 0.03])
        self.sliderH = RangeSlider(axH, "Hue Threshold", 0, 179)
        axS = self.fig.add_axes([0.30, 0.22, 0.50, 0.03])
        self.sliderS = RangeSlider(axS, "Saturation Threshold", 0, 255)
        axV = self.fig.add_axes([0.20, 0.14, 0.60, 0.03])
        self.sliderV = RangeSlider(axV, "Value Threshold", 0, 255)

        #update initial image to initial values
        self.update()

        #set sliders to trigger frame updates when changed
        self.sliderH.on_changed(self.update)
        self.sliderS.on_changed(self.update)
        self.sliderV.on_changed(self.update)

        #create slideshow button to click through frames
        axB = plt.axes([0.81, 0.04, 0.1, 0.075])
        bnext = Button(axB, 'Next',color=mcolors.CSS4_COLORS["palegreen"])
        bnext.on_clicked(self.next)

        plt.show()

    #get next video frame or exit program if none remain or is image
    def next(self, event=""):
        if self.image or not self.cap.isOpened():
            self.exit()
        ret, frame = self.cap.read()
        if not ret:
            self.exit()
        self.__setup_frame(frame)
        self.update()

    #set the next frame
    def __setup_frame(self, init_frame):
        self.hsv = cv.cvtColor(init_frame, cv.COLOR_BGR2HSV)
        self.display = cv.cvtColor(init_frame, cv.COLOR_BGR2RGB)
        self.ZEROS = np.zeros_like(self.display)

    #update the view based on current hsv slider values
    def update(self, range=[0,0]):
        filtH=cv.inRange(self.hsv[:,:,0], self.sliderH.val[0], self.sliderH.val[1])
        filtS=cv.inRange(self.hsv[:,:,1], self.sliderS.val[0], self.sliderS.val[1])
        filtV=cv.inRange(self.hsv[:,:,2], self.sliderV.val[0], self.sliderV.val[1])
        filt = np.logical_and(np.logical_and(filtH, filtS),filtV)
        filt3D = np.repeat(np.expand_dims(filt, axis=2),3, axis=2)
        
        frame = np.where(filt3D, self.display, self.ZEROS)
        self.im.set_data(frame)
        self.fig.canvas.draw_idle()

    #terminate the program
    def exit(self, code = 0):
        print("Done.")
        sys.exit(code)

#takes relative pathname of file as only cmdline argument
def main():
    file_name = sys.argv[1]
    if(file_name[-4:]==".png" or file_name[-4:]==".jpg"):
        Frame(file_name, True)
    elif (file_name[-4:]==".mp4"):
        Frame(file_name, False)
    else:
        return False

main()
