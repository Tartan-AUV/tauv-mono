import cv2 
import numpy as np 
import sys 

class gateDetector: 
    def __init__(self):
        self.numBits = 8
        self.imageWidth = 640
        self.imageHeight = 480
        self.maxVal = 2**self.numBits - 1 
        
    
    def openImage (self, path):
        img = cv2.imread(path)
        return img 
    def enhanceRedChroma(self,image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        Y, U, V = cv2.split(image)  

        scaleU = 1
        scaleV = 3

        midPoint = (2**(self.numBits-1)) - 1

        meanU = U.mean()
        offsetU = midPoint-scaleU*meanU 
        newU = cv2.convertScaleAbs(U, alpha=scaleU, beta=offsetU)

        meanV = V.mean()
        offsetV = midPoint-scaleV*meanV  
        newV = cv2.convertScaleAbs(V, alpha=scaleV, beta=offsetV)


        new_image = cv2.merge([Y, newU, newV])
        new_image = cv2.cvtColor(new_image,cv2.COLOR_YUV2BGR)
        return new_image
    
    def showImage(self, img1, img2):
        cv2.imshow('Original Gate', img1)
        cv2.imshow('Different Gate', img2)

        cv2.waitKey(0) # waits until a key is pressed
        cv2.destroyAllWindows() # destroys the window showing image
    def increaseContrast(self,image):
        scale = 2

        midPoint = (2**(self.numBits-1)) - 1

        B, G, R = cv2.split(image)

        meanB = B.mean()
        offsetB = midPoint-scale*meanB 
        newB = cv2.convertScaleAbs(B, alpha=scale, beta=offsetB)

        meanG = G.mean()
        offsetG = midPoint-scale*meanG 
        newG = cv2.convertScaleAbs(G, alpha=scale, beta=offsetG)
 

        meanR = R.mean()
        offsetR = midPoint-scale*meanR  
        newR = cv2.convertScaleAbs(R, alpha=scale, beta=offsetR)

        new_image = cv2.merge([newB, newG, newR])
        return new_image
    def getBinary(self, img):
        blurDim = self.imageHeight//8
        if blurDim % 2 == 0: 
            blurDim = blurDim + 1
        blurImg = cv2.medianBlur(img, blurDim)

        colorBinaryImg = self.maxVal-cv2.absdiff(img, blurImg)
        gaussDim = blurDim//2
        if gaussDim % 2 == 0: 
            gaussDim = gaussDim + 1       
        gaussBlurImg = cv2.GaussianBlur(colorBinaryImg, (gaussDim, gaussDim), 0)
        grayImg = cv2.cvtColor(gaussBlurImg, cv2.COLOR_BGR2GRAY)
        
        threshImg = cv2.adaptiveThreshold(grayImg, self.maxVal,
             cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, gaussDim, 2)
        binImg = cv2.bitwise_not(threshImg)
        binImg = cv2.dilate(binImg,np.ones((5,1)),iterations = 2)
        binImg = cv2.erode(binImg,np.ones((5,1)),iterations = 2)
        binImg = cv2.dilate(binImg,np.ones((1,3)),iterations = 2)
        binImg = cv2.erode(binImg,np.ones((1,3)),iterations = 2)



        return binImg
    def getBars(self, img):
        barWidth = self.imageWidth//60
        columnSum = np.sum(img, axis = 0)
        firstBar = -1 
        firstMinVal = float('inf')
        for i, elem in enumerate(columnSum):
            if elem < firstMinVal: 
                firstBar = i 
                firstMinVal = elem 
        secondMinVal = float('inf')
        for i, elem in enumerate(columnSum):
            if elem < secondMinVal and abs(i - firstBar) > barWidth: 
                secondBar = i 
                secondMinVal = elem 
        if firstBar > secondBar: 
            firstBar, secondBar = secondBar, firstBar
        return (firstBar, secondBar)

    def findPost(self, img):
        (height, width, channels) = img.shape
        self.imageHeight = height 
        self.imageWidth = width
        yuvImg = self.enhanceRedChroma(img)
        contrastImg = self.increaseContrast(yuvImg)
        binaryImg = self.getBinary(contrastImg)
        (leftBar, rightBar) = self.getBars(binaryImg)
        # print(leftBar, rightBar)
        # self.showImage(img, binaryImg)

        return (leftBar, rightBar)

def main():
    imagePath = sys.argv[1]
    myGateDetector = gateDetector()
    myImg = myGateDetector.openImage(imagePath)
    (leftBar, rightBar) = myGateDetector.findPost(myImg)
    return (leftBar, rightBar)
if __name__ == '__main__':
    main()