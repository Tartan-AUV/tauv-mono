## This is the starting folder for starting the yolov4 training process. 

## Includes
- /darknet/ is a cloned darknet Git repository (https://github.com/AlexeyAB/darknet)
- /obj/ is an empty folder that you will fill with files and compress
- /training/ is an empty folder where weights will be saved
- /obj.data is an object data file template
- /obj.names is an object name file template
- /process.py is the training processing file
- /yolov4-custom.cfg is custom config file copied from darknet repository

## Directions
1. Follow the directions in the 'labeling' directory in the TAUV-Tools repository to generate contents of the 'obj' folder. Compress this to 'obj.zip'. 
2. Make adjustments to 'yolov4-custom.cfg' indicated by # comments. Change lines max_batches, steps, classes (x3 at each [yolo] layer), filters (x3 at each [convolutional] layer).
3. Edit 'obj.names' template file. Each class name is on a new line in the same order as in the 'class_list.txt' file in 'obj.zip'.  
4. Edit 'obj.data' template file. Fill in the number of classes.
5. Ensure that the 'process.py' has the correct format (currently .jpg) matching training data. 
6. Edit '/darknet/Makefile' to enable the following: OPENCV=1, GPU=1, CUDNN=1, CUDNN_HALF=1, LIBSO=1.
7. Run command ```make``` to build darknet. 
8. ...
9. Follow the following commands in your local directory to set up the training. 

## Training 


