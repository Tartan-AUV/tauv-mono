# Video Extraction
This folder provides the files necessary to convert rosbag data to mp4 videos as the first step in the data processing pipeline for YOLO object detection training.

## Prerequisites
In order for the scripts to work it's necessary to install the following libraries and packages with `pip install`.
* `numpy`
* `pyrosenv`
* `roslibpy`
* `opencv-python`
* `opencv-contrib-python`
* `pyyaml`
* `ffmpeg`

## Input
Follow the directions in the "[Image Collection and Training](https://github.com/Tartan-AUV/TAUV-Tools/wiki#image-collection-and-training)" Wiki from the "[Retrieve rosbag data](https://github.com/Tartan-AUV/TAUV-Tools/wiki#retrieve-rosbag-data)" section to ensure your input data is formatted correctly.

The "[Convert bags to mp4 videos](https://github.com/Tartan-AUV/TAUV-Tools/wiki#convert-bags-to-mp4-videos)" section outlines how this folder is used in the data processing pipeline.
> `python3 export_images.py <INPUT_DIR> --output_dir <OUTPUT_DIR>`

## Output
Running the `export_images` script as described above produces a directory for each of the collected rosbags containing a subdirectory of .png images and an .mp4 video for each of the collected rostopics (usually darknet detections, depth, and color from front and bottom cameras).

Continue onto the "[Make yaml files](https://github.com/Tartan-AUV/TAUV-Tools/wiki#make-yaml-files)" section of the "[Image Collection and Training](https://github.com/Tartan-AUV/TAUV-Tools/wiki#image-collection-and-training)" Wiki to proceed with the remaining steps in the data processing pipeline. 



