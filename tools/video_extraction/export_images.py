import os
import argparse
import time
# import tarfile

import cv2

import rosbag
from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
import yaml

import numpy as np

# import ros_numpy

# from cv_bridge.boost.cv_bridge_boost import cvtColor2

from encoding_to_dtypes import image_to_numpy

lastiter = None
iter_p_sec = None

def printProgressBar(iteration, total, prefix = 'Progress:', suffix = 'Complete', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
    global lastiter
    global iter_p_sec

    eta = -1.0
    now = time.time()
    tau = 60
    if lastiter is not None:
        dt = now - lastiter
        iter_p_sec_instant = 1/dt
        if iter_p_sec is None:
            iter_p_sec = iter_p_sec_instant
        alpha = dt / tau
        iter_p_sec = alpha * iter_p_sec_instant + (1-alpha) * iter_p_sec
        eta = (total - iteration) / iter_p_sec
    lastiter = now
    if total == 0:
        percent = 0
        filledLength = 0
    else:
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix} | {eta:.1f}s remaining', end = printEnd)

    if iteration == total: 
        print()


def tar_each_topic(bag_dir):
    for topic in os.listdir(bag_dir):
        if os.path.isdir(os.path.join(bag_dir, (topic))):
            bag_name = os.path.basename(os.path.normpath(bag_dir))
            print(bag_name)
            topic_dir = os.path.join(bag_dir, (topic))
            print(f'Taring {topic_dir}')
            print (f'{topic_dir+ "_" + bag_name}.tar')
            with tarfile.open(f'{topic_dir+ "_" + bag_name}.tar', "w:gz") as tar:
                tar.add(topic_dir, arcname=os.path.basename(topic_dir))


def create_video(bag_dir): 
    for topic in os.listdir(bag_dir):
        if os.path.isdir(os.path.join(bag_dir, (topic))):
            bag_name = os.path.basename(os.path.normpath(bag_dir))
            topic_dir = os.path.join(bag_dir, (topic))
            print(f'Videoing {topic_dir}')
            video_name =  (f'{topic_dir+ "_" + bag_name}.mp4')
            command_str = 'cat {}/*.png | ffmpeg -r 15 -f image2pipe -i - {}'.format(topic_dir, video_name)
            os.system(command_str)


def main():
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("bag_directory", help="Input bag directory.")
    parser.add_argument("--output_dir", help="Output directory.")

    args = parser.parse_args()
    
    if not os.path.isdir(args.output_dir):
        print(f'{args.output_dir} is not a directory')
        exit()

    if not os.path.isdir(args.bag_directory):
        print(f'{args.bag_directory} is not a directory')
        exit()
    
    for current_bag in os.listdir(args.bag_directory):
        current_path = os.path.join(args.bag_directory, current_bag)
        bag_name = os.path.splitext(os.path.basename(os.path.normpath(current_path)))

        if bag_name[1] != '.bag':
            print(f'{current_path} is not bag file')
            print('Continuing...')
            continue

        bag_path = os.path.join(args.output_dir, bag_name[0])
        if os.path.isdir(bag_path):
            print(f'{current_path} already exists in {args.output_dir}')
            print('Continuing...')
            continue

        os.mkdir(bag_path)

        print(f'Extracting images from {bag_name[0]}{bag_name[1]}')

        try:
            bag = rosbag.Bag(current_path, 'r')
            topics = yaml.load(bag._get_yaml_info(), Loader=yaml.BaseLoader)['topics']
        except KeyError:
            print('Could not get topics')
            print('Continiong...')
            continue

        video_topics = []
        total_messages = 0
        for topic in topics:
            if topic['type'] == 'sensor_msgs/Image':
                video_topics.append(topic['topic'])
                total_messages += int(topic['messages'])
        
        count = 0
        printProgressBar(count, total_messages, prefix=f'{bag_name[0]} Progress:')
        for topic, msg, t in bag.read_messages(topics=video_topics):
            folder_name = "-".join((topic.split('/')[-3:-1]))
            topic_path = os.path.join(bag_path, folder_name.replace('/', '-'))
            if not os.path.isdir(topic_path):
                os.mkdir(topic_path)

            cv_img = image_to_numpy(msg)
            cur_time = int(msg.header.stamp.nsecs) + int(msg.header.stamp.secs) * 1000000000
            cv2.imwrite(os.path.join(topic_path, f'{cur_time}.png'), cv_img)

            count += 1
            printProgressBar(count, total_messages, prefix=f'{bag_name[0]} Progress:')
        
        print('Creating video for each topic...')
        create_video(bag_path)

if __name__ == '__main__':
    main()
