import os
import argparse
import multiprocessing as mp
import queue

import cv2

import rosbag
import yaml

from encoding_to_dtypes import image_to_numpy

parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
parser.add_argument("bag_directory", help="Input bag directory.")
parser.add_argument("--output_dir", help="Output directory.")

args = parser.parse_args()

NUM_OF_THREADS = int(mp.cpu_count() / 2)

def create_video(bag_dir): 
    for topic in os.listdir(bag_dir):
        if os.path.isdir(os.path.join(bag_dir, (topic))):
            bag_name = os.path.basename(os.path.normpath(bag_dir))
            topic_dir = os.path.join(bag_dir, (topic))
            print(f'Videoing {topic_dir}')
            video_name =  (f'{topic_dir+ "_" + bag_name}.mp4')
            command_str = 'cat {}/*.png | ffmpeg -loglevel panic -r 15 -f image2pipe -i - {}'.format(topic_dir, video_name)
            os.system(command_str)

def process_bag(current_bag):
    current_path = os.path.join(args.bag_directory, current_bag)
    bag_name = os.path.splitext(os.path.basename(os.path.normpath(current_path)))

    if bag_name[1] != '.bag':
        print(f'{current_path} is not bag file')
        print('Continuing...')
        return

    bag_path = os.path.join(args.output_dir, bag_name[0])
    if os.path.isdir(bag_path):
        print(f'{current_path} already exists in {args.output_dir}')
        print('Continuing...')
        return

    os.mkdir(bag_path)

    print(f'Extracting images from {bag_name[0]}{bag_name[1]}')

    bag = rosbag.Bag(current_path, 'r')
    topics = yaml.load(bag._get_yaml_info(), Loader=yaml.BaseLoader)['topics']

    video_topics = []
    for topic in topics:
        if topic['type'] == 'sensor_msgs/Image':
            video_topics.append(topic['topic'])
    
    for topic, msg, t in bag.read_messages(topics=video_topics):
        folder_name = "-".join((topic.split('/')[-3:-1]))
        topic_path = os.path.join(bag_path, folder_name.replace('/', '-'))
        if not os.path.isdir(topic_path):
            os.mkdir(topic_path)

        cv_img = image_to_numpy(msg)
        cur_time = int(msg.header.stamp.nsecs) + int(msg.header.stamp.secs) * 1000000000
        cv2.imwrite(os.path.join(topic_path, f'{cur_time}.png'), cv_img)

    print('Creating video for each topic...')
    create_video(bag_path)


def worker(input):
    while True:
        try:
            bag = input.get_nowait()
        except queue.Empty:
            break
        else:
            print('processing bag with', mp.current_process())
            process_bag(bag)
            print(f'finished with {bag}')
    return True
 

def main():
    if not os.path.isdir(args.output_dir):
        print(f'{args.output_dir} is not a directory')
        exit()

    if not os.path.isdir(args.bag_directory):
        print(f'{args.bag_directory} is not a directory')
        exit()

    input = mp.Queue()
    processes = []
    
    for current_bag in os.listdir(args.bag_directory):
        input.put(current_bag)

    for _ in range(NUM_OF_THREADS):
        p = mp.Process(target=worker, args=(input,))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()

    print("All Processes finished.")

if __name__ == '__main__':
    main()
