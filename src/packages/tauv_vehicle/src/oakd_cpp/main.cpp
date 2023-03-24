#include <ros/ros.h>
#include <image_transport/image_transport.h>

#include <Pipeline.h>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "compressed_streaming_node");
    ros::NodeHandle nh;
    ros::spin();
}