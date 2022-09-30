#include <ros/ros.h>
#include <image_transport/image_transport.h>

#include <Pipeline.h>

Pipeline *p;

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  p->pushImage(msg);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "compressed_streaming_node");
  ros::NodeHandle nh;

  std::string pipelineConfig;
  nh.getParam("pipeline_config", pipelineConfig);
  if(pipelineConfig.empty()) {
    std::cerr << "Pipeline configuration missing.\n";
    return 1;
  }

  std::string imageTopic;
  nh.getParam("image_topic", imageTopic);

  p = new Pipeline(pipelineConfig);
  p->start();

  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub = it.subscribe(imageTopic, 1, imageCallback);
  ros::spin();
}