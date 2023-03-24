#ifndef PIPELINE_H
#define PIPELINE_H

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>

#include <image_transport/image_transport.h>

#include <thread>

class Pipeline {
public:
    Pipeline(const std::string &pipelineConfig);

    void start();

    void pushImage(const sensor_msgs::ImageConstPtr& msg);

private:
    GMainLoop *loop;
    GstElement *pipeline;
    GstElement *src;
    GstBus *bus;

    std::thread *gstThread;

    sensor_msgs::ImageConstPtr imagePtr; // boost shared pointer to message data

    static void freeImageMemory(void *imagePtr);
};

#endif