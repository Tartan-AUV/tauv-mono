#ifndef PIPELINE_H
#define PIPELINE_H

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
// #include <gst/app/gstappbuffer.h>
#include <gst/gstbuffer.h>
#include <image_transport/image_transport.h>

#include <thread>

class Pipeline {
public:
    Pipeline();

    void start();

    void pushImage(const void* data, size_t size, int width, int height);

private:
    GMainLoop *loop;
    GstElement *pipeline;
    GstElement *src;
    GstElement *sink;
    GstBus *bus;

    std::thread *gstThread;

    sensor_msgs::ImageConstPtr imagePtr; // boost shared pointer to message data

    static void freeImageMemory(void *imagePtr);

    static void new_sample(GstAppSink *sink, gpointer data);
};

#endif