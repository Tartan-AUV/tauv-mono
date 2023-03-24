#include <Pipeline.h>
#include <iostream>

Pipeline::Pipeline(const std::string &pipelineConfig) {
    gst_init(nullptr, nullptr);
    pipeline = gst_parse_launch(("appsrc name=mysource ! " + pipelineConfig).c_str(), NULL);
    g_assert(pipeline);
    bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    g_assert(bus);
    src = gst_bin_get_by_name(GST_BIN(pipeline), "mysource");
    g_assert(src);
    g_assert(GST_IS_APP_SRC(src));

    //gst_app_src_set_caps(GST_APP_SRC(src), caps);


    std::cout << "GStreamer pipeline initialized.\n";
}

void Pipeline::start() {
    loop = g_main_loop_new(nullptr, FALSE);
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    gstThread = new std::thread(g_main_loop_run, loop);
}

void Pipeline::pushImage(const sensor_msgs::ImageConstPtr &img) {
    imagePtr = img;
    GstBuffer *buffer = gst_buffer_new_wrapped_full(GST_MEMORY_FLAG_READONLY,
                                                    (gpointer) img->data.data(),
                                                    img->data.size(),
                                                    0, 
                                                    img->data.size(),
                                                    (gpointer) &imagePtr,
                                                    Pipeline::freeImageMemory);

//    std::cout << "Image encoding: " << imagePtr->encoding << '\n';
//    g_assert(imagePtr->encoding == "rgb8");
    GstCaps *caps = gst_caps_new_simple("video/x-raw", 
                                        "format", G_TYPE_STRING, "BGRA",
                                        "width", G_TYPE_INT, imagePtr->width,
                                        "height", G_TYPE_INT, imagePtr->height,
                                        NULL);
    GstSample *sample = gst_sample_new(buffer, caps, NULL, NULL);
    gst_app_src_push_sample(GST_APP_SRC(src), sample);

    std::cout << "Pushed\n";
}

void Pipeline::freeImageMemory(void *ptr) { // static
    std::cout << "freeImageMemory called\n";
    auto imagePtr = static_cast<sensor_msgs::ImageConstPtr*>(ptr);
    imagePtr->reset();
}
