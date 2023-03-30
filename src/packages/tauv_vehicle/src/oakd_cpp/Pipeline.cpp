#include <Pipeline.h>
#include <iostream>

Pipeline::Pipeline() {
    gst_init(nullptr, nullptr);
    pipeline = gst_parse_launch("appsrc name=mysource  ! h265parse ! nvv4l2decoder  ! appsink name=mysink ", NULL);
    // pipeline = gst_parse_launch("appsrc name=mysource ! h264parse !  fakesink ", NULL);
    g_assert(pipeline);
    bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    g_assert(bus);
    src = gst_bin_get_by_name(GST_BIN(pipeline), "mysource");
    g_assert(src);
    // gst_app_src_set_caps(GST_APP_SRC(src), gst_caps_new_simple("video/x-h264", "stream-format", G_TYPE_STRING, "avc3", "alignment", G_TYPE_STRING, "au", NULL));

    // gst_app_src_set_caps(GST_APP_SRC(src), gst_caps_new_simple("video/x-h264", "width", G_TYPE_INT, 1920, "height", G_TYPE_INT, 1080, "framerate", GST_TYPE_FRACTION, 30, 1, NULL));
    g_object_set(G_OBJECT(src), "is-live", TRUE, "block", TRUE, NULL);


    sink = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
    g_assert(GST_IS_APP_SINK(sink));
    // g_signal_connect (sink, "new-sample", G_CALLBACK (Pipeline::new_sample), nullptr);
    GstAppSinkCallbacks callbacks = { Pipeline::new_sample, NULL, NULL };
    gst_app_sink_set_callbacks(GST_APP_SINK(sink), &callbacks, NULL, NULL);


    std::cout << "GStreamer pipeline initialized.\n";
}

void Pipeline::start() {
    loop = g_main_loop_new(nullptr, FALSE);
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    gstThread = new std::thread(g_main_loop_run, loop);
}

void Pipeline::pushImage(const void* data, size_t size, int width, int height) {
//     GstBuffer *buffer = gst_buffer_new_wrapped_full(GST_MEMORY_FLAG_READONLY,
//                                                     (gpointer) data,
//                                                     size,
//                                                     0, 
//                                                     size,
//                                                     (gpointer) data,
//                                                     Pipeline::freeImageMemory);

// //    std::cout << "Image encoding: " << imagePtr->encoding << '\n';
// //    g_assert(imagePtr->encoding == "rgb8");
//     GstCaps *caps = gst_caps_new_simple("video/x-h265",
//                                         "stream_format", G_TYPE_STRING, "byte-stream",
//                                         "alginment", G_TYPE_STRING, "au",
//                                         "width", G_TYPE_INT, width,
//                                         "height", G_TYPE_INT, height,
//                                         NULL);
//     GstSample *sample = gst_sample_new(buffer, caps, NULL, NULL);
//     gst_app_src_push_sample(GST_APP_SRC(src), sample);

    GstBuffer *buffer = gst_buffer_new_wrapped_full(GST_MEMORY_FLAG_READONLY, 
                                                (gpointer)data, 
                                                size, 0, 
                                                size, NULL, NULL);
    gst_app_src_push_buffer(GST_APP_SRC(src), buffer);

    std::cout << "Pushed\n";
}

void Pipeline::freeImageMemory(void *ptr) { // static
    std::cout << "freeImageMemory called\n";
}

void Pipeline::new_sample(GstAppSink *sink, gpointer data) {
    std::cout << "Got sample!";
}
