#include <string>
#include <gst/gst.h>
#include <gst/gstbuffer.h>
#include <gst/app/gstappsrc.h>
#include <glib.h>
#include <iostream>
#include <csignal>
#include "depthai/depthai.hpp"


using namespace std;

typedef std::shared_ptr<dai::DataOutputQueue> DaiOutQueue;

typedef struct pipeline_frame pframe_t;
struct pipeline_frame {
    DaiOutQueue queue;
    GstElement *source;
    GstElement *pipeline;
    GMainLoop *loop;
};

bool verbose = true;


static std::atomic<bool> alive{true};
static void sigintHandler(int signum) {
    alive = false;
}

/**
 * Handles EOS signal from the main loop. This should not be called while decoding packets synchronously
 * 
 * @param bus the GstBus main bus
 * @param message a pointer to the error message
 * @param data contains the source of the error
*/
static void on_eos(GstBus *bus, GstMessage *msg, gpointer data) {
    GMainLoop *loop = (GMainLoop *) data;
    g_print("Reached end of stream.\n");
    g_main_loop_quit(loop);
}

/**
 * Handles error messages that occur during the main loop
 * 
 * @param bus the GstBus main bus
 * @param message a pointer to the error message
 * @param data contains the source of the error
*/
static gboolean on_error(GstBus *bus, GstMessage *message, gpointer data) {
    GError *err;
    gchar *debug_info;

    gst_message_parse_error(message, &err, &debug_info);
    g_printerr("Error received from element %s: %s\n", GST_OBJECT_NAME (message->src), err->message);
    g_printerr("Debugging information: %s\n", debug_info ? debug_info : "none");
    g_error_free(err);
    g_free(debug_info);
    g_main_loop_quit((GMainLoop*)data);

    return FALSE;
}

/**
 * Handles incoming debug info from the pipeline
 * 
 * @param bus the GstBus main bus
 * @param data the incoming data, including source and message
*/
static void info (GstBus *bus, GstMessage *msg, gpointer data) {
    g_print("DEBUG INFO: %s\n", GST_OBJECT_NAME (msg->src));
}

/**
 * Handles the aborting of GST pipeline
 * 
 * @param loop the GSTMainLoop object
 * @param pipeline the GstElement *pipeline
*/
static void gst_abort(GMainLoop *loop, GstElement *pipeline) {
    GstStateChangeReturn ret;
    ret = gst_element_set_state(pipeline, GST_STATE_NULL);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        fprintf(stderr, "Failed to set GStreamer pipeline to NULL state...\n");
        fprintf(stderr, "Continuing to abort...\n");
    }
    g_main_loop_quit(loop);
}

/** 
 * This function frees GstBuffers once they are flagged as unused
 * 
 * @param data contains the buffer to be freed
*/
static void buffer_free_notify(gpointer data) {
    g_free(data);
}

/**
 * This callback function handles the wrapping of oakd h265 packets into 
 * buffers and pushes them through the GStreamer pipeline
 * 
 * @param data contains a pointer to a pframe_t 
 * @return a gboolean indicating whether the main loop should proceed 
*/
static gboolean decode(gpointer data) {
    pframe_t *frame = (pframe_t *)(data);
    if (!alive) {
        fprintf(stderr, "\nExiting (KeyboardInterrupt)...\n");
        gst_abort(frame->loop, frame->pipeline);
        return FALSE;
    }
    DaiOutQueue q = frame->queue;
    GstElement *source = frame->source;

    // Get the next H.265 frame from the output queue
    auto packet_ptr = q->tryGet<dai::ImgFrame>();
    if (!packet_ptr) {
        return TRUE;
    }

    /* Copy the contents of the packet to a gchar* */
    auto packet = *packet_ptr;
    auto p_info = packet.getData();
    const auto& d = packet.getData().data();
    gchar *packet_data = (gchar*) g_memdup(d, p_info.size());

    if (verbose) {
        // Compute and print the latency
        auto now = std::chrono::steady_clock::now();
        auto timestamp = std::chrono::time_point_cast<std::chrono::milliseconds>(packet.getTimestamp()).time_since_epoch().count();
        auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() - timestamp;
        fprintf(stderr, "Latency: %zu ms", latency);
        fprintf(stderr, " | Packet size: %zu bytes", p_info.size());
    }

    // Construct a wrapped GstBuffer from the frame data, set free notification
    GstBuffer *buffer = gst_buffer_new_wrapped_full(GST_MEMORY_FLAG_READONLY, packet_data, p_info.size(), 0, p_info.size(), NULL, &buffer_free_notify);

    // Push the buffer to the appsrc element
    GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(source), buffer);

    if (ret != GST_FLOW_OK) {
        g_printerr("Error pushing buffer to appsrc, flow state: %s\n", gst_flow_get_name(ret));
        g_free(packet_data);
        return FALSE;
    }
    else if (verbose) {
        fprintf(stderr, " | Packet pushed with status: %s\n", gst_flow_get_name(ret));
    }

    // Return TRUE to keep the loop running
    return TRUE;
}



/**
 * Initializes OAKD and GStreamer pipeline, runs GStreamer main loop
*/
int main(int argc, char *argv[]) {
    std::signal(SIGINT, &sigintHandler);

    // Create dai pipeline
    dai::Pipeline dai_pipeline;

    // Define sources and outputs
    auto camRgb = dai_pipeline.create<dai::node::ColorCamera>();
    auto videoEnc = dai_pipeline.create<dai::node::VideoEncoder>();
    auto xout = dai_pipeline.create<dai::node::XLinkOut>();

    xout->setStreamName("h265");

    int fps = 10
    camRgb->setBoardSocket(dai::CameraBoardSocket::RGB);
    videoEnc->setDefaultProfilePreset(fps, dai::VideoEncoderProperties::Profile::H264_MAIN);

    // Linking
    camRgb->video.link(videoEnc->input);
    videoEnc->bitstream.link(xout->input);

    // Connect to device and start pipeline
    dai::Device device(dai_pipeline);

    if(verbose) {
        fprintf(stderr, "Initialized OAK-D Pipeline...\n");
    }

    // Output queue will be used to get the encoded data from the output defined above
    int queue_size = 30;
    auto q = device.getOutputQueue("h264", queue_size, true);

    cout << "Press Ctrl+C to exit" << endl;

    /* Initialize GStreamer */
    gst_init(NULL, NULL);
    GMainLoop *loop = g_main_loop_new(NULL, FALSE);
    GstBus *bus;

    const char *cmdline = "appsrc name=in ! h264parse name=parse ! avdec_h264 name=avdec ! queue name=q ! autovideosink name=out";
    GError *error = NULL;

    // Parse the GStreamer pipeline string
    GstElement *pipeline = gst_parse_launch(cmdline, &error);

    if (!pipeline) {
        std::cerr << "Parse error: " << error->message << std::endl;
        g_clear_error(&error);
        return 1;
    }

    /* Set up the frame for the callback function */
    GstElement *source = gst_bin_get_by_name (GST_BIN (pipeline), "in");
    GstElement *parser = gst_bin_get_by_name (GST_BIN (pipeline), "parse");
    GstElement *decoder = gst_bin_get_by_name (GST_BIN (pipeline), "avdec");
    GstElement *queue = gst_bin_get_by_name (GST_BIN (pipeline), "q");
    GstElement *sink = gst_bin_get_by_name (GST_BIN (pipeline), "out");

    pframe_t *frame = (pframe_t *)calloc(1, sizeof(pframe_t));
    frame->queue = q;
    frame->loop = loop;
    frame->pipeline = pipeline;
    frame->source = source;

    /* Add callback function to retrieve & decode OAK-D frames synchronously */
    guint timeout_id = g_timeout_add(1000/fps, decode, frame);

    /* Set up bus for handling messages */
    bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
    gst_bus_add_signal_watch (bus);
    g_signal_connect (bus, "message::eos", G_CALLBACK (on_eos), loop);
    g_signal_connect (bus, "message::error", (GCallback)on_error, loop);
    g_signal_connect (bus, "message::info", (GCallback)info, loop);

    /* Set the pipeline to the playing state */
    if(verbose) {
        fprintf(stderr, "Setting GStreamer pipeline to PLAY...\n");
    }
    GstStateChangeReturn ret;
    ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        fprintf(stderr, "Failed to initialize GStreamer pipeline...\n");
    }
    else if (verbose) {
        fprintf(stderr, "Successfully changed GStreamer pipeline to state %s\n", gst_element_state_change_return_get_name(ret));
        fprintf(stderr, "Running the main loop...\n");
    }

    /* Run the main loop */
    g_main_loop_run(loop);

    /* Free elements */
    if(verbose) fprintf(stderr, "Freeing pipeline elements...\n");
    g_object_unref(pipeline);
    g_object_unref(source);
    g_object_unref(parser);
    g_object_unref(decoder);
    g_object_unref(queue);
    g_object_unref(sink);
    g_object_unref(bus);
    g_main_loop_unref(loop);
    free(frame);

    if(verbose) fprintf(stderr, "Pipeline elements freed.\n");
    return 0;
}   
