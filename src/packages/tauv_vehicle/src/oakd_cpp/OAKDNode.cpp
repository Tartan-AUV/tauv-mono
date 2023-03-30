// Gleb Ryabtsev, 2023

#include "OAKDNode.hpp"
#include <iostream>

OAKDNode::OAKDNode()
{
    
}

OAKDNode::~OAKDNode()
{
    
}

void OAKDNode::start()
{
    dai::Pipeline pipeline;

    // Define sources and outputs
    auto color = pipeline.create<dai::node::ColorCamera>();
    auto left = pipeline.create<dai::node::MonoCamera>();
    auto right = pipeline.create<dai::node::MonoCamera>();

    auto ve_color = pipeline.create<dai::node::VideoEncoder>();
    auto ve_depth = pipeline.create<dai::node::VideoEncoder>();

    auto out_color = pipeline.create<dai::node::XLinkOut>();
    auto out_depth = pipeline.create<dai::node::XLinkOut>();

    out_color->setStreamName("outColor");
    out_depth->setStreamName("outDepth");

    // Properties
    color->setBoardSocket(dai::CameraBoardSocket::RGB);
    left->setBoardSocket(dai::CameraBoardSocket::LEFT);
    right->setBoardSocket(dai::CameraBoardSocket::RIGHT);

    // Create encoders
    ve_color->setDefaultProfilePreset(30, dai::VideoEncoderProperties::Profile::H265_MAIN);
    ve_depth->setDefaultProfilePreset(30, dai::VideoEncoderProperties::Profile::H265_MAIN);

    // Linking
    color->out.link(ve_color->input);
    ve_color->bitstream.link(out_color->input);

    // Connect to device and start pipeline
    dai::Device device(pipeline);

    // Get output queues
    auto queue_color = device.getOutputQueue("outColor");

    while (true) { // todo: ros shut down
        auto frame_color = queue_color->get<dai::ImgFrame>();
        cout << "new color frame\n";
    }
}

void OAKDNode::load_config()
{

}
