// Auto-generated. Do not edit!

// (in-package tauv_common.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let tauv_msgs = _finder('tauv_msgs');

//-----------------------------------------------------------


//-----------------------------------------------------------

class RegisterObjectDetectionsRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.objdets = null;
      this.detector_tag = null;
    }
    else {
      if (initObj.hasOwnProperty('objdets')) {
        this.objdets = initObj.objdets
      }
      else {
        this.objdets = [];
      }
      if (initObj.hasOwnProperty('detector_tag')) {
        this.detector_tag = initObj.detector_tag
      }
      else {
        this.detector_tag = '';
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type RegisterObjectDetectionsRequest
    // Serialize message field [objdets]
    // Serialize the length for message field [objdets]
    bufferOffset = _serializer.uint32(obj.objdets.length, buffer, bufferOffset);
    obj.objdets.forEach((val) => {
      bufferOffset = tauv_msgs.msg.BucketDetection.serialize(val, buffer, bufferOffset);
    });
    // Serialize message field [detector_tag]
    bufferOffset = _serializer.string(obj.detector_tag, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type RegisterObjectDetectionsRequest
    let len;
    let data = new RegisterObjectDetectionsRequest(null);
    // Deserialize message field [objdets]
    // Deserialize array length for message field [objdets]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.objdets = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.objdets[i] = tauv_msgs.msg.BucketDetection.deserialize(buffer, bufferOffset)
    }
    // Deserialize message field [detector_tag]
    data.detector_tag = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    object.objdets.forEach((val) => {
      length += tauv_msgs.msg.BucketDetection.getMessageSize(val);
    });
    length += _getByteLength(object.detector_tag);
    return length + 8;
  }

  static datatype() {
    // Returns string type for a service object
    return 'tauv_common/RegisterObjectDetectionsRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '2b52cb0a9177986f466eafd31fee1ddc';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    tauv_msgs/BucketDetection[] objdets
    string detector_tag
    
    ================================================================================
    MSG: tauv_msgs/BucketDetection
    Header header
    geometry_msgs/Point position
    float32 length
    float32 width
    float32 height
    geometry_msgs/Vector3 normal
    sensor_msgs/Image image
    vision_msgs/BoundingBox2D bbox_2d
    jsk_recognition_msgs/BoundingBox bbox_3d
    string tag
    uint32 detection_number
    ================================================================================
    MSG: std_msgs/Header
    # Standard metadata for higher-level stamped data types.
    # This is generally used to communicate timestamped data 
    # in a particular coordinate frame.
    # 
    # sequence ID: consecutively increasing ID 
    uint32 seq
    #Two-integer timestamp that is expressed as:
    # * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
    # * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
    # time-handling sugar is provided by the client library
    time stamp
    #Frame this data is associated with
    string frame_id
    
    ================================================================================
    MSG: geometry_msgs/Point
    # This contains the position of a point in free space
    float64 x
    float64 y
    float64 z
    
    ================================================================================
    MSG: geometry_msgs/Vector3
    # This represents a vector in free space. 
    # It is only meant to represent a direction. Therefore, it does not
    # make sense to apply a translation to it (e.g., when applying a 
    # generic rigid transformation to a Vector3, tf2 will only apply the
    # rotation). If you want your data to be translatable too, use the
    # geometry_msgs/Point message instead.
    
    float64 x
    float64 y
    float64 z
    ================================================================================
    MSG: sensor_msgs/Image
    # This message contains an uncompressed image
    # (0, 0) is at top-left corner of image
    #
    
    Header header        # Header timestamp should be acquisition time of image
                         # Header frame_id should be optical frame of camera
                         # origin of frame should be optical center of camera
                         # +x should point to the right in the image
                         # +y should point down in the image
                         # +z should point into to plane of the image
                         # If the frame_id here and the frame_id of the CameraInfo
                         # message associated with the image conflict
                         # the behavior is undefined
    
    uint32 height         # image height, that is, number of rows
    uint32 width          # image width, that is, number of columns
    
    # The legal values for encoding are in file src/image_encodings.cpp
    # If you want to standardize a new string format, join
    # ros-users@lists.sourceforge.net and send an email proposing a new encoding.
    
    string encoding       # Encoding of pixels -- channel meaning, ordering, size
                          # taken from the list of strings in include/sensor_msgs/image_encodings.h
    
    uint8 is_bigendian    # is this data bigendian?
    uint32 step           # Full row length in bytes
    uint8[] data          # actual matrix data, size is (step * rows)
    
    ================================================================================
    MSG: vision_msgs/BoundingBox2D
    # A 2D bounding box that can be rotated about its center.
    # All dimensions are in pixels, but represented using floating-point
    #   values to allow sub-pixel precision. If an exact pixel crop is required
    #   for a rotated bounding box, it can be calculated using Bresenham's line
    #   algorithm.
    
    # The 2D position (in pixels) and orientation of the bounding box center.
    geometry_msgs/Pose2D center
    
    # The size (in pixels) of the bounding box surrounding the object relative
    #   to the pose of its center.
    float64 size_x
    float64 size_y
    
    ================================================================================
    MSG: geometry_msgs/Pose2D
    # Deprecated
    # Please use the full 3D pose.
    
    # In general our recommendation is to use a full 3D representation of everything and for 2D specific applications make the appropriate projections into the plane for their calculations but optimally will preserve the 3D information during processing.
    
    # If we have parallel copies of 2D datatypes every UI and other pipeline will end up needing to have dual interfaces to plot everything. And you will end up with not being able to use 3D tools for 2D use cases even if they're completely valid, as you'd have to reimplement it with different inputs and outputs. It's not particularly hard to plot the 2D pose or compute the yaw error for the Pose message and there are already tools and libraries that can do this for you.
    
    
    # This expresses a position and orientation on a 2D manifold.
    
    float64 x
    float64 y
    float64 theta
    
    ================================================================================
    MSG: jsk_recognition_msgs/BoundingBox
    # BoundingBox represents a oriented bounding box.
    Header header
    geometry_msgs/Pose pose
    geometry_msgs/Vector3 dimensions  # size of bounding box (x, y, z)
    # You can use this field to hold value such as likelihood
    float32 value
    uint32 label
    
    ================================================================================
    MSG: geometry_msgs/Pose
    # A representation of pose in free space, composed of position and orientation. 
    Point position
    Quaternion orientation
    
    ================================================================================
    MSG: geometry_msgs/Quaternion
    # This represents an orientation in free space in quaternion form.
    
    float64 x
    float64 y
    float64 z
    float64 w
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new RegisterObjectDetectionsRequest(null);
    if (msg.objdets !== undefined) {
      resolved.objdets = new Array(msg.objdets.length);
      for (let i = 0; i < resolved.objdets.length; ++i) {
        resolved.objdets[i] = tauv_msgs.msg.BucketDetection.Resolve(msg.objdets[i]);
      }
    }
    else {
      resolved.objdets = []
    }

    if (msg.detector_tag !== undefined) {
      resolved.detector_tag = msg.detector_tag;
    }
    else {
      resolved.detector_tag = ''
    }

    return resolved;
    }
};

class RegisterObjectDetectionsResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.success = null;
    }
    else {
      if (initObj.hasOwnProperty('success')) {
        this.success = initObj.success
      }
      else {
        this.success = false;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type RegisterObjectDetectionsResponse
    // Serialize message field [success]
    bufferOffset = _serializer.bool(obj.success, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type RegisterObjectDetectionsResponse
    let len;
    let data = new RegisterObjectDetectionsResponse(null);
    // Deserialize message field [success]
    data.success = _deserializer.bool(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'tauv_common/RegisterObjectDetectionsResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '358e233cde0c8a8bcfea4ce193f8fc15';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    bool success
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new RegisterObjectDetectionsResponse(null);
    if (msg.success !== undefined) {
      resolved.success = msg.success;
    }
    else {
      resolved.success = false
    }

    return resolved;
    }
};

module.exports = {
  Request: RegisterObjectDetectionsRequest,
  Response: RegisterObjectDetectionsResponse,
  md5sum() { return '5cc8e5672e1e74a20b716902a91f95c7'; },
  datatype() { return 'tauv_common/RegisterObjectDetections'; }
};
