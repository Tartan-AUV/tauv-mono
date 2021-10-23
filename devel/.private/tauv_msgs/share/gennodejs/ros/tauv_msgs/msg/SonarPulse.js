// Auto-generated. Do not edit!

// (in-package tauv_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class SonarPulse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.mode = null;
      this.gain_setting = null;
      this.angle = null;
      this.transmit_duration = null;
      this.sample_period = null;
      this.transmit_frequency = null;
      this.number_of_samples = null;
      this.data_length = null;
      this.data = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('mode')) {
        this.mode = initObj.mode
      }
      else {
        this.mode = 0;
      }
      if (initObj.hasOwnProperty('gain_setting')) {
        this.gain_setting = initObj.gain_setting
      }
      else {
        this.gain_setting = 0;
      }
      if (initObj.hasOwnProperty('angle')) {
        this.angle = initObj.angle
      }
      else {
        this.angle = 0.0;
      }
      if (initObj.hasOwnProperty('transmit_duration')) {
        this.transmit_duration = initObj.transmit_duration
      }
      else {
        this.transmit_duration = 0.0;
      }
      if (initObj.hasOwnProperty('sample_period')) {
        this.sample_period = initObj.sample_period
      }
      else {
        this.sample_period = 0.0;
      }
      if (initObj.hasOwnProperty('transmit_frequency')) {
        this.transmit_frequency = initObj.transmit_frequency
      }
      else {
        this.transmit_frequency = 0.0;
      }
      if (initObj.hasOwnProperty('number_of_samples')) {
        this.number_of_samples = initObj.number_of_samples
      }
      else {
        this.number_of_samples = 0;
      }
      if (initObj.hasOwnProperty('data_length')) {
        this.data_length = initObj.data_length
      }
      else {
        this.data_length = 0;
      }
      if (initObj.hasOwnProperty('data')) {
        this.data = initObj.data
      }
      else {
        this.data = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type SonarPulse
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [mode]
    bufferOffset = _serializer.uint8(obj.mode, buffer, bufferOffset);
    // Serialize message field [gain_setting]
    bufferOffset = _serializer.uint8(obj.gain_setting, buffer, bufferOffset);
    // Serialize message field [angle]
    bufferOffset = _serializer.float32(obj.angle, buffer, bufferOffset);
    // Serialize message field [transmit_duration]
    bufferOffset = _serializer.float32(obj.transmit_duration, buffer, bufferOffset);
    // Serialize message field [sample_period]
    bufferOffset = _serializer.float32(obj.sample_period, buffer, bufferOffset);
    // Serialize message field [transmit_frequency]
    bufferOffset = _serializer.float32(obj.transmit_frequency, buffer, bufferOffset);
    // Serialize message field [number_of_samples]
    bufferOffset = _serializer.uint16(obj.number_of_samples, buffer, bufferOffset);
    // Serialize message field [data_length]
    bufferOffset = _serializer.uint16(obj.data_length, buffer, bufferOffset);
    // Serialize message field [data]
    bufferOffset = _arraySerializer.uint8(obj.data, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type SonarPulse
    let len;
    let data = new SonarPulse(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [mode]
    data.mode = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [gain_setting]
    data.gain_setting = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [angle]
    data.angle = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [transmit_duration]
    data.transmit_duration = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [sample_period]
    data.sample_period = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [transmit_frequency]
    data.transmit_frequency = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [number_of_samples]
    data.number_of_samples = _deserializer.uint16(buffer, bufferOffset);
    // Deserialize message field [data_length]
    data.data_length = _deserializer.uint16(buffer, bufferOffset);
    // Deserialize message field [data]
    data.data = _arrayDeserializer.uint8(buffer, bufferOffset, null)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    length += object.data.length;
    return length + 26;
  }

  static datatype() {
    // Returns string type for a message object
    return 'tauv_msgs/SonarPulse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '5a4534993e7634b0f0c3a4e8eec771be';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    Header header
    
    uint8 mode
    uint8 gain_setting
    float32 angle
    float32 transmit_duration
    float32 sample_period
    float32 transmit_frequency
    uint16 number_of_samples
    
    uint16 data_length
    uint8[] data
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
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new SonarPulse(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.mode !== undefined) {
      resolved.mode = msg.mode;
    }
    else {
      resolved.mode = 0
    }

    if (msg.gain_setting !== undefined) {
      resolved.gain_setting = msg.gain_setting;
    }
    else {
      resolved.gain_setting = 0
    }

    if (msg.angle !== undefined) {
      resolved.angle = msg.angle;
    }
    else {
      resolved.angle = 0.0
    }

    if (msg.transmit_duration !== undefined) {
      resolved.transmit_duration = msg.transmit_duration;
    }
    else {
      resolved.transmit_duration = 0.0
    }

    if (msg.sample_period !== undefined) {
      resolved.sample_period = msg.sample_period;
    }
    else {
      resolved.sample_period = 0.0
    }

    if (msg.transmit_frequency !== undefined) {
      resolved.transmit_frequency = msg.transmit_frequency;
    }
    else {
      resolved.transmit_frequency = 0.0
    }

    if (msg.number_of_samples !== undefined) {
      resolved.number_of_samples = msg.number_of_samples;
    }
    else {
      resolved.number_of_samples = 0
    }

    if (msg.data_length !== undefined) {
      resolved.data_length = msg.data_length;
    }
    else {
      resolved.data_length = 0
    }

    if (msg.data !== undefined) {
      resolved.data = msg.data;
    }
    else {
      resolved.data = []
    }

    return resolved;
    }
};

module.exports = SonarPulse;
