// Auto-generated. Do not edit!

// (in-package tauv_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class PidVals {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.a_p = null;
      this.a_i = null;
      this.a_d = null;
      this.a_sat = null;
      this.l_p = null;
      this.l_i = null;
      this.l_d = null;
      this.l_sat = null;
    }
    else {
      if (initObj.hasOwnProperty('a_p')) {
        this.a_p = initObj.a_p
      }
      else {
        this.a_p = 0.0;
      }
      if (initObj.hasOwnProperty('a_i')) {
        this.a_i = initObj.a_i
      }
      else {
        this.a_i = 0.0;
      }
      if (initObj.hasOwnProperty('a_d')) {
        this.a_d = initObj.a_d
      }
      else {
        this.a_d = 0.0;
      }
      if (initObj.hasOwnProperty('a_sat')) {
        this.a_sat = initObj.a_sat
      }
      else {
        this.a_sat = 0.0;
      }
      if (initObj.hasOwnProperty('l_p')) {
        this.l_p = initObj.l_p
      }
      else {
        this.l_p = 0.0;
      }
      if (initObj.hasOwnProperty('l_i')) {
        this.l_i = initObj.l_i
      }
      else {
        this.l_i = 0.0;
      }
      if (initObj.hasOwnProperty('l_d')) {
        this.l_d = initObj.l_d
      }
      else {
        this.l_d = 0.0;
      }
      if (initObj.hasOwnProperty('l_sat')) {
        this.l_sat = initObj.l_sat
      }
      else {
        this.l_sat = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type PidVals
    // Serialize message field [a_p]
    bufferOffset = _serializer.float32(obj.a_p, buffer, bufferOffset);
    // Serialize message field [a_i]
    bufferOffset = _serializer.float32(obj.a_i, buffer, bufferOffset);
    // Serialize message field [a_d]
    bufferOffset = _serializer.float32(obj.a_d, buffer, bufferOffset);
    // Serialize message field [a_sat]
    bufferOffset = _serializer.float32(obj.a_sat, buffer, bufferOffset);
    // Serialize message field [l_p]
    bufferOffset = _serializer.float32(obj.l_p, buffer, bufferOffset);
    // Serialize message field [l_i]
    bufferOffset = _serializer.float32(obj.l_i, buffer, bufferOffset);
    // Serialize message field [l_d]
    bufferOffset = _serializer.float32(obj.l_d, buffer, bufferOffset);
    // Serialize message field [l_sat]
    bufferOffset = _serializer.float32(obj.l_sat, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type PidVals
    let len;
    let data = new PidVals(null);
    // Deserialize message field [a_p]
    data.a_p = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [a_i]
    data.a_i = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [a_d]
    data.a_d = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [a_sat]
    data.a_sat = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [l_p]
    data.l_p = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [l_i]
    data.l_i = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [l_d]
    data.l_d = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [l_sat]
    data.l_sat = _deserializer.float32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 32;
  }

  static datatype() {
    // Returns string type for a message object
    return 'tauv_msgs/PidVals';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'd4db47770a0caf47edbb925bd3a9269a';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float32 a_p
    float32 a_i
    float32 a_d
    float32 a_sat
    float32 l_p
    float32 l_i
    float32 l_d
    float32 l_sat
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new PidVals(null);
    if (msg.a_p !== undefined) {
      resolved.a_p = msg.a_p;
    }
    else {
      resolved.a_p = 0.0
    }

    if (msg.a_i !== undefined) {
      resolved.a_i = msg.a_i;
    }
    else {
      resolved.a_i = 0.0
    }

    if (msg.a_d !== undefined) {
      resolved.a_d = msg.a_d;
    }
    else {
      resolved.a_d = 0.0
    }

    if (msg.a_sat !== undefined) {
      resolved.a_sat = msg.a_sat;
    }
    else {
      resolved.a_sat = 0.0
    }

    if (msg.l_p !== undefined) {
      resolved.l_p = msg.l_p;
    }
    else {
      resolved.l_p = 0.0
    }

    if (msg.l_i !== undefined) {
      resolved.l_i = msg.l_i;
    }
    else {
      resolved.l_i = 0.0
    }

    if (msg.l_d !== undefined) {
      resolved.l_d = msg.l_d;
    }
    else {
      resolved.l_d = 0.0
    }

    if (msg.l_sat !== undefined) {
      resolved.l_sat = msg.l_sat;
    }
    else {
      resolved.l_sat = 0.0
    }

    return resolved;
    }
};

module.exports = PidVals;
