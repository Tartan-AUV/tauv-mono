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

class InertialVals {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.mass = null;
      this.buoyancy = null;
      this.ixx = null;
      this.iyy = null;
      this.izz = null;
    }
    else {
      if (initObj.hasOwnProperty('mass')) {
        this.mass = initObj.mass
      }
      else {
        this.mass = 0.0;
      }
      if (initObj.hasOwnProperty('buoyancy')) {
        this.buoyancy = initObj.buoyancy
      }
      else {
        this.buoyancy = 0.0;
      }
      if (initObj.hasOwnProperty('ixx')) {
        this.ixx = initObj.ixx
      }
      else {
        this.ixx = 0.0;
      }
      if (initObj.hasOwnProperty('iyy')) {
        this.iyy = initObj.iyy
      }
      else {
        this.iyy = 0.0;
      }
      if (initObj.hasOwnProperty('izz')) {
        this.izz = initObj.izz
      }
      else {
        this.izz = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type InertialVals
    // Serialize message field [mass]
    bufferOffset = _serializer.float32(obj.mass, buffer, bufferOffset);
    // Serialize message field [buoyancy]
    bufferOffset = _serializer.float32(obj.buoyancy, buffer, bufferOffset);
    // Serialize message field [ixx]
    bufferOffset = _serializer.float32(obj.ixx, buffer, bufferOffset);
    // Serialize message field [iyy]
    bufferOffset = _serializer.float32(obj.iyy, buffer, bufferOffset);
    // Serialize message field [izz]
    bufferOffset = _serializer.float32(obj.izz, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type InertialVals
    let len;
    let data = new InertialVals(null);
    // Deserialize message field [mass]
    data.mass = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [buoyancy]
    data.buoyancy = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [ixx]
    data.ixx = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [iyy]
    data.iyy = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [izz]
    data.izz = _deserializer.float32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 20;
  }

  static datatype() {
    // Returns string type for a message object
    return 'tauv_msgs/InertialVals';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'dc905a6a26bfe30465ae55cdfc3db94e';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float32 mass
    float32 buoyancy
    float32 ixx
    float32 iyy
    float32 izz
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new InertialVals(null);
    if (msg.mass !== undefined) {
      resolved.mass = msg.mass;
    }
    else {
      resolved.mass = 0.0
    }

    if (msg.buoyancy !== undefined) {
      resolved.buoyancy = msg.buoyancy;
    }
    else {
      resolved.buoyancy = 0.0
    }

    if (msg.ixx !== undefined) {
      resolved.ixx = msg.ixx;
    }
    else {
      resolved.ixx = 0.0
    }

    if (msg.iyy !== undefined) {
      resolved.iyy = msg.iyy;
    }
    else {
      resolved.iyy = 0.0
    }

    if (msg.izz !== undefined) {
      resolved.izz = msg.izz;
    }
    else {
      resolved.izz = 0.0
    }

    return resolved;
    }
};

module.exports = InertialVals;
