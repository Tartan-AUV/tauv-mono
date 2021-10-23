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

class ControllerCmd {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.a_x = null;
      this.a_y = null;
      this.a_z = null;
      this.a_yaw = null;
      this.p_roll = null;
      this.p_pitch = null;
    }
    else {
      if (initObj.hasOwnProperty('a_x')) {
        this.a_x = initObj.a_x
      }
      else {
        this.a_x = 0.0;
      }
      if (initObj.hasOwnProperty('a_y')) {
        this.a_y = initObj.a_y
      }
      else {
        this.a_y = 0.0;
      }
      if (initObj.hasOwnProperty('a_z')) {
        this.a_z = initObj.a_z
      }
      else {
        this.a_z = 0.0;
      }
      if (initObj.hasOwnProperty('a_yaw')) {
        this.a_yaw = initObj.a_yaw
      }
      else {
        this.a_yaw = 0.0;
      }
      if (initObj.hasOwnProperty('p_roll')) {
        this.p_roll = initObj.p_roll
      }
      else {
        this.p_roll = 0.0;
      }
      if (initObj.hasOwnProperty('p_pitch')) {
        this.p_pitch = initObj.p_pitch
      }
      else {
        this.p_pitch = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ControllerCmd
    // Serialize message field [a_x]
    bufferOffset = _serializer.float32(obj.a_x, buffer, bufferOffset);
    // Serialize message field [a_y]
    bufferOffset = _serializer.float32(obj.a_y, buffer, bufferOffset);
    // Serialize message field [a_z]
    bufferOffset = _serializer.float32(obj.a_z, buffer, bufferOffset);
    // Serialize message field [a_yaw]
    bufferOffset = _serializer.float32(obj.a_yaw, buffer, bufferOffset);
    // Serialize message field [p_roll]
    bufferOffset = _serializer.float32(obj.p_roll, buffer, bufferOffset);
    // Serialize message field [p_pitch]
    bufferOffset = _serializer.float32(obj.p_pitch, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ControllerCmd
    let len;
    let data = new ControllerCmd(null);
    // Deserialize message field [a_x]
    data.a_x = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [a_y]
    data.a_y = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [a_z]
    data.a_z = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [a_yaw]
    data.a_yaw = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [p_roll]
    data.p_roll = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [p_pitch]
    data.p_pitch = _deserializer.float32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 24;
  }

  static datatype() {
    // Returns string type for a message object
    return 'tauv_msgs/ControllerCmd';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'c0612f34c73db057150c241e2726f1e9';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float32 a_x  # Acceleration in fixed frame x direction
    float32 a_y  # Acceleration in fixed frame y direction
    float32 a_z  # Acceleration in fixed frame z direction
    float32 a_yaw  # Acceleration in fixed frame yaw direction
    float32 p_roll  # roll target
    float32 p_pitch  # pitch target
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new ControllerCmd(null);
    if (msg.a_x !== undefined) {
      resolved.a_x = msg.a_x;
    }
    else {
      resolved.a_x = 0.0
    }

    if (msg.a_y !== undefined) {
      resolved.a_y = msg.a_y;
    }
    else {
      resolved.a_y = 0.0
    }

    if (msg.a_z !== undefined) {
      resolved.a_z = msg.a_z;
    }
    else {
      resolved.a_z = 0.0
    }

    if (msg.a_yaw !== undefined) {
      resolved.a_yaw = msg.a_yaw;
    }
    else {
      resolved.a_yaw = 0.0
    }

    if (msg.p_roll !== undefined) {
      resolved.p_roll = msg.p_roll;
    }
    else {
      resolved.p_roll = 0.0
    }

    if (msg.p_pitch !== undefined) {
      resolved.p_pitch = msg.p_pitch;
    }
    else {
      resolved.p_pitch = 0.0
    }

    return resolved;
    }
};

module.exports = ControllerCmd;
