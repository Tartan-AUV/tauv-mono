// Auto-generated. Do not edit!

// (in-package tauv_msgs.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let PidVals = require('../msg/PidVals.js');

//-----------------------------------------------------------


//-----------------------------------------------------------

class TunePidRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.vals = null;
    }
    else {
      if (initObj.hasOwnProperty('vals')) {
        this.vals = initObj.vals
      }
      else {
        this.vals = new PidVals();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type TunePidRequest
    // Serialize message field [vals]
    bufferOffset = PidVals.serialize(obj.vals, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type TunePidRequest
    let len;
    let data = new TunePidRequest(null);
    // Deserialize message field [vals]
    data.vals = PidVals.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 32;
  }

  static datatype() {
    // Returns string type for a service object
    return 'tauv_msgs/TunePidRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '4c7aa6b5ba42b455483bf8587b081bd9';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    tauv_msgs/PidVals vals
    
    ================================================================================
    MSG: tauv_msgs/PidVals
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
    const resolved = new TunePidRequest(null);
    if (msg.vals !== undefined) {
      resolved.vals = PidVals.Resolve(msg.vals)
    }
    else {
      resolved.vals = new PidVals()
    }

    return resolved;
    }
};

class TunePidResponse {
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
    // Serializes a message object of type TunePidResponse
    // Serialize message field [success]
    bufferOffset = _serializer.bool(obj.success, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type TunePidResponse
    let len;
    let data = new TunePidResponse(null);
    // Deserialize message field [success]
    data.success = _deserializer.bool(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'tauv_msgs/TunePidResponse';
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
    const resolved = new TunePidResponse(null);
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
  Request: TunePidRequest,
  Response: TunePidResponse,
  md5sum() { return 'c7f7e64a8df4cb6364bb9c28032e070e'; },
  datatype() { return 'tauv_msgs/TunePid'; }
};
