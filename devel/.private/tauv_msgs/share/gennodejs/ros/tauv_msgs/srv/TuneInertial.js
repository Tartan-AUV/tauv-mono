// Auto-generated. Do not edit!

// (in-package tauv_msgs.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let InertialVals = require('../msg/InertialVals.js');

//-----------------------------------------------------------


//-----------------------------------------------------------

class TuneInertialRequest {
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
        this.vals = new InertialVals();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type TuneInertialRequest
    // Serialize message field [vals]
    bufferOffset = InertialVals.serialize(obj.vals, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type TuneInertialRequest
    let len;
    let data = new TuneInertialRequest(null);
    // Deserialize message field [vals]
    data.vals = InertialVals.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 20;
  }

  static datatype() {
    // Returns string type for a service object
    return 'tauv_msgs/TuneInertialRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'ab258f28927b44cb5b309830480cef99';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    tauv_msgs/InertialVals vals
    
    ================================================================================
    MSG: tauv_msgs/InertialVals
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
    const resolved = new TuneInertialRequest(null);
    if (msg.vals !== undefined) {
      resolved.vals = InertialVals.Resolve(msg.vals)
    }
    else {
      resolved.vals = new InertialVals()
    }

    return resolved;
    }
};

class TuneInertialResponse {
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
    // Serializes a message object of type TuneInertialResponse
    // Serialize message field [success]
    bufferOffset = _serializer.bool(obj.success, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type TuneInertialResponse
    let len;
    let data = new TuneInertialResponse(null);
    // Deserialize message field [success]
    data.success = _deserializer.bool(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'tauv_msgs/TuneInertialResponse';
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
    const resolved = new TuneInertialResponse(null);
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
  Request: TuneInertialRequest,
  Response: TuneInertialResponse,
  md5sum() { return 'b07a11376c5b2ae9ec7888a1a74564d4'; },
  datatype() { return 'tauv_msgs/TuneInertial'; }
};
