// Auto-generated. Do not edit!

// (in-package tauv_common.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------


//-----------------------------------------------------------

class ThrusterManagerInfoRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
    }
    else {
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ThrusterManagerInfoRequest
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ThrusterManagerInfoRequest
    let len;
    let data = new ThrusterManagerInfoRequest(null);
    return data;
  }

  static getMessageSize(object) {
    return 0;
  }

  static datatype() {
    // Returns string type for a service object
    return 'tauv_common/ThrusterManagerInfoRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'd41d8cd98f00b204e9800998ecf8427e';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    # Copyright (c) 2016 The UUV Simulator Authors.
    # All rights reserved.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new ThrusterManagerInfoRequest(null);
    return resolved;
    }
};

class ThrusterManagerInfoResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.n_thrusters = null;
      this.allocation_matrix = null;
      this.reference_frame = null;
    }
    else {
      if (initObj.hasOwnProperty('n_thrusters')) {
        this.n_thrusters = initObj.n_thrusters
      }
      else {
        this.n_thrusters = 0;
      }
      if (initObj.hasOwnProperty('allocation_matrix')) {
        this.allocation_matrix = initObj.allocation_matrix
      }
      else {
        this.allocation_matrix = [];
      }
      if (initObj.hasOwnProperty('reference_frame')) {
        this.reference_frame = initObj.reference_frame
      }
      else {
        this.reference_frame = '';
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ThrusterManagerInfoResponse
    // Serialize message field [n_thrusters]
    bufferOffset = _serializer.int32(obj.n_thrusters, buffer, bufferOffset);
    // Serialize message field [allocation_matrix]
    bufferOffset = _arraySerializer.float64(obj.allocation_matrix, buffer, bufferOffset, null);
    // Serialize message field [reference_frame]
    bufferOffset = _serializer.string(obj.reference_frame, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ThrusterManagerInfoResponse
    let len;
    let data = new ThrusterManagerInfoResponse(null);
    // Deserialize message field [n_thrusters]
    data.n_thrusters = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [allocation_matrix]
    data.allocation_matrix = _arrayDeserializer.float64(buffer, bufferOffset, null)
    // Deserialize message field [reference_frame]
    data.reference_frame = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += 8 * object.allocation_matrix.length;
    length += _getByteLength(object.reference_frame);
    return length + 12;
  }

  static datatype() {
    // Returns string type for a service object
    return 'tauv_common/ThrusterManagerInfoResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '66fb8ab2f9c5649d97263c955edb636e';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    int32 n_thrusters
    float64[] allocation_matrix
    string reference_frame
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new ThrusterManagerInfoResponse(null);
    if (msg.n_thrusters !== undefined) {
      resolved.n_thrusters = msg.n_thrusters;
    }
    else {
      resolved.n_thrusters = 0
    }

    if (msg.allocation_matrix !== undefined) {
      resolved.allocation_matrix = msg.allocation_matrix;
    }
    else {
      resolved.allocation_matrix = []
    }

    if (msg.reference_frame !== undefined) {
      resolved.reference_frame = msg.reference_frame;
    }
    else {
      resolved.reference_frame = ''
    }

    return resolved;
    }
};

module.exports = {
  Request: ThrusterManagerInfoRequest,
  Response: ThrusterManagerInfoResponse,
  md5sum() { return '66fb8ab2f9c5649d97263c955edb636e'; },
  datatype() { return 'tauv_common/ThrusterManagerInfo'; }
};
