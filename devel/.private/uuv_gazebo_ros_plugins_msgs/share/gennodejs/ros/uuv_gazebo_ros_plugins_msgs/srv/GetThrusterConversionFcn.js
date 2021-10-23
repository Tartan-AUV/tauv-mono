// Auto-generated. Do not edit!

// (in-package uuv_gazebo_ros_plugins_msgs.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

let ThrusterConversionFcn = require('../msg/ThrusterConversionFcn.js');

//-----------------------------------------------------------

class GetThrusterConversionFcnRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
    }
    else {
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type GetThrusterConversionFcnRequest
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GetThrusterConversionFcnRequest
    let len;
    let data = new GetThrusterConversionFcnRequest(null);
    return data;
  }

  static getMessageSize(object) {
    return 0;
  }

  static datatype() {
    // Returns string type for a service object
    return 'uuv_gazebo_ros_plugins_msgs/GetThrusterConversionFcnRequest';
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
    const resolved = new GetThrusterConversionFcnRequest(null);
    return resolved;
    }
};

class GetThrusterConversionFcnResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.fcn = null;
    }
    else {
      if (initObj.hasOwnProperty('fcn')) {
        this.fcn = initObj.fcn
      }
      else {
        this.fcn = new ThrusterConversionFcn();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type GetThrusterConversionFcnResponse
    // Serialize message field [fcn]
    bufferOffset = ThrusterConversionFcn.serialize(obj.fcn, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GetThrusterConversionFcnResponse
    let len;
    let data = new GetThrusterConversionFcnResponse(null);
    // Deserialize message field [fcn]
    data.fcn = ThrusterConversionFcn.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += ThrusterConversionFcn.getMessageSize(object.fcn);
    return length;
  }

  static datatype() {
    // Returns string type for a service object
    return 'uuv_gazebo_ros_plugins_msgs/GetThrusterConversionFcnResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'b489744fdf1ea3660acd86f33ee041a7';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    uuv_gazebo_ros_plugins_msgs/ThrusterConversionFcn fcn
    
    
    ================================================================================
    MSG: uuv_gazebo_ros_plugins_msgs/ThrusterConversionFcn
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
    
    string function_name
    string[] tags
    float64[] data
    float64[] lookup_table_input
    float64[] lookup_table_output
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new GetThrusterConversionFcnResponse(null);
    if (msg.fcn !== undefined) {
      resolved.fcn = ThrusterConversionFcn.Resolve(msg.fcn)
    }
    else {
      resolved.fcn = new ThrusterConversionFcn()
    }

    return resolved;
    }
};

module.exports = {
  Request: GetThrusterConversionFcnRequest,
  Response: GetThrusterConversionFcnResponse,
  md5sum() { return 'b489744fdf1ea3660acd86f33ee041a7'; },
  datatype() { return 'uuv_gazebo_ros_plugins_msgs/GetThrusterConversionFcn'; }
};
