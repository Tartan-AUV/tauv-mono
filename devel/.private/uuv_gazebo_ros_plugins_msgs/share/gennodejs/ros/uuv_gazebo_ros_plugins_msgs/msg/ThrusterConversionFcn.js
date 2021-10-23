// Auto-generated. Do not edit!

// (in-package uuv_gazebo_ros_plugins_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class ThrusterConversionFcn {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.function_name = null;
      this.tags = null;
      this.data = null;
      this.lookup_table_input = null;
      this.lookup_table_output = null;
    }
    else {
      if (initObj.hasOwnProperty('function_name')) {
        this.function_name = initObj.function_name
      }
      else {
        this.function_name = '';
      }
      if (initObj.hasOwnProperty('tags')) {
        this.tags = initObj.tags
      }
      else {
        this.tags = [];
      }
      if (initObj.hasOwnProperty('data')) {
        this.data = initObj.data
      }
      else {
        this.data = [];
      }
      if (initObj.hasOwnProperty('lookup_table_input')) {
        this.lookup_table_input = initObj.lookup_table_input
      }
      else {
        this.lookup_table_input = [];
      }
      if (initObj.hasOwnProperty('lookup_table_output')) {
        this.lookup_table_output = initObj.lookup_table_output
      }
      else {
        this.lookup_table_output = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ThrusterConversionFcn
    // Serialize message field [function_name]
    bufferOffset = _serializer.string(obj.function_name, buffer, bufferOffset);
    // Serialize message field [tags]
    bufferOffset = _arraySerializer.string(obj.tags, buffer, bufferOffset, null);
    // Serialize message field [data]
    bufferOffset = _arraySerializer.float64(obj.data, buffer, bufferOffset, null);
    // Serialize message field [lookup_table_input]
    bufferOffset = _arraySerializer.float64(obj.lookup_table_input, buffer, bufferOffset, null);
    // Serialize message field [lookup_table_output]
    bufferOffset = _arraySerializer.float64(obj.lookup_table_output, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ThrusterConversionFcn
    let len;
    let data = new ThrusterConversionFcn(null);
    // Deserialize message field [function_name]
    data.function_name = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [tags]
    data.tags = _arrayDeserializer.string(buffer, bufferOffset, null)
    // Deserialize message field [data]
    data.data = _arrayDeserializer.float64(buffer, bufferOffset, null)
    // Deserialize message field [lookup_table_input]
    data.lookup_table_input = _arrayDeserializer.float64(buffer, bufferOffset, null)
    // Deserialize message field [lookup_table_output]
    data.lookup_table_output = _arrayDeserializer.float64(buffer, bufferOffset, null)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.function_name);
    object.tags.forEach((val) => {
      length += 4 + _getByteLength(val);
    });
    length += 8 * object.data.length;
    length += 8 * object.lookup_table_input.length;
    length += 8 * object.lookup_table_output.length;
    return length + 20;
  }

  static datatype() {
    // Returns string type for a message object
    return 'uuv_gazebo_ros_plugins_msgs/ThrusterConversionFcn';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '5cc7c4f30276fbc995f2325f64846776';
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
    const resolved = new ThrusterConversionFcn(null);
    if (msg.function_name !== undefined) {
      resolved.function_name = msg.function_name;
    }
    else {
      resolved.function_name = ''
    }

    if (msg.tags !== undefined) {
      resolved.tags = msg.tags;
    }
    else {
      resolved.tags = []
    }

    if (msg.data !== undefined) {
      resolved.data = msg.data;
    }
    else {
      resolved.data = []
    }

    if (msg.lookup_table_input !== undefined) {
      resolved.lookup_table_input = msg.lookup_table_input;
    }
    else {
      resolved.lookup_table_input = []
    }

    if (msg.lookup_table_output !== undefined) {
      resolved.lookup_table_output = msg.lookup_table_output;
    }
    else {
      resolved.lookup_table_output = []
    }

    return resolved;
    }
};

module.exports = ThrusterConversionFcn;
