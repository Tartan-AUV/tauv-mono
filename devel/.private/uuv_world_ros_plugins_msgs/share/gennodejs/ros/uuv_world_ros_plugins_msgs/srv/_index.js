
"use strict";

let GetCurrentModel = require('./GetCurrentModel.js')
let SetCurrentDirection = require('./SetCurrentDirection.js')
let TransformFromSphericalCoord = require('./TransformFromSphericalCoord.js')
let TransformToSphericalCoord = require('./TransformToSphericalCoord.js')
let SetOriginSphericalCoord = require('./SetOriginSphericalCoord.js')
let SetCurrentVelocity = require('./SetCurrentVelocity.js')
let SetCurrentModel = require('./SetCurrentModel.js')
let GetOriginSphericalCoord = require('./GetOriginSphericalCoord.js')

module.exports = {
  GetCurrentModel: GetCurrentModel,
  SetCurrentDirection: SetCurrentDirection,
  TransformFromSphericalCoord: TransformFromSphericalCoord,
  TransformToSphericalCoord: TransformToSphericalCoord,
  SetOriginSphericalCoord: SetOriginSphericalCoord,
  SetCurrentVelocity: SetCurrentVelocity,
  SetCurrentModel: SetCurrentModel,
  GetOriginSphericalCoord: GetOriginSphericalCoord,
};
