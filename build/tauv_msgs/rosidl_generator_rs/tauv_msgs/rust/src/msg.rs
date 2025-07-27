pub mod rmw {
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[link(name = "tauv_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__tauv_msgs__msg__WaterlinkedDvlFrame() -> *const std::ffi::c_void;
}

#[link(name = "tauv_msgs__rosidl_generator_c")]
extern "C" {
    fn tauv_msgs__msg__WaterlinkedDvlFrame__init(msg: *mut WaterlinkedDvlFrame) -> bool;
    fn tauv_msgs__msg__WaterlinkedDvlFrame__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<WaterlinkedDvlFrame>, size: usize) -> bool;
    fn tauv_msgs__msg__WaterlinkedDvlFrame__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<WaterlinkedDvlFrame>);
    fn tauv_msgs__msg__WaterlinkedDvlFrame__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<WaterlinkedDvlFrame>, out_seq: *mut rosidl_runtime_rs::Sequence<WaterlinkedDvlFrame>) -> bool;
}

// Corresponds to tauv_msgs__msg__WaterlinkedDvlFrame
#[repr(C)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct WaterlinkedDvlFrame {
    pub header: std_msgs::msg::rmw::Header,
    pub time: f64,
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
    pub fom: f64,
    pub covariance: [f64; 9],
    pub altitude: f64,
    pub transducer_velocity: [f64; 4],
    pub transducer_distance: [f64; 4],
    pub transducer_rssi: [f64; 4],
    pub transducer_nsd: [f64; 4],
    pub transducer_beam_valid: [bool; 4],
    pub velocity_valid: bool,
    pub status: i32,
    pub time_of_validity: i64,
    pub time_of_transmission: i64,
}



impl Default for WaterlinkedDvlFrame {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !tauv_msgs__msg__WaterlinkedDvlFrame__init(&mut msg as *mut _) {
        panic!("Call to tauv_msgs__msg__WaterlinkedDvlFrame__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for WaterlinkedDvlFrame {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { tauv_msgs__msg__WaterlinkedDvlFrame__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { tauv_msgs__msg__WaterlinkedDvlFrame__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { tauv_msgs__msg__WaterlinkedDvlFrame__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for WaterlinkedDvlFrame {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for WaterlinkedDvlFrame where Self: Sized {
  const TYPE_NAME: &'static str = "tauv_msgs/msg/WaterlinkedDvlFrame";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__tauv_msgs__msg__WaterlinkedDvlFrame() }
  }
}


#[link(name = "tauv_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__tauv_msgs__msg__RpmCommand() -> *const std::ffi::c_void;
}

#[link(name = "tauv_msgs__rosidl_generator_c")]
extern "C" {
    fn tauv_msgs__msg__RpmCommand__init(msg: *mut RpmCommand) -> bool;
    fn tauv_msgs__msg__RpmCommand__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<RpmCommand>, size: usize) -> bool;
    fn tauv_msgs__msg__RpmCommand__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<RpmCommand>);
    fn tauv_msgs__msg__RpmCommand__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<RpmCommand>, out_seq: *mut rosidl_runtime_rs::Sequence<RpmCommand>) -> bool;
}

// Corresponds to tauv_msgs__msg__RpmCommand
#[repr(C)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct RpmCommand {
    pub rpms: [i32; 8],
    pub enables: [u8; 8],
}



impl Default for RpmCommand {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !tauv_msgs__msg__RpmCommand__init(&mut msg as *mut _) {
        panic!("Call to tauv_msgs__msg__RpmCommand__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for RpmCommand {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { tauv_msgs__msg__RpmCommand__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { tauv_msgs__msg__RpmCommand__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { tauv_msgs__msg__RpmCommand__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for RpmCommand {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for RpmCommand where Self: Sized {
  const TYPE_NAME: &'static str = "tauv_msgs/msg/RpmCommand";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__tauv_msgs__msg__RpmCommand() }
  }
}


#[link(name = "tauv_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__tauv_msgs__msg__EscTelemetry() -> *const std::ffi::c_void;
}

#[link(name = "tauv_msgs__rosidl_generator_c")]
extern "C" {
    fn tauv_msgs__msg__EscTelemetry__init(msg: *mut EscTelemetry) -> bool;
    fn tauv_msgs__msg__EscTelemetry__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<EscTelemetry>, size: usize) -> bool;
    fn tauv_msgs__msg__EscTelemetry__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<EscTelemetry>);
    fn tauv_msgs__msg__EscTelemetry__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<EscTelemetry>, out_seq: *mut rosidl_runtime_rs::Sequence<EscTelemetry>) -> bool;
}

// Corresponds to tauv_msgs__msg__EscTelemetry
#[repr(C)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct EscTelemetry {
    pub header: std_msgs::msg::rmw::Header,
    pub id: u8,
    pub rpm: i32,
    pub voltage: f32,
    pub current: f32,
    pub temperature: f32,
    pub fault_code: u8,
}



impl Default for EscTelemetry {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !tauv_msgs__msg__EscTelemetry__init(&mut msg as *mut _) {
        panic!("Call to tauv_msgs__msg__EscTelemetry__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for EscTelemetry {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { tauv_msgs__msg__EscTelemetry__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { tauv_msgs__msg__EscTelemetry__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { tauv_msgs__msg__EscTelemetry__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for EscTelemetry {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for EscTelemetry where Self: Sized {
  const TYPE_NAME: &'static str = "tauv_msgs/msg/EscTelemetry";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__tauv_msgs__msg__EscTelemetry() }
  }
}


#[link(name = "tauv_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__tauv_msgs__msg__DepthSensorFrame() -> *const std::ffi::c_void;
}

#[link(name = "tauv_msgs__rosidl_generator_c")]
extern "C" {
    fn tauv_msgs__msg__DepthSensorFrame__init(msg: *mut DepthSensorFrame) -> bool;
    fn tauv_msgs__msg__DepthSensorFrame__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<DepthSensorFrame>, size: usize) -> bool;
    fn tauv_msgs__msg__DepthSensorFrame__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<DepthSensorFrame>);
    fn tauv_msgs__msg__DepthSensorFrame__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<DepthSensorFrame>, out_seq: *mut rosidl_runtime_rs::Sequence<DepthSensorFrame>) -> bool;
}

// Corresponds to tauv_msgs__msg__DepthSensorFrame
#[repr(C)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct DepthSensorFrame {
    pub header: std_msgs::msg::rmw::Header,
    pub depth: f32,
    pub pressure: f32,
    pub temperature: f32,
}



impl Default for DepthSensorFrame {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !tauv_msgs__msg__DepthSensorFrame__init(&mut msg as *mut _) {
        panic!("Call to tauv_msgs__msg__DepthSensorFrame__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for DepthSensorFrame {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { tauv_msgs__msg__DepthSensorFrame__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { tauv_msgs__msg__DepthSensorFrame__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { tauv_msgs__msg__DepthSensorFrame__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for DepthSensorFrame {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for DepthSensorFrame where Self: Sized {
  const TYPE_NAME: &'static str = "tauv_msgs/msg/DepthSensorFrame";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__tauv_msgs__msg__DepthSensorFrame() }
  }
}


#[link(name = "tauv_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__tauv_msgs__msg__Depth() -> *const std::ffi::c_void;
}

#[link(name = "tauv_msgs__rosidl_generator_c")]
extern "C" {
    fn tauv_msgs__msg__Depth__init(msg: *mut Depth) -> bool;
    fn tauv_msgs__msg__Depth__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<Depth>, size: usize) -> bool;
    fn tauv_msgs__msg__Depth__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Depth>);
    fn tauv_msgs__msg__Depth__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<Depth>, out_seq: *mut rosidl_runtime_rs::Sequence<Depth>) -> bool;
}

// Corresponds to tauv_msgs__msg__Depth
#[repr(C)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Depth {
    pub header: std_msgs::msg::rmw::Header,
    pub depth: f64,
    pub variance: f64,
}



impl Default for Depth {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !tauv_msgs__msg__Depth__init(&mut msg as *mut _) {
        panic!("Call to tauv_msgs__msg__Depth__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for Depth {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { tauv_msgs__msg__Depth__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { tauv_msgs__msg__Depth__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { tauv_msgs__msg__Depth__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for Depth {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for Depth where Self: Sized {
  const TYPE_NAME: &'static str = "tauv_msgs/msg/Depth";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__tauv_msgs__msg__Depth() }
  }
}


#[link(name = "tauv_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__tauv_msgs__msg__TargetThrust() -> *const std::ffi::c_void;
}

#[link(name = "tauv_msgs__rosidl_generator_c")]
extern "C" {
    fn tauv_msgs__msg__TargetThrust__init(msg: *mut TargetThrust) -> bool;
    fn tauv_msgs__msg__TargetThrust__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<TargetThrust>, size: usize) -> bool;
    fn tauv_msgs__msg__TargetThrust__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<TargetThrust>);
    fn tauv_msgs__msg__TargetThrust__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<TargetThrust>, out_seq: *mut rosidl_runtime_rs::Sequence<TargetThrust>) -> bool;
}

// Corresponds to tauv_msgs__msg__TargetThrust
#[repr(C)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct TargetThrust {
    pub target_thrust: [f64; 8],
}



impl Default for TargetThrust {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !tauv_msgs__msg__TargetThrust__init(&mut msg as *mut _) {
        panic!("Call to tauv_msgs__msg__TargetThrust__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for TargetThrust {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { tauv_msgs__msg__TargetThrust__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { tauv_msgs__msg__TargetThrust__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { tauv_msgs__msg__TargetThrust__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for TargetThrust {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for TargetThrust where Self: Sized {
  const TYPE_NAME: &'static str = "tauv_msgs/msg/TargetThrust";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__tauv_msgs__msg__TargetThrust() }
  }
}


#[link(name = "tauv_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__tauv_msgs__msg__NavigationState() -> *const std::ffi::c_void;
}

#[link(name = "tauv_msgs__rosidl_generator_c")]
extern "C" {
    fn tauv_msgs__msg__NavigationState__init(msg: *mut NavigationState) -> bool;
    fn tauv_msgs__msg__NavigationState__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<NavigationState>, size: usize) -> bool;
    fn tauv_msgs__msg__NavigationState__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<NavigationState>);
    fn tauv_msgs__msg__NavigationState__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<NavigationState>, out_seq: *mut rosidl_runtime_rs::Sequence<NavigationState>) -> bool;
}

// Corresponds to tauv_msgs__msg__NavigationState
#[repr(C)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct NavigationState {
    pub header: std_msgs::msg::rmw::Header,
    pub body_pose: geometry_msgs::msg::rmw::Pose,
    pub v_b: geometry_msgs::msg::rmw::Vector3,
    pub a_b: geometry_msgs::msg::rmw::Vector3,
    pub omega_b: geometry_msgs::msg::rmw::Vector3,
}



impl Default for NavigationState {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !tauv_msgs__msg__NavigationState__init(&mut msg as *mut _) {
        panic!("Call to tauv_msgs__msg__NavigationState__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for NavigationState {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { tauv_msgs__msg__NavigationState__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { tauv_msgs__msg__NavigationState__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { tauv_msgs__msg__NavigationState__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for NavigationState {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for NavigationState where Self: Sized {
  const TYPE_NAME: &'static str = "tauv_msgs/msg/NavigationState";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__tauv_msgs__msg__NavigationState() }
  }
}


#[link(name = "tauv_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__tauv_msgs__msg__VelocityAttitudeCommand() -> *const std::ffi::c_void;
}

#[link(name = "tauv_msgs__rosidl_generator_c")]
extern "C" {
    fn tauv_msgs__msg__VelocityAttitudeCommand__init(msg: *mut VelocityAttitudeCommand) -> bool;
    fn tauv_msgs__msg__VelocityAttitudeCommand__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<VelocityAttitudeCommand>, size: usize) -> bool;
    fn tauv_msgs__msg__VelocityAttitudeCommand__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<VelocityAttitudeCommand>);
    fn tauv_msgs__msg__VelocityAttitudeCommand__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<VelocityAttitudeCommand>, out_seq: *mut rosidl_runtime_rs::Sequence<VelocityAttitudeCommand>) -> bool;
}

// Corresponds to tauv_msgs__msg__VelocityAttitudeCommand
#[repr(C)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct VelocityAttitudeCommand {
    pub header: std_msgs::msg::rmw::Header,
    pub target_velocity: geometry_msgs::msg::rmw::Vector3,
    pub target_attitude: geometry_msgs::msg::rmw::Quaternion,
    pub feedforward_acceleration: geometry_msgs::msg::rmw::Vector3,
    pub velocity_control_enabled: bool,
    pub attitude_control_enabled: bool,
}



impl Default for VelocityAttitudeCommand {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !tauv_msgs__msg__VelocityAttitudeCommand__init(&mut msg as *mut _) {
        panic!("Call to tauv_msgs__msg__VelocityAttitudeCommand__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for VelocityAttitudeCommand {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { tauv_msgs__msg__VelocityAttitudeCommand__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { tauv_msgs__msg__VelocityAttitudeCommand__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { tauv_msgs__msg__VelocityAttitudeCommand__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for VelocityAttitudeCommand {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for VelocityAttitudeCommand where Self: Sized {
  const TYPE_NAME: &'static str = "tauv_msgs/msg/VelocityAttitudeCommand";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__tauv_msgs__msg__VelocityAttitudeCommand() }
  }
}


}  // mod rmw


#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct WaterlinkedDvlFrame {
    pub header: std_msgs::msg::Header,
    pub time: f64,
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
    pub fom: f64,
    pub covariance: [f64; 9],
    pub altitude: f64,
    pub transducer_velocity: [f64; 4],
    pub transducer_distance: [f64; 4],
    pub transducer_rssi: [f64; 4],
    pub transducer_nsd: [f64; 4],
    pub transducer_beam_valid: [bool; 4],
    pub velocity_valid: bool,
    pub status: i32,
    pub time_of_validity: i64,
    pub time_of_transmission: i64,
}



impl Default for WaterlinkedDvlFrame {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(crate::msg::rmw::WaterlinkedDvlFrame::default())
  }
}

impl rosidl_runtime_rs::Message for WaterlinkedDvlFrame {
  type RmwMsg = crate::msg::rmw::WaterlinkedDvlFrame;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        time: msg.time,
        vx: msg.vx,
        vy: msg.vy,
        vz: msg.vz,
        fom: msg.fom,
        covariance: msg.covariance,
        altitude: msg.altitude,
        transducer_velocity: msg.transducer_velocity,
        transducer_distance: msg.transducer_distance,
        transducer_rssi: msg.transducer_rssi,
        transducer_nsd: msg.transducer_nsd,
        transducer_beam_valid: msg.transducer_beam_valid,
        velocity_valid: msg.velocity_valid,
        status: msg.status,
        time_of_validity: msg.time_of_validity,
        time_of_transmission: msg.time_of_transmission,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
      time: msg.time,
      vx: msg.vx,
      vy: msg.vy,
      vz: msg.vz,
      fom: msg.fom,
        covariance: msg.covariance,
      altitude: msg.altitude,
        transducer_velocity: msg.transducer_velocity,
        transducer_distance: msg.transducer_distance,
        transducer_rssi: msg.transducer_rssi,
        transducer_nsd: msg.transducer_nsd,
        transducer_beam_valid: msg.transducer_beam_valid,
      velocity_valid: msg.velocity_valid,
      status: msg.status,
      time_of_validity: msg.time_of_validity,
      time_of_transmission: msg.time_of_transmission,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      time: msg.time,
      vx: msg.vx,
      vy: msg.vy,
      vz: msg.vz,
      fom: msg.fom,
      covariance: msg.covariance,
      altitude: msg.altitude,
      transducer_velocity: msg.transducer_velocity,
      transducer_distance: msg.transducer_distance,
      transducer_rssi: msg.transducer_rssi,
      transducer_nsd: msg.transducer_nsd,
      transducer_beam_valid: msg.transducer_beam_valid,
      velocity_valid: msg.velocity_valid,
      status: msg.status,
      time_of_validity: msg.time_of_validity,
      time_of_transmission: msg.time_of_transmission,
    }
  }
}


#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct RpmCommand {
    pub rpms: [i32; 8],
    pub enables: [u8; 8],
}



impl Default for RpmCommand {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(crate::msg::rmw::RpmCommand::default())
  }
}

impl rosidl_runtime_rs::Message for RpmCommand {
  type RmwMsg = crate::msg::rmw::RpmCommand;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        rpms: msg.rpms,
        enables: msg.enables,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        rpms: msg.rpms,
        enables: msg.enables,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      rpms: msg.rpms,
      enables: msg.enables,
    }
  }
}


#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct EscTelemetry {
    pub header: std_msgs::msg::Header,
    pub id: u8,
    pub rpm: i32,
    pub voltage: f32,
    pub current: f32,
    pub temperature: f32,
    pub fault_code: u8,
}



impl Default for EscTelemetry {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(crate::msg::rmw::EscTelemetry::default())
  }
}

impl rosidl_runtime_rs::Message for EscTelemetry {
  type RmwMsg = crate::msg::rmw::EscTelemetry;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        id: msg.id,
        rpm: msg.rpm,
        voltage: msg.voltage,
        current: msg.current,
        temperature: msg.temperature,
        fault_code: msg.fault_code,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
      id: msg.id,
      rpm: msg.rpm,
      voltage: msg.voltage,
      current: msg.current,
      temperature: msg.temperature,
      fault_code: msg.fault_code,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      id: msg.id,
      rpm: msg.rpm,
      voltage: msg.voltage,
      current: msg.current,
      temperature: msg.temperature,
      fault_code: msg.fault_code,
    }
  }
}


#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct DepthSensorFrame {
    pub header: std_msgs::msg::Header,
    pub depth: f32,
    pub pressure: f32,
    pub temperature: f32,
}



impl Default for DepthSensorFrame {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(crate::msg::rmw::DepthSensorFrame::default())
  }
}

impl rosidl_runtime_rs::Message for DepthSensorFrame {
  type RmwMsg = crate::msg::rmw::DepthSensorFrame;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        depth: msg.depth,
        pressure: msg.pressure,
        temperature: msg.temperature,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
      depth: msg.depth,
      pressure: msg.pressure,
      temperature: msg.temperature,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      depth: msg.depth,
      pressure: msg.pressure,
      temperature: msg.temperature,
    }
  }
}


#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Depth {
    pub header: std_msgs::msg::Header,
    pub depth: f64,
    pub variance: f64,
}



impl Default for Depth {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(crate::msg::rmw::Depth::default())
  }
}

impl rosidl_runtime_rs::Message for Depth {
  type RmwMsg = crate::msg::rmw::Depth;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        depth: msg.depth,
        variance: msg.variance,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
      depth: msg.depth,
      variance: msg.variance,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      depth: msg.depth,
      variance: msg.variance,
    }
  }
}


#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct TargetThrust {
    pub target_thrust: [f64; 8],
}



impl Default for TargetThrust {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(crate::msg::rmw::TargetThrust::default())
  }
}

impl rosidl_runtime_rs::Message for TargetThrust {
  type RmwMsg = crate::msg::rmw::TargetThrust;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        target_thrust: msg.target_thrust,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        target_thrust: msg.target_thrust,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      target_thrust: msg.target_thrust,
    }
  }
}


#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct NavigationState {
    pub header: std_msgs::msg::Header,
    pub body_pose: geometry_msgs::msg::Pose,
    pub v_b: geometry_msgs::msg::Vector3,
    pub a_b: geometry_msgs::msg::Vector3,
    pub omega_b: geometry_msgs::msg::Vector3,
}



impl Default for NavigationState {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(crate::msg::rmw::NavigationState::default())
  }
}

impl rosidl_runtime_rs::Message for NavigationState {
  type RmwMsg = crate::msg::rmw::NavigationState;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        body_pose: geometry_msgs::msg::Pose::into_rmw_message(std::borrow::Cow::Owned(msg.body_pose)).into_owned(),
        v_b: geometry_msgs::msg::Vector3::into_rmw_message(std::borrow::Cow::Owned(msg.v_b)).into_owned(),
        a_b: geometry_msgs::msg::Vector3::into_rmw_message(std::borrow::Cow::Owned(msg.a_b)).into_owned(),
        omega_b: geometry_msgs::msg::Vector3::into_rmw_message(std::borrow::Cow::Owned(msg.omega_b)).into_owned(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        body_pose: geometry_msgs::msg::Pose::into_rmw_message(std::borrow::Cow::Borrowed(&msg.body_pose)).into_owned(),
        v_b: geometry_msgs::msg::Vector3::into_rmw_message(std::borrow::Cow::Borrowed(&msg.v_b)).into_owned(),
        a_b: geometry_msgs::msg::Vector3::into_rmw_message(std::borrow::Cow::Borrowed(&msg.a_b)).into_owned(),
        omega_b: geometry_msgs::msg::Vector3::into_rmw_message(std::borrow::Cow::Borrowed(&msg.omega_b)).into_owned(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      body_pose: geometry_msgs::msg::Pose::from_rmw_message(msg.body_pose),
      v_b: geometry_msgs::msg::Vector3::from_rmw_message(msg.v_b),
      a_b: geometry_msgs::msg::Vector3::from_rmw_message(msg.a_b),
      omega_b: geometry_msgs::msg::Vector3::from_rmw_message(msg.omega_b),
    }
  }
}


#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct VelocityAttitudeCommand {
    pub header: std_msgs::msg::Header,
    pub target_velocity: geometry_msgs::msg::Vector3,
    pub target_attitude: geometry_msgs::msg::Quaternion,
    pub feedforward_acceleration: geometry_msgs::msg::Vector3,
    pub velocity_control_enabled: bool,
    pub attitude_control_enabled: bool,
}



impl Default for VelocityAttitudeCommand {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(crate::msg::rmw::VelocityAttitudeCommand::default())
  }
}

impl rosidl_runtime_rs::Message for VelocityAttitudeCommand {
  type RmwMsg = crate::msg::rmw::VelocityAttitudeCommand;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        target_velocity: geometry_msgs::msg::Vector3::into_rmw_message(std::borrow::Cow::Owned(msg.target_velocity)).into_owned(),
        target_attitude: geometry_msgs::msg::Quaternion::into_rmw_message(std::borrow::Cow::Owned(msg.target_attitude)).into_owned(),
        feedforward_acceleration: geometry_msgs::msg::Vector3::into_rmw_message(std::borrow::Cow::Owned(msg.feedforward_acceleration)).into_owned(),
        velocity_control_enabled: msg.velocity_control_enabled,
        attitude_control_enabled: msg.attitude_control_enabled,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        target_velocity: geometry_msgs::msg::Vector3::into_rmw_message(std::borrow::Cow::Borrowed(&msg.target_velocity)).into_owned(),
        target_attitude: geometry_msgs::msg::Quaternion::into_rmw_message(std::borrow::Cow::Borrowed(&msg.target_attitude)).into_owned(),
        feedforward_acceleration: geometry_msgs::msg::Vector3::into_rmw_message(std::borrow::Cow::Borrowed(&msg.feedforward_acceleration)).into_owned(),
      velocity_control_enabled: msg.velocity_control_enabled,
      attitude_control_enabled: msg.attitude_control_enabled,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      target_velocity: geometry_msgs::msg::Vector3::from_rmw_message(msg.target_velocity),
      target_attitude: geometry_msgs::msg::Quaternion::from_rmw_message(msg.target_attitude),
      feedforward_acceleration: geometry_msgs::msg::Vector3::from_rmw_message(msg.feedforward_acceleration),
      velocity_control_enabled: msg.velocity_control_enabled,
      attitude_control_enabled: msg.attitude_control_enabled,
    }
  }
}


