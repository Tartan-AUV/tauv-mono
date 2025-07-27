// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from tauv_msgs:msg/TargetThrust.idl
// generated code does not contain a copyright notice

#include "tauv_msgs/msg/detail/target_thrust__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_tauv_msgs
const rosidl_type_hash_t *
tauv_msgs__msg__TargetThrust__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x3e, 0x19, 0xc1, 0x0a, 0x4c, 0xaf, 0x44, 0xcb,
      0x8c, 0xde, 0xa1, 0x4e, 0x4a, 0xab, 0x17, 0xfb,
      0xb3, 0x5d, 0x15, 0x22, 0x14, 0xe1, 0x9c, 0x0b,
      0x1e, 0x4a, 0x03, 0x1c, 0x6f, 0x2e, 0x18, 0xf2,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types

// Hashes for external referenced types
#ifndef NDEBUG
#endif

static char tauv_msgs__msg__TargetThrust__TYPE_NAME[] = "tauv_msgs/msg/TargetThrust";

// Define type names, field names, and default values
static char tauv_msgs__msg__TargetThrust__FIELD_NAME__target_thrust[] = "target_thrust";

static rosidl_runtime_c__type_description__Field tauv_msgs__msg__TargetThrust__FIELDS[] = {
  {
    {tauv_msgs__msg__TargetThrust__FIELD_NAME__target_thrust, 13, 13},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE_ARRAY,
      8,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
tauv_msgs__msg__TargetThrust__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {tauv_msgs__msg__TargetThrust__TYPE_NAME, 26, 26},
      {tauv_msgs__msg__TargetThrust__FIELDS, 1, 1},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "# TargetThrust.msg\n"
  "\n"
  "float64[8] target_thrust  # [N]";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
tauv_msgs__msg__TargetThrust__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {tauv_msgs__msg__TargetThrust__TYPE_NAME, 26, 26},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 52, 52},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
tauv_msgs__msg__TargetThrust__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *tauv_msgs__msg__TargetThrust__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}
