// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from tauv_msgs:msg/RpmCommand.idl
// generated code does not contain a copyright notice

#include "tauv_msgs/msg/detail/rpm_command__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_tauv_msgs
const rosidl_type_hash_t *
tauv_msgs__msg__RpmCommand__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x37, 0xed, 0xdd, 0x00, 0x43, 0x6e, 0x7c, 0x46,
      0x09, 0x91, 0x2e, 0xb9, 0x54, 0xe6, 0x4b, 0xd5,
      0x24, 0x2c, 0xa4, 0xa0, 0x5f, 0xb2, 0xc2, 0x36,
      0x51, 0x5f, 0x3d, 0x1d, 0x22, 0x82, 0xb0, 0xd6,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types

// Hashes for external referenced types
#ifndef NDEBUG
#endif

static char tauv_msgs__msg__RpmCommand__TYPE_NAME[] = "tauv_msgs/msg/RpmCommand";

// Define type names, field names, and default values
static char tauv_msgs__msg__RpmCommand__FIELD_NAME__rpms[] = "rpms";
static char tauv_msgs__msg__RpmCommand__FIELD_NAME__enables[] = "enables";

static rosidl_runtime_c__type_description__Field tauv_msgs__msg__RpmCommand__FIELDS[] = {
  {
    {tauv_msgs__msg__RpmCommand__FIELD_NAME__rpms, 4, 4},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32_ARRAY,
      8,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {tauv_msgs__msg__RpmCommand__FIELD_NAME__enables, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_UINT8_ARRAY,
      8,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
tauv_msgs__msg__RpmCommand__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {tauv_msgs__msg__RpmCommand__TYPE_NAME, 24, 24},
      {tauv_msgs__msg__RpmCommand__FIELDS, 2, 2},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "# RpmCommand.msg\n"
  "\n"
  "int32[8] rpms\n"
  "uint8[8] enables";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
tauv_msgs__msg__RpmCommand__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {tauv_msgs__msg__RpmCommand__TYPE_NAME, 24, 24},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 48, 48},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
tauv_msgs__msg__RpmCommand__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *tauv_msgs__msg__RpmCommand__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}
