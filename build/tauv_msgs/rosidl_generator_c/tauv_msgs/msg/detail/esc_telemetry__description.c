// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from tauv_msgs:msg/EscTelemetry.idl
// generated code does not contain a copyright notice

#include "tauv_msgs/msg/detail/esc_telemetry__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_tauv_msgs
const rosidl_type_hash_t *
tauv_msgs__msg__EscTelemetry__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x9c, 0x52, 0x58, 0x1b, 0xcd, 0x13, 0xc0, 0xab,
      0x68, 0x15, 0x9c, 0x54, 0x9e, 0x48, 0x51, 0x7e,
      0xce, 0xe4, 0x7b, 0x86, 0x09, 0x9c, 0x04, 0x25,
      0x0c, 0xd8, 0x19, 0x9f, 0x34, 0x3c, 0x2a, 0xf3,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types
#include "builtin_interfaces/msg/detail/time__functions.h"
#include "std_msgs/msg/detail/header__functions.h"

// Hashes for external referenced types
#ifndef NDEBUG
static const rosidl_type_hash_t builtin_interfaces__msg__Time__EXPECTED_HASH = {1, {
    0xb1, 0x06, 0x23, 0x5e, 0x25, 0xa4, 0xc5, 0xed,
    0x35, 0x09, 0x8a, 0xa0, 0xa6, 0x1a, 0x3e, 0xe9,
    0xc9, 0xb1, 0x8d, 0x19, 0x7f, 0x39, 0x8b, 0x0e,
    0x42, 0x06, 0xce, 0xa9, 0xac, 0xf9, 0xc1, 0x97,
  }};
static const rosidl_type_hash_t std_msgs__msg__Header__EXPECTED_HASH = {1, {
    0xf4, 0x9f, 0xb3, 0xae, 0x2c, 0xf0, 0x70, 0xf7,
    0x93, 0x64, 0x5f, 0xf7, 0x49, 0x68, 0x3a, 0xc6,
    0xb0, 0x62, 0x03, 0xe4, 0x1c, 0x89, 0x1e, 0x17,
    0x70, 0x1b, 0x1c, 0xb5, 0x97, 0xce, 0x6a, 0x01,
  }};
#endif

static char tauv_msgs__msg__EscTelemetry__TYPE_NAME[] = "tauv_msgs/msg/EscTelemetry";
static char builtin_interfaces__msg__Time__TYPE_NAME[] = "builtin_interfaces/msg/Time";
static char std_msgs__msg__Header__TYPE_NAME[] = "std_msgs/msg/Header";

// Define type names, field names, and default values
static char tauv_msgs__msg__EscTelemetry__FIELD_NAME__header[] = "header";
static char tauv_msgs__msg__EscTelemetry__FIELD_NAME__id[] = "id";
static char tauv_msgs__msg__EscTelemetry__FIELD_NAME__rpm[] = "rpm";
static char tauv_msgs__msg__EscTelemetry__FIELD_NAME__voltage[] = "voltage";
static char tauv_msgs__msg__EscTelemetry__FIELD_NAME__current[] = "current";
static char tauv_msgs__msg__EscTelemetry__FIELD_NAME__temperature[] = "temperature";
static char tauv_msgs__msg__EscTelemetry__FIELD_NAME__fault_code[] = "fault_code";

static rosidl_runtime_c__type_description__Field tauv_msgs__msg__EscTelemetry__FIELDS[] = {
  {
    {tauv_msgs__msg__EscTelemetry__FIELD_NAME__header, 6, 6},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {std_msgs__msg__Header__TYPE_NAME, 19, 19},
    },
    {NULL, 0, 0},
  },
  {
    {tauv_msgs__msg__EscTelemetry__FIELD_NAME__id, 2, 2},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_UINT8,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {tauv_msgs__msg__EscTelemetry__FIELD_NAME__rpm, 3, 3},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {tauv_msgs__msg__EscTelemetry__FIELD_NAME__voltage, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {tauv_msgs__msg__EscTelemetry__FIELD_NAME__current, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {tauv_msgs__msg__EscTelemetry__FIELD_NAME__temperature, 11, 11},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {tauv_msgs__msg__EscTelemetry__FIELD_NAME__fault_code, 10, 10},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_UINT8,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription tauv_msgs__msg__EscTelemetry__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {std_msgs__msg__Header__TYPE_NAME, 19, 19},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
tauv_msgs__msg__EscTelemetry__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {tauv_msgs__msg__EscTelemetry__TYPE_NAME, 26, 26},
      {tauv_msgs__msg__EscTelemetry__FIELDS, 7, 7},
    },
    {tauv_msgs__msg__EscTelemetry__REFERENCED_TYPE_DESCRIPTIONS, 2, 2},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&std_msgs__msg__Header__EXPECTED_HASH, std_msgs__msg__Header__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[1].fields = std_msgs__msg__Header__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "# EscTelemetry.msg\n"
  "\n"
  "std_msgs/Header header\n"
  "\n"
  "uint8 id          # ESC identifier\n"
  "int32 rpm         # Rotations per minute\n"
  "float32 voltage   # Voltage in volts\n"
  "float32 current   # Current in amps\n"
  "float32 temperature # Temperature in Celsius\n"
  "uint8 fault_code  # ESC fault code, if any";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
tauv_msgs__msg__EscTelemetry__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {tauv_msgs__msg__EscTelemetry__TYPE_NAME, 26, 26},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 281, 281},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
tauv_msgs__msg__EscTelemetry__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[3];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 3, 3};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *tauv_msgs__msg__EscTelemetry__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *std_msgs__msg__Header__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}
