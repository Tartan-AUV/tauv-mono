// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from tauv_msgs:msg/WaterlinkedDvlFrame.idl
// generated code does not contain a copyright notice

#include "tauv_msgs/msg/detail/waterlinked_dvl_frame__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_tauv_msgs
const rosidl_type_hash_t *
tauv_msgs__msg__WaterlinkedDvlFrame__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xc7, 0x8e, 0x54, 0x6a, 0x6d, 0x02, 0x18, 0xb0,
      0x83, 0xd9, 0x42, 0xfb, 0x38, 0x56, 0xe6, 0xee,
      0x0d, 0x3a, 0xfc, 0xab, 0x54, 0x87, 0x47, 0x84,
      0x39, 0x5c, 0x46, 0x74, 0xa8, 0xa8, 0x35, 0x42,
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

static char tauv_msgs__msg__WaterlinkedDvlFrame__TYPE_NAME[] = "tauv_msgs/msg/WaterlinkedDvlFrame";
static char builtin_interfaces__msg__Time__TYPE_NAME[] = "builtin_interfaces/msg/Time";
static char std_msgs__msg__Header__TYPE_NAME[] = "std_msgs/msg/Header";

// Define type names, field names, and default values
static char tauv_msgs__msg__WaterlinkedDvlFrame__FIELD_NAME__header[] = "header";
static char tauv_msgs__msg__WaterlinkedDvlFrame__FIELD_NAME__time[] = "time";
static char tauv_msgs__msg__WaterlinkedDvlFrame__FIELD_NAME__vx[] = "vx";
static char tauv_msgs__msg__WaterlinkedDvlFrame__FIELD_NAME__vy[] = "vy";
static char tauv_msgs__msg__WaterlinkedDvlFrame__FIELD_NAME__vz[] = "vz";
static char tauv_msgs__msg__WaterlinkedDvlFrame__FIELD_NAME__fom[] = "fom";
static char tauv_msgs__msg__WaterlinkedDvlFrame__FIELD_NAME__covariance[] = "covariance";
static char tauv_msgs__msg__WaterlinkedDvlFrame__FIELD_NAME__altitude[] = "altitude";
static char tauv_msgs__msg__WaterlinkedDvlFrame__FIELD_NAME__transducer_velocity[] = "transducer_velocity";
static char tauv_msgs__msg__WaterlinkedDvlFrame__FIELD_NAME__transducer_distance[] = "transducer_distance";
static char tauv_msgs__msg__WaterlinkedDvlFrame__FIELD_NAME__transducer_rssi[] = "transducer_rssi";
static char tauv_msgs__msg__WaterlinkedDvlFrame__FIELD_NAME__transducer_nsd[] = "transducer_nsd";
static char tauv_msgs__msg__WaterlinkedDvlFrame__FIELD_NAME__transducer_beam_valid[] = "transducer_beam_valid";
static char tauv_msgs__msg__WaterlinkedDvlFrame__FIELD_NAME__velocity_valid[] = "velocity_valid";
static char tauv_msgs__msg__WaterlinkedDvlFrame__FIELD_NAME__status[] = "status";
static char tauv_msgs__msg__WaterlinkedDvlFrame__FIELD_NAME__time_of_validity[] = "time_of_validity";
static char tauv_msgs__msg__WaterlinkedDvlFrame__FIELD_NAME__time_of_transmission[] = "time_of_transmission";

static rosidl_runtime_c__type_description__Field tauv_msgs__msg__WaterlinkedDvlFrame__FIELDS[] = {
  {
    {tauv_msgs__msg__WaterlinkedDvlFrame__FIELD_NAME__header, 6, 6},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {std_msgs__msg__Header__TYPE_NAME, 19, 19},
    },
    {NULL, 0, 0},
  },
  {
    {tauv_msgs__msg__WaterlinkedDvlFrame__FIELD_NAME__time, 4, 4},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {tauv_msgs__msg__WaterlinkedDvlFrame__FIELD_NAME__vx, 2, 2},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {tauv_msgs__msg__WaterlinkedDvlFrame__FIELD_NAME__vy, 2, 2},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {tauv_msgs__msg__WaterlinkedDvlFrame__FIELD_NAME__vz, 2, 2},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {tauv_msgs__msg__WaterlinkedDvlFrame__FIELD_NAME__fom, 3, 3},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {tauv_msgs__msg__WaterlinkedDvlFrame__FIELD_NAME__covariance, 10, 10},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE_ARRAY,
      9,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {tauv_msgs__msg__WaterlinkedDvlFrame__FIELD_NAME__altitude, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {tauv_msgs__msg__WaterlinkedDvlFrame__FIELD_NAME__transducer_velocity, 19, 19},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE_ARRAY,
      4,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {tauv_msgs__msg__WaterlinkedDvlFrame__FIELD_NAME__transducer_distance, 19, 19},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE_ARRAY,
      4,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {tauv_msgs__msg__WaterlinkedDvlFrame__FIELD_NAME__transducer_rssi, 15, 15},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE_ARRAY,
      4,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {tauv_msgs__msg__WaterlinkedDvlFrame__FIELD_NAME__transducer_nsd, 14, 14},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE_ARRAY,
      4,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {tauv_msgs__msg__WaterlinkedDvlFrame__FIELD_NAME__transducer_beam_valid, 21, 21},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_BOOLEAN_ARRAY,
      4,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {tauv_msgs__msg__WaterlinkedDvlFrame__FIELD_NAME__velocity_valid, 14, 14},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_BOOLEAN,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {tauv_msgs__msg__WaterlinkedDvlFrame__FIELD_NAME__status, 6, 6},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {tauv_msgs__msg__WaterlinkedDvlFrame__FIELD_NAME__time_of_validity, 16, 16},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT64,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {tauv_msgs__msg__WaterlinkedDvlFrame__FIELD_NAME__time_of_transmission, 20, 20},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT64,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription tauv_msgs__msg__WaterlinkedDvlFrame__REFERENCED_TYPE_DESCRIPTIONS[] = {
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
tauv_msgs__msg__WaterlinkedDvlFrame__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {tauv_msgs__msg__WaterlinkedDvlFrame__TYPE_NAME, 33, 33},
      {tauv_msgs__msg__WaterlinkedDvlFrame__FIELDS, 17, 17},
    },
    {tauv_msgs__msg__WaterlinkedDvlFrame__REFERENCED_TYPE_DESCRIPTIONS, 2, 2},
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
  "# WaterlinkedDvlFrame.msg\n"
  "std_msgs/Header header\n"
  "\n"
  "# Timestamp from the DVL device\n"
  "float64 time\n"
  "\n"
  "# Velocity components in m/s\n"
  "float64 vx\n"
  "float64 vy\n"
  "float64 vz\n"
  "\n"
  "# Figure of merit (lower is better)\n"
  "float64 fom\n"
  "\n"
  "# 3x3 covariance matrix, flattened row-major\n"
  "float64[9] covariance\n"
  "\n"
  "# Altitude above seabed in meters\n"
  "float64 altitude\n"
  "\n"
  "# Transducer-specific measurements (4 beams assumed)\n"
  "float64[4] transducer_velocity\n"
  "float64[4] transducer_distance\n"
  "float64[4] transducer_rssi\n"
  "float64[4] transducer_nsd\n"
  "bool[4]    transducer_beam_valid\n"
  "\n"
  "# Whether velocity measurement is valid\n"
  "bool velocity_valid\n"
  "\n"
  "# DVL status code\n"
  "int32 status\n"
  "\n"
  "# DVL timestamps (e.g. in microseconds since epoch or device boot time)\n"
  "int64 time_of_validity\n"
  "int64 time_of_transmission";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
tauv_msgs__msg__WaterlinkedDvlFrame__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {tauv_msgs__msg__WaterlinkedDvlFrame__TYPE_NAME, 33, 33},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 745, 745},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
tauv_msgs__msg__WaterlinkedDvlFrame__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[3];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 3, 3};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *tauv_msgs__msg__WaterlinkedDvlFrame__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *std_msgs__msg__Header__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}
