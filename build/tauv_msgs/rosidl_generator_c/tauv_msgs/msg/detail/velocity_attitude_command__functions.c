// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from tauv_msgs:msg/VelocityAttitudeCommand.idl
// generated code does not contain a copyright notice
#include "tauv_msgs/msg/detail/velocity_attitude_command__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `target_velocity`
// Member `feedforward_acceleration`
#include "geometry_msgs/msg/detail/vector3__functions.h"
// Member `target_attitude`
#include "geometry_msgs/msg/detail/quaternion__functions.h"

bool
tauv_msgs__msg__VelocityAttitudeCommand__init(tauv_msgs__msg__VelocityAttitudeCommand * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    tauv_msgs__msg__VelocityAttitudeCommand__fini(msg);
    return false;
  }
  // target_velocity
  if (!geometry_msgs__msg__Vector3__init(&msg->target_velocity)) {
    tauv_msgs__msg__VelocityAttitudeCommand__fini(msg);
    return false;
  }
  // target_attitude
  if (!geometry_msgs__msg__Quaternion__init(&msg->target_attitude)) {
    tauv_msgs__msg__VelocityAttitudeCommand__fini(msg);
    return false;
  }
  // feedforward_acceleration
  if (!geometry_msgs__msg__Vector3__init(&msg->feedforward_acceleration)) {
    tauv_msgs__msg__VelocityAttitudeCommand__fini(msg);
    return false;
  }
  // velocity_control_enabled
  // attitude_control_enabled
  return true;
}

void
tauv_msgs__msg__VelocityAttitudeCommand__fini(tauv_msgs__msg__VelocityAttitudeCommand * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // target_velocity
  geometry_msgs__msg__Vector3__fini(&msg->target_velocity);
  // target_attitude
  geometry_msgs__msg__Quaternion__fini(&msg->target_attitude);
  // feedforward_acceleration
  geometry_msgs__msg__Vector3__fini(&msg->feedforward_acceleration);
  // velocity_control_enabled
  // attitude_control_enabled
}

bool
tauv_msgs__msg__VelocityAttitudeCommand__are_equal(const tauv_msgs__msg__VelocityAttitudeCommand * lhs, const tauv_msgs__msg__VelocityAttitudeCommand * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__are_equal(
      &(lhs->header), &(rhs->header)))
  {
    return false;
  }
  // target_velocity
  if (!geometry_msgs__msg__Vector3__are_equal(
      &(lhs->target_velocity), &(rhs->target_velocity)))
  {
    return false;
  }
  // target_attitude
  if (!geometry_msgs__msg__Quaternion__are_equal(
      &(lhs->target_attitude), &(rhs->target_attitude)))
  {
    return false;
  }
  // feedforward_acceleration
  if (!geometry_msgs__msg__Vector3__are_equal(
      &(lhs->feedforward_acceleration), &(rhs->feedforward_acceleration)))
  {
    return false;
  }
  // velocity_control_enabled
  if (lhs->velocity_control_enabled != rhs->velocity_control_enabled) {
    return false;
  }
  // attitude_control_enabled
  if (lhs->attitude_control_enabled != rhs->attitude_control_enabled) {
    return false;
  }
  return true;
}

bool
tauv_msgs__msg__VelocityAttitudeCommand__copy(
  const tauv_msgs__msg__VelocityAttitudeCommand * input,
  tauv_msgs__msg__VelocityAttitudeCommand * output)
{
  if (!input || !output) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__copy(
      &(input->header), &(output->header)))
  {
    return false;
  }
  // target_velocity
  if (!geometry_msgs__msg__Vector3__copy(
      &(input->target_velocity), &(output->target_velocity)))
  {
    return false;
  }
  // target_attitude
  if (!geometry_msgs__msg__Quaternion__copy(
      &(input->target_attitude), &(output->target_attitude)))
  {
    return false;
  }
  // feedforward_acceleration
  if (!geometry_msgs__msg__Vector3__copy(
      &(input->feedforward_acceleration), &(output->feedforward_acceleration)))
  {
    return false;
  }
  // velocity_control_enabled
  output->velocity_control_enabled = input->velocity_control_enabled;
  // attitude_control_enabled
  output->attitude_control_enabled = input->attitude_control_enabled;
  return true;
}

tauv_msgs__msg__VelocityAttitudeCommand *
tauv_msgs__msg__VelocityAttitudeCommand__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tauv_msgs__msg__VelocityAttitudeCommand * msg = (tauv_msgs__msg__VelocityAttitudeCommand *)allocator.allocate(sizeof(tauv_msgs__msg__VelocityAttitudeCommand), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(tauv_msgs__msg__VelocityAttitudeCommand));
  bool success = tauv_msgs__msg__VelocityAttitudeCommand__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
tauv_msgs__msg__VelocityAttitudeCommand__destroy(tauv_msgs__msg__VelocityAttitudeCommand * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    tauv_msgs__msg__VelocityAttitudeCommand__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
tauv_msgs__msg__VelocityAttitudeCommand__Sequence__init(tauv_msgs__msg__VelocityAttitudeCommand__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tauv_msgs__msg__VelocityAttitudeCommand * data = NULL;

  if (size) {
    data = (tauv_msgs__msg__VelocityAttitudeCommand *)allocator.zero_allocate(size, sizeof(tauv_msgs__msg__VelocityAttitudeCommand), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = tauv_msgs__msg__VelocityAttitudeCommand__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        tauv_msgs__msg__VelocityAttitudeCommand__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
tauv_msgs__msg__VelocityAttitudeCommand__Sequence__fini(tauv_msgs__msg__VelocityAttitudeCommand__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      tauv_msgs__msg__VelocityAttitudeCommand__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

tauv_msgs__msg__VelocityAttitudeCommand__Sequence *
tauv_msgs__msg__VelocityAttitudeCommand__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tauv_msgs__msg__VelocityAttitudeCommand__Sequence * array = (tauv_msgs__msg__VelocityAttitudeCommand__Sequence *)allocator.allocate(sizeof(tauv_msgs__msg__VelocityAttitudeCommand__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = tauv_msgs__msg__VelocityAttitudeCommand__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
tauv_msgs__msg__VelocityAttitudeCommand__Sequence__destroy(tauv_msgs__msg__VelocityAttitudeCommand__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    tauv_msgs__msg__VelocityAttitudeCommand__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
tauv_msgs__msg__VelocityAttitudeCommand__Sequence__are_equal(const tauv_msgs__msg__VelocityAttitudeCommand__Sequence * lhs, const tauv_msgs__msg__VelocityAttitudeCommand__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!tauv_msgs__msg__VelocityAttitudeCommand__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
tauv_msgs__msg__VelocityAttitudeCommand__Sequence__copy(
  const tauv_msgs__msg__VelocityAttitudeCommand__Sequence * input,
  tauv_msgs__msg__VelocityAttitudeCommand__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(tauv_msgs__msg__VelocityAttitudeCommand);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    tauv_msgs__msg__VelocityAttitudeCommand * data =
      (tauv_msgs__msg__VelocityAttitudeCommand *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!tauv_msgs__msg__VelocityAttitudeCommand__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          tauv_msgs__msg__VelocityAttitudeCommand__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!tauv_msgs__msg__VelocityAttitudeCommand__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
