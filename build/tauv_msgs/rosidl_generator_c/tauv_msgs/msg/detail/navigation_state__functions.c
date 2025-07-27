// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from tauv_msgs:msg/NavigationState.idl
// generated code does not contain a copyright notice
#include "tauv_msgs/msg/detail/navigation_state__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `body_pose`
#include "geometry_msgs/msg/detail/pose__functions.h"
// Member `v_b`
// Member `a_b`
// Member `omega_b`
#include "geometry_msgs/msg/detail/vector3__functions.h"

bool
tauv_msgs__msg__NavigationState__init(tauv_msgs__msg__NavigationState * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    tauv_msgs__msg__NavigationState__fini(msg);
    return false;
  }
  // body_pose
  if (!geometry_msgs__msg__Pose__init(&msg->body_pose)) {
    tauv_msgs__msg__NavigationState__fini(msg);
    return false;
  }
  // v_b
  if (!geometry_msgs__msg__Vector3__init(&msg->v_b)) {
    tauv_msgs__msg__NavigationState__fini(msg);
    return false;
  }
  // a_b
  if (!geometry_msgs__msg__Vector3__init(&msg->a_b)) {
    tauv_msgs__msg__NavigationState__fini(msg);
    return false;
  }
  // omega_b
  if (!geometry_msgs__msg__Vector3__init(&msg->omega_b)) {
    tauv_msgs__msg__NavigationState__fini(msg);
    return false;
  }
  return true;
}

void
tauv_msgs__msg__NavigationState__fini(tauv_msgs__msg__NavigationState * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // body_pose
  geometry_msgs__msg__Pose__fini(&msg->body_pose);
  // v_b
  geometry_msgs__msg__Vector3__fini(&msg->v_b);
  // a_b
  geometry_msgs__msg__Vector3__fini(&msg->a_b);
  // omega_b
  geometry_msgs__msg__Vector3__fini(&msg->omega_b);
}

bool
tauv_msgs__msg__NavigationState__are_equal(const tauv_msgs__msg__NavigationState * lhs, const tauv_msgs__msg__NavigationState * rhs)
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
  // body_pose
  if (!geometry_msgs__msg__Pose__are_equal(
      &(lhs->body_pose), &(rhs->body_pose)))
  {
    return false;
  }
  // v_b
  if (!geometry_msgs__msg__Vector3__are_equal(
      &(lhs->v_b), &(rhs->v_b)))
  {
    return false;
  }
  // a_b
  if (!geometry_msgs__msg__Vector3__are_equal(
      &(lhs->a_b), &(rhs->a_b)))
  {
    return false;
  }
  // omega_b
  if (!geometry_msgs__msg__Vector3__are_equal(
      &(lhs->omega_b), &(rhs->omega_b)))
  {
    return false;
  }
  return true;
}

bool
tauv_msgs__msg__NavigationState__copy(
  const tauv_msgs__msg__NavigationState * input,
  tauv_msgs__msg__NavigationState * output)
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
  // body_pose
  if (!geometry_msgs__msg__Pose__copy(
      &(input->body_pose), &(output->body_pose)))
  {
    return false;
  }
  // v_b
  if (!geometry_msgs__msg__Vector3__copy(
      &(input->v_b), &(output->v_b)))
  {
    return false;
  }
  // a_b
  if (!geometry_msgs__msg__Vector3__copy(
      &(input->a_b), &(output->a_b)))
  {
    return false;
  }
  // omega_b
  if (!geometry_msgs__msg__Vector3__copy(
      &(input->omega_b), &(output->omega_b)))
  {
    return false;
  }
  return true;
}

tauv_msgs__msg__NavigationState *
tauv_msgs__msg__NavigationState__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tauv_msgs__msg__NavigationState * msg = (tauv_msgs__msg__NavigationState *)allocator.allocate(sizeof(tauv_msgs__msg__NavigationState), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(tauv_msgs__msg__NavigationState));
  bool success = tauv_msgs__msg__NavigationState__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
tauv_msgs__msg__NavigationState__destroy(tauv_msgs__msg__NavigationState * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    tauv_msgs__msg__NavigationState__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
tauv_msgs__msg__NavigationState__Sequence__init(tauv_msgs__msg__NavigationState__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tauv_msgs__msg__NavigationState * data = NULL;

  if (size) {
    data = (tauv_msgs__msg__NavigationState *)allocator.zero_allocate(size, sizeof(tauv_msgs__msg__NavigationState), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = tauv_msgs__msg__NavigationState__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        tauv_msgs__msg__NavigationState__fini(&data[i - 1]);
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
tauv_msgs__msg__NavigationState__Sequence__fini(tauv_msgs__msg__NavigationState__Sequence * array)
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
      tauv_msgs__msg__NavigationState__fini(&array->data[i]);
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

tauv_msgs__msg__NavigationState__Sequence *
tauv_msgs__msg__NavigationState__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tauv_msgs__msg__NavigationState__Sequence * array = (tauv_msgs__msg__NavigationState__Sequence *)allocator.allocate(sizeof(tauv_msgs__msg__NavigationState__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = tauv_msgs__msg__NavigationState__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
tauv_msgs__msg__NavigationState__Sequence__destroy(tauv_msgs__msg__NavigationState__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    tauv_msgs__msg__NavigationState__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
tauv_msgs__msg__NavigationState__Sequence__are_equal(const tauv_msgs__msg__NavigationState__Sequence * lhs, const tauv_msgs__msg__NavigationState__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!tauv_msgs__msg__NavigationState__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
tauv_msgs__msg__NavigationState__Sequence__copy(
  const tauv_msgs__msg__NavigationState__Sequence * input,
  tauv_msgs__msg__NavigationState__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(tauv_msgs__msg__NavigationState);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    tauv_msgs__msg__NavigationState * data =
      (tauv_msgs__msg__NavigationState *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!tauv_msgs__msg__NavigationState__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          tauv_msgs__msg__NavigationState__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!tauv_msgs__msg__NavigationState__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
