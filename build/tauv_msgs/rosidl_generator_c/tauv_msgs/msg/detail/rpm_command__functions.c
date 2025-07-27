// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from tauv_msgs:msg/RpmCommand.idl
// generated code does not contain a copyright notice
#include "tauv_msgs/msg/detail/rpm_command__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
tauv_msgs__msg__RpmCommand__init(tauv_msgs__msg__RpmCommand * msg)
{
  if (!msg) {
    return false;
  }
  // rpms
  // enables
  return true;
}

void
tauv_msgs__msg__RpmCommand__fini(tauv_msgs__msg__RpmCommand * msg)
{
  if (!msg) {
    return;
  }
  // rpms
  // enables
}

bool
tauv_msgs__msg__RpmCommand__are_equal(const tauv_msgs__msg__RpmCommand * lhs, const tauv_msgs__msg__RpmCommand * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // rpms
  for (size_t i = 0; i < 8; ++i) {
    if (lhs->rpms[i] != rhs->rpms[i]) {
      return false;
    }
  }
  // enables
  for (size_t i = 0; i < 8; ++i) {
    if (lhs->enables[i] != rhs->enables[i]) {
      return false;
    }
  }
  return true;
}

bool
tauv_msgs__msg__RpmCommand__copy(
  const tauv_msgs__msg__RpmCommand * input,
  tauv_msgs__msg__RpmCommand * output)
{
  if (!input || !output) {
    return false;
  }
  // rpms
  for (size_t i = 0; i < 8; ++i) {
    output->rpms[i] = input->rpms[i];
  }
  // enables
  for (size_t i = 0; i < 8; ++i) {
    output->enables[i] = input->enables[i];
  }
  return true;
}

tauv_msgs__msg__RpmCommand *
tauv_msgs__msg__RpmCommand__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tauv_msgs__msg__RpmCommand * msg = (tauv_msgs__msg__RpmCommand *)allocator.allocate(sizeof(tauv_msgs__msg__RpmCommand), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(tauv_msgs__msg__RpmCommand));
  bool success = tauv_msgs__msg__RpmCommand__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
tauv_msgs__msg__RpmCommand__destroy(tauv_msgs__msg__RpmCommand * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    tauv_msgs__msg__RpmCommand__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
tauv_msgs__msg__RpmCommand__Sequence__init(tauv_msgs__msg__RpmCommand__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tauv_msgs__msg__RpmCommand * data = NULL;

  if (size) {
    data = (tauv_msgs__msg__RpmCommand *)allocator.zero_allocate(size, sizeof(tauv_msgs__msg__RpmCommand), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = tauv_msgs__msg__RpmCommand__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        tauv_msgs__msg__RpmCommand__fini(&data[i - 1]);
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
tauv_msgs__msg__RpmCommand__Sequence__fini(tauv_msgs__msg__RpmCommand__Sequence * array)
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
      tauv_msgs__msg__RpmCommand__fini(&array->data[i]);
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

tauv_msgs__msg__RpmCommand__Sequence *
tauv_msgs__msg__RpmCommand__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tauv_msgs__msg__RpmCommand__Sequence * array = (tauv_msgs__msg__RpmCommand__Sequence *)allocator.allocate(sizeof(tauv_msgs__msg__RpmCommand__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = tauv_msgs__msg__RpmCommand__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
tauv_msgs__msg__RpmCommand__Sequence__destroy(tauv_msgs__msg__RpmCommand__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    tauv_msgs__msg__RpmCommand__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
tauv_msgs__msg__RpmCommand__Sequence__are_equal(const tauv_msgs__msg__RpmCommand__Sequence * lhs, const tauv_msgs__msg__RpmCommand__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!tauv_msgs__msg__RpmCommand__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
tauv_msgs__msg__RpmCommand__Sequence__copy(
  const tauv_msgs__msg__RpmCommand__Sequence * input,
  tauv_msgs__msg__RpmCommand__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(tauv_msgs__msg__RpmCommand);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    tauv_msgs__msg__RpmCommand * data =
      (tauv_msgs__msg__RpmCommand *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!tauv_msgs__msg__RpmCommand__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          tauv_msgs__msg__RpmCommand__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!tauv_msgs__msg__RpmCommand__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
