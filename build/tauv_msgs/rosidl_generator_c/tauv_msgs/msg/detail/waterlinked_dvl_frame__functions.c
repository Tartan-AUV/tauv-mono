// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from tauv_msgs:msg/WaterlinkedDvlFrame.idl
// generated code does not contain a copyright notice
#include "tauv_msgs/msg/detail/waterlinked_dvl_frame__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"

bool
tauv_msgs__msg__WaterlinkedDvlFrame__init(tauv_msgs__msg__WaterlinkedDvlFrame * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    tauv_msgs__msg__WaterlinkedDvlFrame__fini(msg);
    return false;
  }
  // time
  // vx
  // vy
  // vz
  // fom
  // covariance
  // altitude
  // transducer_velocity
  // transducer_distance
  // transducer_rssi
  // transducer_nsd
  // transducer_beam_valid
  // velocity_valid
  // status
  // time_of_validity
  // time_of_transmission
  return true;
}

void
tauv_msgs__msg__WaterlinkedDvlFrame__fini(tauv_msgs__msg__WaterlinkedDvlFrame * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // time
  // vx
  // vy
  // vz
  // fom
  // covariance
  // altitude
  // transducer_velocity
  // transducer_distance
  // transducer_rssi
  // transducer_nsd
  // transducer_beam_valid
  // velocity_valid
  // status
  // time_of_validity
  // time_of_transmission
}

bool
tauv_msgs__msg__WaterlinkedDvlFrame__are_equal(const tauv_msgs__msg__WaterlinkedDvlFrame * lhs, const tauv_msgs__msg__WaterlinkedDvlFrame * rhs)
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
  // time
  if (lhs->time != rhs->time) {
    return false;
  }
  // vx
  if (lhs->vx != rhs->vx) {
    return false;
  }
  // vy
  if (lhs->vy != rhs->vy) {
    return false;
  }
  // vz
  if (lhs->vz != rhs->vz) {
    return false;
  }
  // fom
  if (lhs->fom != rhs->fom) {
    return false;
  }
  // covariance
  for (size_t i = 0; i < 9; ++i) {
    if (lhs->covariance[i] != rhs->covariance[i]) {
      return false;
    }
  }
  // altitude
  if (lhs->altitude != rhs->altitude) {
    return false;
  }
  // transducer_velocity
  for (size_t i = 0; i < 4; ++i) {
    if (lhs->transducer_velocity[i] != rhs->transducer_velocity[i]) {
      return false;
    }
  }
  // transducer_distance
  for (size_t i = 0; i < 4; ++i) {
    if (lhs->transducer_distance[i] != rhs->transducer_distance[i]) {
      return false;
    }
  }
  // transducer_rssi
  for (size_t i = 0; i < 4; ++i) {
    if (lhs->transducer_rssi[i] != rhs->transducer_rssi[i]) {
      return false;
    }
  }
  // transducer_nsd
  for (size_t i = 0; i < 4; ++i) {
    if (lhs->transducer_nsd[i] != rhs->transducer_nsd[i]) {
      return false;
    }
  }
  // transducer_beam_valid
  for (size_t i = 0; i < 4; ++i) {
    if (lhs->transducer_beam_valid[i] != rhs->transducer_beam_valid[i]) {
      return false;
    }
  }
  // velocity_valid
  if (lhs->velocity_valid != rhs->velocity_valid) {
    return false;
  }
  // status
  if (lhs->status != rhs->status) {
    return false;
  }
  // time_of_validity
  if (lhs->time_of_validity != rhs->time_of_validity) {
    return false;
  }
  // time_of_transmission
  if (lhs->time_of_transmission != rhs->time_of_transmission) {
    return false;
  }
  return true;
}

bool
tauv_msgs__msg__WaterlinkedDvlFrame__copy(
  const tauv_msgs__msg__WaterlinkedDvlFrame * input,
  tauv_msgs__msg__WaterlinkedDvlFrame * output)
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
  // time
  output->time = input->time;
  // vx
  output->vx = input->vx;
  // vy
  output->vy = input->vy;
  // vz
  output->vz = input->vz;
  // fom
  output->fom = input->fom;
  // covariance
  for (size_t i = 0; i < 9; ++i) {
    output->covariance[i] = input->covariance[i];
  }
  // altitude
  output->altitude = input->altitude;
  // transducer_velocity
  for (size_t i = 0; i < 4; ++i) {
    output->transducer_velocity[i] = input->transducer_velocity[i];
  }
  // transducer_distance
  for (size_t i = 0; i < 4; ++i) {
    output->transducer_distance[i] = input->transducer_distance[i];
  }
  // transducer_rssi
  for (size_t i = 0; i < 4; ++i) {
    output->transducer_rssi[i] = input->transducer_rssi[i];
  }
  // transducer_nsd
  for (size_t i = 0; i < 4; ++i) {
    output->transducer_nsd[i] = input->transducer_nsd[i];
  }
  // transducer_beam_valid
  for (size_t i = 0; i < 4; ++i) {
    output->transducer_beam_valid[i] = input->transducer_beam_valid[i];
  }
  // velocity_valid
  output->velocity_valid = input->velocity_valid;
  // status
  output->status = input->status;
  // time_of_validity
  output->time_of_validity = input->time_of_validity;
  // time_of_transmission
  output->time_of_transmission = input->time_of_transmission;
  return true;
}

tauv_msgs__msg__WaterlinkedDvlFrame *
tauv_msgs__msg__WaterlinkedDvlFrame__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tauv_msgs__msg__WaterlinkedDvlFrame * msg = (tauv_msgs__msg__WaterlinkedDvlFrame *)allocator.allocate(sizeof(tauv_msgs__msg__WaterlinkedDvlFrame), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(tauv_msgs__msg__WaterlinkedDvlFrame));
  bool success = tauv_msgs__msg__WaterlinkedDvlFrame__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
tauv_msgs__msg__WaterlinkedDvlFrame__destroy(tauv_msgs__msg__WaterlinkedDvlFrame * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    tauv_msgs__msg__WaterlinkedDvlFrame__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
tauv_msgs__msg__WaterlinkedDvlFrame__Sequence__init(tauv_msgs__msg__WaterlinkedDvlFrame__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tauv_msgs__msg__WaterlinkedDvlFrame * data = NULL;

  if (size) {
    data = (tauv_msgs__msg__WaterlinkedDvlFrame *)allocator.zero_allocate(size, sizeof(tauv_msgs__msg__WaterlinkedDvlFrame), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = tauv_msgs__msg__WaterlinkedDvlFrame__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        tauv_msgs__msg__WaterlinkedDvlFrame__fini(&data[i - 1]);
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
tauv_msgs__msg__WaterlinkedDvlFrame__Sequence__fini(tauv_msgs__msg__WaterlinkedDvlFrame__Sequence * array)
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
      tauv_msgs__msg__WaterlinkedDvlFrame__fini(&array->data[i]);
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

tauv_msgs__msg__WaterlinkedDvlFrame__Sequence *
tauv_msgs__msg__WaterlinkedDvlFrame__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tauv_msgs__msg__WaterlinkedDvlFrame__Sequence * array = (tauv_msgs__msg__WaterlinkedDvlFrame__Sequence *)allocator.allocate(sizeof(tauv_msgs__msg__WaterlinkedDvlFrame__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = tauv_msgs__msg__WaterlinkedDvlFrame__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
tauv_msgs__msg__WaterlinkedDvlFrame__Sequence__destroy(tauv_msgs__msg__WaterlinkedDvlFrame__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    tauv_msgs__msg__WaterlinkedDvlFrame__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
tauv_msgs__msg__WaterlinkedDvlFrame__Sequence__are_equal(const tauv_msgs__msg__WaterlinkedDvlFrame__Sequence * lhs, const tauv_msgs__msg__WaterlinkedDvlFrame__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!tauv_msgs__msg__WaterlinkedDvlFrame__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
tauv_msgs__msg__WaterlinkedDvlFrame__Sequence__copy(
  const tauv_msgs__msg__WaterlinkedDvlFrame__Sequence * input,
  tauv_msgs__msg__WaterlinkedDvlFrame__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(tauv_msgs__msg__WaterlinkedDvlFrame);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    tauv_msgs__msg__WaterlinkedDvlFrame * data =
      (tauv_msgs__msg__WaterlinkedDvlFrame *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!tauv_msgs__msg__WaterlinkedDvlFrame__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          tauv_msgs__msg__WaterlinkedDvlFrame__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!tauv_msgs__msg__WaterlinkedDvlFrame__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
