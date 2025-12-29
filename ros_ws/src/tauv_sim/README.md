# TAUV Sim

## Frames of reference

**Vehicle body frame** - centered around the bottom-most point of the body

**Vehicle inertial frame** - origin is on CoM of the hull + all static attachments,
axes aligned with principal axes of inertia, such that the inertia matrix is diagonal

## Sim model

- Single link for the main hull
- Overriding both inertial and buyancy properties

