"""Time utilities for TAUV."""

from typing import Union

from rclpy.duration import Duration as RclpyDuration
from rclpy.time import Time as RclpyTime


class Duration(RclpyDuration):
    """Duration class extending rclpy.duration.Duration with additional convenience methods."""

    def to_sec(self) -> float:
        """Convert duration to seconds as a float.

        Returns:
            Duration in seconds as a floating point number.
        """
        return self.nanoseconds / 1e9


class Time(RclpyTime):
    """Time class extending rclpy.time.Time to return our custom Duration on subtraction."""

    def __sub__(self, other):
        """Subtract two Time objects, returning our custom Duration.

        Args:
            other: Another Time object to subtract from this one.

        Returns:
            Custom Duration object representing the time difference.
        """
        # Call parent's subtraction to get rclpy.duration.Duration
        rclpy_duration = super().__sub__(other)

        # Convert to our custom Duration, preserving the same nanoseconds value
        return Duration(nanoseconds=rclpy_duration.nanoseconds)

    def __add__(self, other: Union['Duration', RclpyDuration]) -> 'Time':
        """Add a Duration to this Time object, returning a new Time.

        Args:
            other: Duration object (either util.time.Duration or rclpy.duration.Duration)
                   to add to this time.

        Returns:
            New Time object representing the sum of this time and the duration.
        """
        rclpy_time: RclpyTime = super().__add__(other)

        return Time(nanoseconds=rclpy_time.nanoseconds)
