"""Time utilities for TAUV."""

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
