import rospy

from .alarm_util import AlarmType, FailureLevel
import alarms

from tauv_msgs.msg import AlarmReport

class AlarmClient:
    def __init__(self, monitor=False) -> None:
        self.monitor = monitor
        self._lastupdated = rospy.Time(0)
        self._active = set()
        self._alarmhash = {a.id : a for a in list(AlarmType.__subclasses__())}

        # connect to server if we're not in monitor mode.
        rospy.Subscriber('/alarms/report', AlarmReport, self._update_report)
        if not monitor:
            

    def _update_report(msg: AlarmReport):
        if msg.header.stamp > self._lastupdated:
            self._lastupdated = msg.header.stamp
            self._active = set(msg.active_alarms)
        pass