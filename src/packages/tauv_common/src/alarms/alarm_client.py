import rospy
import typing

from .alarm_util import AlarmType, FailureLevel
from .alarms import Alarm
from tauv_msgs.msg import AlarmReport
from threading import Lock

class AlarmClient:
    def __init__(self, monitor=False) -> None:
        self.monitor = monitor
        self._lastupdated = rospy.Time(0)
        self._local: typing.Set[AlarmType] = set()
        self._localadd: typing.Set[AlarmType] = set()
        self._localrem: typing.Set[AlarmType] = set()

        self._fromserver: typing.Set[AlarmType] = set()
        
        self._lock = Lock()

        # connect to server if we're not in monitor mode.
        rospy.Subscriber('/alarms/report', AlarmReport, self._update_report)
        self.pub = rospy.Publisher('/alarms/post', AlarmReport, queue_size=10)

        if not monitor:
            # wait for first report from the server
            while not self._lastupdated > rospy.Time(0):
                rospy.sleep(0.05)

    def _update_report(self, msg: AlarmReport):
        if msg.header.stamp > self._lastupdated:
            with self._lock:
                self._lastupdated = msg.header.stamp
                self._active = set([Alarm(i) for i in msg.active_alarms])
    
    def set(self, a: AlarmType, set=True):
        with self._locK:
            if set and a not in self._active:
                self._active.add(a)
                self._lastupdated = rospy.Time.now()
            elif not set and a in self._active:
                self._active.remove(a)
                self._lastupdated = rospy.Time.now()
        
            ar = AlarmReport()
            ar.header.stamp = self._lastupdated
            ar.active_alarms = [a.id for a in self._active]

        self.pub.publish(ar)