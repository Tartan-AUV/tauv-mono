import rospy

from .alarm_util import AlarmType, FailureLevel
from .alarms import Alarm
from tauv_msgs.msg import AlarmReport, AlarmWithMessage
from tauv_msgs.srv import SyncAlarms
from threading import Lock

class AlarmServer:
    def __init__(self):
        self._active: typing.Set[AlarmType] = set()
        self._stamp: rospy.Time = rospy.Time.now()
        self._lock = Lock()

        for at in Alarm:
            if at.default_set:
                self._active.add(at)

        self.pub = rospy.Publisher('/alarms/report', AlarmReport, queue_size=10)
        rospy.Service('/alarms/sync', SyncAlarms, self.handle_request)
        rospy.Timer(rospy.Duration(0.1), self.pub_report)

    def pub_report(self, timer_event):
        report = AlarmReport()
        with self._lock:
            report.stamp = rospy.Time.now()
            report.active_alarms = [a.id for a in self._active]
        self.pub.publish(report)

    def handle_request(self, srv: SyncAlarms._request_class):
        res = SyncAlarms._response_class()
        with self._lock:
            self._stamp = rospy.Time.now()
            for a in srv.diff:
                if a.set:
                    self._active.add(Alarm(a.id))
                else:
                    self._active.discard(Alarm(a.id))
            res.active_alarms = [a.id for a in self._active]
            res.stamp = self._stamp
            res.success = True
        return res

def main():
    rospy.init_node('alarm_server')
    AlarmServer()
    rospy.spin()