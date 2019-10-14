#!/usr/bin/env python
from maestro import Maestro
import rospy
import message_filters
from uuv_gazebo_ros_plugins_msgs.msg import FloatStamped


class ActuatorController:
    def __init__(self):
        maestro_tty = rospy.get_param('/vehicle_params/maestro_tty')

        self.has_thrusters = rospy.get_param('/vehicle_params/maestro_thrusters')
        self.has_servos = rospy.get_param('/vehicle_params/maestro_servos')

        if not self.has_thrusters and not self.has_servos:
            raise ValueError('''Error: maestro node is running, but vehicle is not
            configured to use it for servos or thrusters.''')

        self.maestro = Maestro(ttyStr=maestro_tty)

        self.thrusters = rospy.get_param('/vehicle_params/maestro_thruster_channels')
        self.servos = rospy.get_param('/vehicle_params/maestro_servo_channels')

        self.timeout = rospy.Duration.from_sec(rospy.get_param('/vehicle_params/thruster_timeout_s'))

        if self.has_servos:
            # TODO: figure out a topic for servo messages
            raise ValueError('Error: Servo support is TODO')

        if self.has_thrusters:
            if len(self.thrusters) != 8:
                raise ValueError('Error: Thruster driver only supports configurations with 8 thrusters')

            self.thruster_command = {}
            for channel in self.thrusters:
                self.thruster_command[channel] = 0

            inversions = rospy.get_param('/vehicle_params/maestro_inverted_thrusters')
            self.thruster_inversions = {}
            for i, val in enumerate(inversions):
                self.thruster_inversions[self.thrusters[i]] = 1 if val == 0 else -1

            self.last_thruster_msg = None

            self.pwm_reverse = rospy.get_param('/vehicle_params/esc_pwm_reverse')
            self.pwm_forward = rospy.get_param('/vehicle_params/esc_pwm_forwards')
            if self.pwm_reverse > self.pwm_forward:
                raise ValueError('Reverse PWM must be less than forward PWM')

            topics = []
            subscribers = []
            for i in range(8):
                topicname = 'thrusters/' + str(i) + '/input'
                topics.append(topicname)
                subscribers.append(message_filters.Subscriber(topicname, FloatStamped))
            ts = message_filters.ApproximateTimeSynchronizer(subscribers, queue_size=10, slop=0.05)
            ts.registerCallback(self.thruster_callback)

    def start(self):
        r = rospy.Rate(50)  # 50 Hz
        while not rospy.is_shutdown():
            if self.has_thrusters:
                if self.last_thruster_msg is None \
                        or rospy.get_rostime() - self.last_thruster_msg > self.timeout:
                    self.thruster_command = [0] * len(self.thrusters)
                commands = {}
                for channel in self.thrusters:
                    cmd = self.speed_to_pwm(self.thruster_command[channel], channel)
                    commands[channel] = cmd
                    self.maestro.setTarget(cmd, channel)
                #print(commands)

            r.sleep()

    def thruster_callback(self, t0, t1, t2, t3, t4, t5, t6, t7):
        # TODO: find a good way to support arbitrary numbers of thruster commands
        messages = [t0, t1, t2, t3, t4, t5, t6, t7]

        for i, m in enumerate(messages):
            val = m.data
            self.thruster_command[self.thrusters[i]] = val

        self.last_thruster_msg = rospy.get_rostime()

    def speed_to_pwm(self, cmd, channel):
        if cmd < -1:
            print("Thruster Command out of range!")
            cmd = -1
        if cmd > 1:
            print("Thruster Command out of range!")
            cmd = 1
        halfrange = (self.pwm_forward - self.pwm_reverse)/2.0
        zero = halfrange + self.pwm_reverse
        return int(round(zero + halfrange * cmd * self.thruster_inversions[channel]))


def main():
    a = ActuatorController()
    rospy.init_node('actuator_controller')
    a.start()


if __name__ == '__main__':
    main()
