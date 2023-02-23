import rospy
import numpy as np
from typing import Optional

from dynamics.dynamics import Dynamics
from geometry_msgs.msg import Twist, Wrench, WrenchStamped, Vector3, Quaternion
from tauv_msgs.msg import ControllerCommand, NavigationState, ControllerDebug
from tauv_msgs.srv import TuneDynamics, TuneDynamicsRequest, TuneDynamicsResponse, TuneController, TuneControllerRequest, TuneControllerResponse
from tauv_util.types import tl, tm
from tauv_util.pid import PID, pi_clip
from tauv_util.transforms import euler_velocity_to_axis_velocity

from tauv_alarms import Alarm, AlarmClient


class Controller:

    def __init__(self):
        self._ac = AlarmClient()

        self._load_config()

        self._dt: float = 1.0 / self._frequency
        self._navigation_state: Optional[NavigationState] = None
        self._controller_command: Optional[ControllerCommand] = None
        self._dyn: Dynamics = Dynamics(
            m=self._m,
            v=self._v,
            rho=self._rho,
            r_G=self._r_G,
            r_B=self._r_B,
            I=self._I,
            D=self._D,
            D2=self._D2,
            Ma=self._Ma,
        )

        self._navigation_state_sub: rospy.Subscriber = rospy.Subscriber('gnc/navigation_state', NavigationState, self._handle_navigation_state)
        self._controller_command_sub: rospy.Subscriber = rospy.Subscriber('gnc/controller_command', ControllerCommand, self._handle_controller_command)
        self._controller_debug_pub = rospy.Publisher('gnc/controller_debug', ControllerDebug, queue_size=10)
        self._wrench_pub: rospy.Publisher = rospy.Publisher('gnc/target_wrench', WrenchStamped, queue_size=10)
        self._tune_dynamics_srv: rospy.Service = rospy.Service('gnc/tune_dynamics', TuneDynamics, self._handle_tune_dynamics)
        self._tune_controller_srv: rospy.Service = rospy.Service('gnc/tune_controller', TuneController, self._handle_tune_controller)

        self._build_pids()

    def start(self):
        rospy.Timer(rospy.Duration.from_sec(self._dt), self._update)
        rospy.spin()

    def _update(self, timer_event):
        if self._navigation_state is None or self._controller_command is None:
            return

        eta = np.concatenate((
            tl(self._navigation_state.position),
            tl(self._navigation_state.orientation)
        ))

        euler_velocity = tl(self._navigation_state.euler_velocity)
        orientation = tl(self._navigation_state.orientation)
        axis_velocity = euler_velocity_to_axis_velocity(orientation, euler_velocity)
        v = np.concatenate((
            tl(self._navigation_state.linear_velocity),
            axis_velocity
        ))

        roll_error = tl(self._navigation_state.orientation)[0]
        pitch_error = tl(self._navigation_state.orientation)[1]

        roll_effort = self._pids[0](roll_error)
        pitch_effort = self._pids[1](pitch_error)

        vd = np.array([
            self._controller_command.a_x,
            self._controller_command.a_y,
            self._controller_command.a_z,
            roll_effort,
            pitch_effort,
            self._controller_command.a_yaw,
        ])

        tau = self._dyn.compute_tau(eta, v, vd)

        # TODO: Fix max wrench clamping
        # while not np.allclose(np.minimum(np.abs(tau), self._max_wrench), np.abs(tau)):
        #     tau = 0.75 * tau
        #     # TODO FIX THIS
        #     # tau = self._dyn.compute_tau(eta, v, vd)
        # tau = np.sign(tau) * np.minimum(np.abs(tau), self._max_wrench)

        # Need to LPF the wrench here with a certain time constant
        wrench: WrenchStamped = WrenchStamped()
        wrench.header.frame_id = f'{self._tf_namespace}/vehicle'
        wrench.header.stamp = rospy.Time.now()
        wrench.wrench.force = Vector3(tau[0], tau[1], tau[2])
        wrench.wrench.torque = Vector3(tau[3], tau[4], tau[5])
        self._wrench_pub.publish(wrench)

        controller_debug: ControllerDebug = ControllerDebug()
        controller_debug.roll = tl(self._navigation_state.orientation)[0]
        controller_debug.error_roll = roll_error
        controller_debug.integral_roll = self._pids[0]._integral
        controller_debug.derivative_roll = self._pids[0]._derivative
        controller_debug.effort_roll = roll_effort
        controller_debug.pitch = tl(self._navigation_state.orientation)[1]
        controller_debug.error_pitch = pitch_error
        controller_debug.integral_pitch = self._pids[1]._integral
        controller_debug.derivative_pitch = self._pids[1]._derivative
        controller_debug.effort_pitch = pitch_effort
        self._controller_debug_pub.publish(controller_debug)

        self._ac.clear(Alarm.CONTROLLER_NOT_INITIALIZED)

    def _handle_navigation_state(self, msg: NavigationState):
        self._navigation_state = msg

    def _handle_controller_command(self, msg: ControllerCommand):
        self._controller_command = msg

    def _handle_tune_dynamics(self, req: TuneDynamicsRequest) -> TuneDynamicsResponse:
        if req.tuning.update_mass:
            self._m = req.tuning.mass

        if req.tuning.update_volume:
            self._v = req.tuning.volume

        if req.tuning.update_water_density:
            self._rho = req.tuning.water_density

        if req.tuning.update_center_of_gravity:
            self._r_G = req.tuning.center_of_gravity

        if req.tuning.update_center_of_buoyancy:
            self._r_B = req.tuning.center_of_buoyancy

        if req.tuning.update_moments:
            self._I = req.tuning.moments

        if req.tuning.update_linear_damping:
            self._D = req.tuning.linear_damping

        if req.tuning.update_quadratic_damping:
            self._D2 = req.tuning.quadratic_damping

        if req.tuning.update_added_mass:
            self._Ma = req.tuning.added_mass

        self._dyn: Dynamics = Dynamics(
            m = self._m,
            v = self._v,
            rho = self._rho,
            r_G = self._r_G,
            r_B = self._r_B,
            I = self._I,
            D = self._D,
            D2 = self._D2,
            Ma = self._Ma,
        )
        return TuneDynamicsResponse(True)

    def _handle_tune_controller(self, req: TuneControllerRequest) -> TuneControllerResponse:
        fields = {"roll": 0, "pitch": 1}

        for tuning in req.tunings:
            field = fields.get(tuning.axis)

            if field is None:
                return TuneControllerResponse(False)

            self._kp[field] = float(tuning.kp)
            self._ki[field] = float(tuning.ki)
            self._kd[field] = float(tuning.kd)
            self._tau[field] = float(tuning.tau)
            self._limits[field] = np.array(tuning.limits, dtype=np.float64)
        self._build_pids()
        return TuneControllerResponse(True)

    def _build_pids(self):
        pids = []

        for i in range(2):
            pid = PID(
                Kp=self._kp[i],
                Ki=self._ki[i],
                Kd=self._kd[i],
                error_map=pi_clip,
                output_limits=self._limits[i],
                proportional_on_measurement=False,
                sample_time=self._dt,
                d_alpha=self._dt / self._tau[i] if self._tau[i] > 0 else 1
            )
            pids.append(pid)

        self._pids = pids

    def _load_config(self):
        self._tf_namespace = rospy.get_param('tf_namespace')
        self._frequency = rospy.get_param('~frequency')
        self._kp = np.array(rospy.get_param('~kp'))
        self._ki = np.array(rospy.get_param('~ki'))
        self._kd = np.array(rospy.get_param('~kd'))
        self._tau = np.array(rospy.get_param('~tau'))
        self._limits = np.array(rospy.get_param('~limits'))

        self._max_wrench = np.array(rospy.get_param('~max_wrench'))

        self._m = rospy.get_param('~dynamics/mass')
        self._v = rospy.get_param('~dynamics/volume')
        self._rho = rospy.get_param('~dynamics/water_density')
        self._r_G = np.array(rospy.get_param('~dynamics/center_of_gravity'))
        self._r_B = np.array(rospy.get_param('~dynamics/center_of_buoyancy'))
        self._I = np.array(rospy.get_param('~dynamics/moments'))
        self._D = np.array(rospy.get_param('~dynamics/linear_damping'))
        self._D2 = np.array(rospy.get_param('~dynamics/quadratic_damping'))
        self._Ma = np.array(rospy.get_param('~dynamics/added_mass'))



def main():
    rospy.init_node('controller')
    c = Controller()
    c.start()