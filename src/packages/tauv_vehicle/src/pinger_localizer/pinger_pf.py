import numpy as np
import rospy
import tf
from filterpy.monte_carlo import systematic_resample
from dataclasses import dataclass
from geometry_msgs.msg import PoseWithCovarianceStamped, Vector3

@dataclass
class UniformInitParams:
    x_range: tuple(float, float)
    y_range: tuple(float, float)
    z_range: tuple(float, float)

# @dataclass
# class GaussianInitParams:
#     xyz_mean: tuple(float, float, float)
#     xyz_var: tuple(float, float, float)


class ParticleFilter:
    def __init__(self, n_particles, init_params, neff_thresh=None, predict_stddev=0.0254) -> None:
        self._n_particles = n_particles

        if isinstance(init_params, UniformInitParams):
            self._particles = self._sample_uniform_particles(
                init_params.x_range,
                init_params.y_range,
                init_params.z_range
            )
        elif isinstance(init_params, GaussianInitParams):
            self._particles = self._sample_gaussian_particles(
                init_params.xyz_mean,
                init_params.xyz_var
            )
        else: raise RuntimeError

        self._weights = np.ones(self._n_particles) / self._n_particles
        self._neff_thresh = neff_thresh
        if self._neff_thresh is None:
            self._neff_thresh = self._n_particles / 2

        self._predict_noise = predict_stddev
    
    def update(self, vehicle_pos, meas_direction):
        self.predict_particles()
        self.reweight_particles(vehicle_pos, meas_direction)

        if self._neff(self._weights) < self._neff_thresh:
            self.resample_particles()

    def _sample_uniform_particles(self, x_range, y_range, z_range):
        particles = np.empty(shape=(self._n_particles, 3))

        particles[:, 0] = np.random.uniform(low=x_range[0], high=x_range[1], size=self._n_particles)
        particles[:, 1] = np.random.uniform(low=y_range[0], high=y_range[1], size=self._n_particles)
        particles[:, 2] = np.random.uniform(low=z_range[0], high=z_range[1], size=self._n_particles)

        return particles
    
    def _predict_particles(self):
        noise = np.random.normal(loc=0, scale=self._predict_noise, size=(self._n_particles, 3))
        self._particles += noise

    def _sample_gaussian_particles(self, xyz_mean, xyz_variance):
        assert xyz_variance.shape == (3, 3)
        
        return np.random.multivariate_normal(xyz_mean, xyz_variance, size=self._n_particles)
    
    def estimate_state(self):
        position_mean = np.mean(self._particles, axis=0)
        position_variance = np.var(self._particles, axis=0)

        return position_mean, position_variance

    def _reweight_particles(self, vehicle_pos, meas_direction):
        for (i, particle) in enumerate(self._particles):
            pred_direction = particle - vehicle_pos

            pred_direction_norm = np.linalg.norm(pred_direction)
            meas_direction_norm = np.linalg.norm(meas_direction)

            cos_similarity = np.dot(meas_direction, pred_direction) / (pred_direction_norm * meas_direction_norm)
            cos_similarity = (cos_similarity + 1.0) / 2.0

            self._weights[i] *= cos_similarity
        
        self._weights += 1e-300
        self._weights /= np.sum(self._weights)
    
    def _resample_particles(self):
        indices = systematic_resample(self._weights)
        self.resample_from_index(indices)

    def _resample_from_index(self, indices):
        self._particles[:] = self._particles[indices]
        self._weights.resize(len(self._particles))
        self._weights.fill (1.0 / len(self._weights))

    def _neff(self):
        return 1 / np.sum(np.square(self._weights))


class PingerLocalizerPF:
    def __init__(self) -> None:
        self._load_config()
        self._direction_sub = rospy.Subscriber(f'{self._vehicle_ns}/pinger_localizer/direction', Vector3Stamped, self._handle_direction)
        self._pinger_loc_pub = rospy.Publisher(f'{self._vehicle_ns}/pinger_localizer/pinger_loc_pf', PoseWithCovarianceStamped)
        init_params = UniformInitParams(self._x_range, self._y_range, self._z_range)

        self._pf = ParticleFilter(self._n_particles, init_params)

    def _handle_direction(self, direction: Vector3Stamped):
        direction_time = rospy.Time(
            secs=direction.header.stamp.secs,
            nsecs=direction.header.stamp.nsecs
        )

        world_direction = tf.TransformerROS.transformVector3(target_frame='kf/world', v3s=direction)
        (vehicle_pos, _) = tf.Transformer.lookupTransform(target_frame='kf/odom', source_frame='kf/vehicle', time=direction_time)

        self._pf.update(vehicle_pos, world_direction)
        pinger_mean, pinger_var = self._pf.estimate_state()
    
    def load_config(self):
        self._vehicle_ns = rospy.get_param("namespace")
        self._n_particles = rospy.get_param(f"{self._vehicle_ns}/pf__particles")
        self._x_range = rospy.get_param(f"{self._vehicle_ns}/pf_x_range")
        self._y_range = rospy.get_param(f"{self._vehicle_ns}/pf_y_range")
        self._z_range = rospy.get_param(f"{self._vehicle_ns}/pf_z_range")