#!/usr/bin/env python3

import pytest
import numpy as np
from numpy.testing import assert_allclose
import math
from unittest.mock import Mock, patch
import time

# Import the modules to test
from tauv_common.state_estimator_ekf import (
    EkfControl, EkfState, DvlInput, DepthInput, EkfParams, 
    EkfStaticTransforms, Ekf, EkfHistory, MeasurementType
)

# ROS2 imports for testing
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion, Vector3
from tauv_msgs.msg import Depth, WaterlinkedDvlFrame


class TestEkfControl:
    """Test EkfControl class"""
    
    def test_from_imu_msg(self):
        """Test creating EkfControl from IMU message"""
        # Create mock IMU message
        msg = Imu()
        msg.orientation.w = 1.0
        msg.orientation.x = 0.0
        msg.orientation.y = 0.0
        msg.orientation.z = 0.0
        msg.linear_acceleration.x = 1.0
        msg.linear_acceleration.y = 2.0
        msg.linear_acceleration.z = 3.0
        msg.angular_velocity.x = 0.1
        msg.angular_velocity.y = 0.2
        msg.angular_velocity.z = 0.3
        
        control = EkfControl.from_imu_msg(msg)
        
        # Check rotation matrix (identity for unit quaternion)
        assert_allclose(control.odom_R_body, np.eye(3), atol=1e-10)
        
        # Check acceleration
        assert_allclose(control.a_body_B, np.array([1.0, 2.0, 3.0]), atol=1e-10)
        
        # Check angular velocity
        assert_allclose(control.omega_body_B, np.array([0.1, 0.2, 0.3]), atol=1e-10)
    
    def test_from_imu_msg_with_rotation(self):
        """Test creating EkfControl with non-identity rotation"""
        # Create IMU message with 90-degree rotation around Z axis
        msg = Imu()
        msg.orientation.w = 0.7071067811865476  # cos(45°)
        msg.orientation.x = 0.0
        msg.orientation.y = 0.0
        msg.orientation.z = 0.7071067811865476  # sin(45°)
        msg.linear_acceleration.x = 1.0
        msg.linear_acceleration.y = 0.0
        msg.linear_acceleration.z = 0.0
        msg.angular_velocity.x = 0.0
        msg.angular_velocity.y = 0.0
        msg.angular_velocity.z = 0.0
        
        control = EkfControl.from_imu_msg(msg)
        
        # Check rotation matrix (90-degree rotation around Z)
        expected_R = np.array([[0, -1, 0],
                              [1, 0, 0],
                              [0, 0, 1]])
        assert_allclose(control.odom_R_body, expected_R, atol=1e-10)


class TestEkfState:
    """Test EkfState class"""
    
    def test_initialization(self):
        """Test state initialization"""
        r_body_O = np.array([1.0, 2.0, 3.0])
        v_body_O = np.array([4.0, 5.0, 6.0])
        
        state = EkfState(r_body_O, v_body_O)
        
        assert_allclose(state.r_body_O(), r_body_O, atol=1e-10)
        assert_allclose(state.v_body_O(), v_body_O, atol=1e-10)
        assert_allclose(state.data, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), atol=1e-10)
    
    def test_zeros(self):
        """Test creating zero state"""
        state = EkfState.zeros()
        
        assert_allclose(state.r_body_O(), np.zeros(3), atol=1e-10)
        assert_allclose(state.v_body_O(), np.zeros(3), atol=1e-10)
        assert_allclose(state.data, np.zeros(6), atol=1e-10)
    
    def test_indexing(self):
        """Test state indexing"""
        state = EkfState(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
        
        assert state[0] == 1.0
        assert state[3] == 4.0
        assert_allclose(state[:3], np.array([1.0, 2.0, 3.0]), atol=1e-10)


class TestEkfParams:
    """Test EkfParams class"""
    
    def test_initialization(self):
        """Test parameters initialization"""
        params = EkfParams(
            initial_position_stddev_m=0.01,
            initial_velocity_stddev_mps=0.1,
            process_noise_density_pos=0.001,
            process_noise_density_vel=0.001,
            gravity=9.81,
            history_length=20,
            body_frame="body",
            dvl_frame="dvl",
            depth_frame="depth"
        )
        
        assert params.initial_position_stddev_m == 0.01
        assert params.initial_velocity_stddev_mps == 0.1
        assert params.process_noise_density_pos == 0.001
        assert params.process_noise_density_vel == 0.001
        assert params.gravity == 9.81
        assert params.history_length == 20
        assert params.body_frame == "body"
        assert params.dvl_frame == "dvl"
        assert params.depth_frame == "depth"


class TestEkfStaticTransforms:
    """Test EkfStaticTransforms class"""
    
    def test_initialization(self):
        """Test static transforms initialization"""
        r_body_depth_B = np.array([0.0, 0.0, -0.1])
        body_T_dvl = np.eye(4)
        body_T_dvl[:3, 3] = np.array([0.2, 0.0, 0.5])
        
        transforms = EkfStaticTransforms(r_body_depth_B, body_T_dvl)
        
        assert_allclose(transforms.r_body_depth_B, r_body_depth_B, atol=1e-10)
        assert_allclose(transforms.body_T_dvl, body_T_dvl, atol=1e-10)


class TestEkf:
    """Test Ekf class"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.params = EkfParams(
            initial_position_stddev_m=0.01,
            initial_velocity_stddev_mps=0.1,
            process_noise_density_pos=0.001,
            process_noise_density_vel=0.001,
            gravity=9.81,
            history_length=20,
            body_frame="body",
            dvl_frame="dvl",
            depth_frame="depth"
        )
        
        # DVL is 0.5m below and 0.2m forward of body center
        body_T_dvl = np.eye(4)
        body_T_dvl[:3, 3] = np.array([0.2, 0.0, 0.5])
        
        # Depth sensor is 0.1m above body center
        r_body_depth_B = np.array([0.0, 0.0, -0.1])
        
        self.transforms = EkfStaticTransforms(r_body_depth_B, body_T_dvl)
        self.ekf = Ekf(self.params, self.transforms)
    
    def test_initialization(self):
        """Test EKF initialization"""
        # Check gravity vector
        assert_allclose(self.ekf.a_g_O, np.array([0.0, 0.0, self.params.gravity]), atol=1e-10)
        
        # Check process noise covariance matrix
        expected_qc_diag = np.array([
            self.params.process_noise_density_pos**2,
            self.params.process_noise_density_pos**2,
            self.params.process_noise_density_pos**2,
            self.params.process_noise_density_vel**2,
            self.params.process_noise_density_vel**2,
            self.params.process_noise_density_vel**2
        ])
        assert_allclose(np.diag(self.ekf.Qc), expected_qc_diag, atol=1e-10)
        
        # Check transforms
        assert_allclose(self.ekf.r_body_depth_B, self.transforms.r_body_depth_B, atol=1e-10)
        assert_allclose(self.ekf.body_T_dvl, self.transforms.body_T_dvl, atol=1e-10)
    
    def test_predict_stationary(self):
        """Test prediction with stationary vehicle"""
        # Initial state: at origin, stationary
        x0 = EkfState(np.zeros(3), np.zeros(3))
        
        # Control input: no acceleration or rotation, body aligned with odom
        u = EkfControl(
            odom_R_body=np.eye(3),
            a_body_B=np.zeros(3),
            omega_body_B=np.zeros(3)
        )
        
        dt = 0.1
        x1 = self.ekf.predict(x0, u, dt)
        
        # Should remain at origin with zero velocity
        assert_allclose(x1.r_body_O(), np.zeros(3), atol=1e-10)
        assert_allclose(x1.v_body_O(), np.zeros(3), atol=1e-10)
    
    def test_predict_constant_velocity(self):
        """Test prediction with constant velocity"""
        # Initial state: at origin, moving at 1 m/s in x direction
        v0 = np.array([1.0, 0.0, 0.0])
        x0 = EkfState(np.zeros(3), v0)
        
        # Control input: no acceleration
        u = EkfControl(
            odom_R_body=np.eye(3),
            a_body_B=np.zeros(3),
            omega_body_B=np.zeros(3)
        )
        
        dt = 0.5
        x1 = self.ekf.predict(x0, u, dt)
        
        # Position should be velocity * time
        assert_allclose(x1.r_body_O(), np.array([0.5, 0.0, 0.0]), atol=1e-10)
        # Velocity should remain constant
        assert_allclose(x1.v_body_O(), v0, atol=1e-10)
    
    def test_predict_with_acceleration(self):
        """Test prediction with acceleration"""
        # Initial state: at origin, stationary
        x0 = EkfState(np.zeros(3), np.zeros(3))
        
        # Control input: 2 m/s² acceleration in body x direction
        u = EkfControl(
            odom_R_body=np.eye(3),
            a_body_B=np.array([2.0, 0.0, 0.0]),
            omega_body_B=np.zeros(3)
        )
        
        dt = 1.0
        x1 = self.ekf.predict(x0, u, dt)
        
        # Position: 0.5 * a * t²
        assert_allclose(x1.r_body_O(), np.array([1.0, 0.0, 0.0]), atol=1e-10)
        # Velocity: a * t
        assert_allclose(x1.v_body_O(), np.array([2.0, 0.0, 0.0]), atol=1e-10)
    
    def test_predict_covariance(self):
        """Test covariance prediction"""
        # Initial covariance
        P0 = np.eye(6)
        
        dt = 0.1
        P1 = self.ekf.predict_cov(P0, dt)
        
        # Check that uncertainty in position increased due to velocity uncertainty
        assert P1[0, 0] > P0[0, 0]
        assert P1[1, 1] > P0[1, 1]
        assert P1[2, 2] > P0[2, 2]
        
        # Check cross-correlation between position and velocity
        assert_allclose(P1[0, 3], dt, atol=1e-10)
        assert_allclose(P1[1, 4], dt, atol=1e-10)
        assert_allclose(P1[2, 5], dt, atol=1e-10)
    
    def test_h_dvl_stationary(self):
        """Test DVL measurement function with stationary vehicle"""
        # State: stationary at origin
        x = EkfState(np.zeros(3), np.zeros(3))
        
        # Control: no rotation
        u = EkfControl(
            odom_R_body=np.eye(3),
            a_body_B=np.zeros(3),
            omega_body_B=np.zeros(3)
        )
        
        v_dvl = self.ekf.h_dvl(x, u)
        
        # DVL should measure zero velocity
        assert_allclose(v_dvl, np.zeros(3), atol=1e-10)
    
    def test_h_dvl_linear_motion(self):
        """Test DVL measurement function with linear motion"""
        # State: moving at 1 m/s in odom x direction
        x = EkfState(np.zeros(3), np.array([1.0, 0.0, 0.0]))
        
        # Control: no rotation
        u = EkfControl(
            odom_R_body=np.eye(3),
            a_body_B=np.zeros(3),
            omega_body_B=np.zeros(3)
        )
        
        v_dvl = self.ekf.h_dvl(x, u)
        
        # Since DVL is aligned with body (no rotation in transform), 
        # it should measure the same velocity
        assert_allclose(v_dvl, np.array([1.0, 0.0, 0.0]), atol=1e-10)
    
    def test_h_dvl_with_angular_velocity(self):
        """Test DVL measurement function with angular velocity"""
        # Create transforms with DVL offset from body center
        body_T_dvl = np.eye(4)
        body_T_dvl[:3, 3] = np.array([1.0, 0.0, 0.0])  # DVL is 1m forward of body center
        
        transforms = EkfStaticTransforms(np.zeros(3), body_T_dvl)
        ekf = Ekf(self.params, transforms)
        
        # State: stationary
        x = EkfState(np.zeros(3), np.zeros(3))
        
        # Control: rotating at 1 rad/s about z axis
        u = EkfControl(
            odom_R_body=np.eye(3),
            a_body_B=np.zeros(3),
            omega_body_B=np.array([0.0, 0.0, 1.0])
        )
        
        v_dvl = ekf.h_dvl(x, u)
        
        # DVL should measure tangential velocity due to rotation
        # v = ω × r = [0, 0, 1] × [1, 0, 0] = [0, 1, 0]
        assert_allclose(v_dvl, np.array([0.0, 1.0, 0.0]), atol=1e-10)
    
    def test_h_depth(self):
        """Test depth measurement function"""
        # State: at depth of 5m (positive z is down)
        x = EkfState(np.array([0.0, 0.0, 5.0]), np.zeros(3))
        
        # Control: no rotation
        u = EkfControl(
            odom_R_body=np.eye(3),
            a_body_B=np.zeros(3),
            omega_body_B=np.zeros(3)
        )
        
        depth = self.ekf.h_depth(x, u)
        
        # Depth sensor is 0.1m above body center, so it should read 4.9m
        assert_allclose(depth, 4.9, atol=1e-10)
    
    def test_h_depth_with_rotation(self):
        """Test depth measurement function with rotation"""
        # State: at origin
        x = EkfState(np.zeros(3), np.zeros(3))
        
        # Control: pitched 90 degrees (nose down)
        u = EkfControl(
            odom_R_body=np.array([[0, 0, 1],
                                [0, 1, 0],
                                [-1, 0, 0]]),  # 90-degree pitch
            a_body_B=np.zeros(3),
            omega_body_B=np.zeros(3)
        )
        
        depth = self.ekf.h_depth(x, u)
        
        # When pitched 90 degrees nose down, the depth sensor (0.1m above body in body frame)
        # becomes 0.1m forward in odom frame, so depth should still be 0
        assert_allclose(depth, 0.0, atol=1e-10)
    
    def test_update_with_dvl(self):
        """Test EKF update with DVL measurement"""
        # Prior state: moving at 1 m/s in x, but with high uncertainty
        x_prior = EkfState(np.zeros(3), np.array([1.0, 0.0, 0.0]))
        P_prior = np.eye(6) * 10.0  # High uncertainty
        
        # DVL measurement: 0.8 m/s in x (slightly different from prior)
        z_dvl = np.array([0.8, 0.0, 0.0])
        R_dvl = np.eye(3) * 0.01  # Low measurement noise
        
        # Expected measurement based on prior
        u = EkfControl(
            odom_R_body=np.eye(3),
            a_body_B=np.zeros(3),
            omega_body_B=np.zeros(3)
        )
        z_expected = self.ekf.h_dvl(x_prior, u)
        
        # Measurement Jacobian for DVL
        H_dvl = self.ekf.F_dvl
        
        x_post, P_post = self.ekf.update(x_prior, P_prior, z_dvl, R_dvl, z_expected, H_dvl)
        
        # Posterior velocity should be closer to measurement
        assert abs(x_post.v_body_O()[0] - 0.8) < abs(x_prior.v_body_O()[0] - 0.8)
        
        # Posterior covariance should be smaller
        assert P_post[3, 3] < P_prior[3, 3]
    
    def test_update_with_depth(self):
        """Test EKF update with depth measurement"""
        # Prior state: at 5m depth with high uncertainty
        x_prior = EkfState(np.array([0.0, 0.0, 5.0]), np.zeros(3))
        P_prior = np.eye(6) * 10.0
        
        # Depth measurement: 4.5m
        z_depth = np.array([4.5])
        R_depth = np.array([[0.01]])
        
        # Expected measurement
        u = EkfControl(
            odom_R_body=np.eye(3),
            a_body_B=np.zeros(3),
            omega_body_B=np.zeros(3)
        )
        z_expected = np.array([self.ekf.h_depth(x_prior, u)])
        
        # Measurement Jacobian for depth (only sensitive to z position)
        H_depth = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        
        x_post, P_post = self.ekf.update(x_prior, P_prior, z_depth, R_depth, z_expected, H_depth)
        
        # Posterior depth should be closer to measurement (accounting for sensor offset)
        # Measurement is 4.5m, sensor is 0.1m above body, so body should be at 4.6m
        assert abs(x_post.r_body_O()[2] - 4.6) < abs(x_prior.r_body_O()[2] - 4.6)
        
        # Posterior covariance in z should be smaller
        assert P_post[2, 2] < P_prior[2, 2]
    
    def test_update_singular_covariance(self):
        """Test EKF update with singular covariance"""
        x_prior = EkfState(np.zeros(3), np.zeros(3))
        P_prior = np.eye(6)
        
        # Create a more clearly singular measurement covariance
        z = np.array([1.0])
        R = np.array([[0.0]])
        z_expected = np.array([0.0])
        H = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        
        # The innovation covariance Sk = H @ P_prior @ H.T + R
        # With R = 0 and H = [0,0,1,0,0,0], Sk = P_prior[2,2] = 1.0
        # This should not be singular, so let's create a truly singular case
        H_singular = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])  # Zero measurement Jacobian
        R_singular = np.array([[0.0]])
        
        # This should raise an error due to singular innovation covariance
        with pytest.raises((ValueError, np.linalg.LinAlgError)):
            self.ekf.update(x_prior, P_prior, z, R_singular, z_expected, H_singular)


class TestEkfHistory:
    """Test EkfHistory class"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.params = EkfParams(
            initial_position_stddev_m=0.01,
            initial_velocity_stddev_mps=0.1,
            process_noise_density_pos=0.001,
            process_noise_density_vel=0.001,
            gravity=9.81,
            history_length=20,
            body_frame="body",
            dvl_frame="dvl",
            depth_frame="depth"
        )
        
        self.transforms = EkfStaticTransforms(
            r_body_depth_B=np.array([0.0, 0.0, -0.1]),
            body_T_dvl=np.eye(4)
        )
        self.ekf = Ekf(self.params, self.transforms)
    
    def create_test_inputs(self):
        """Create test inputs for history initialization"""
        t_depth = 100.0
        t_dvl = 100.01
        t_imu = 100.02
        
        depth = DepthInput(z=5.0, R=0.01)
        dvl = DvlInput(
            v_dvl_V=np.array([1.0, 0.5, 0.2]),
            R=np.eye(3) * 0.01
        )
        imu = EkfControl(
            odom_R_body=np.eye(3),
            a_body_B=np.array([0.1, 0.0, 0.0]),
            omega_body_B=np.array([0.0, 0.0, 0.1])
        )
        
        return t_depth, t_dvl, t_imu, depth, dvl, imu
    
    def test_initialization_success(self):
        """Test successful initialization of EkfHistory"""
        t_depth, t_dvl, t_imu, depth, dvl, imu = self.create_test_inputs()
        
        history = EkfHistory.try_new(t_depth, t_dvl, t_imu, depth, dvl, imu, self.params)
        
        # Check that timestamps are stored correctly
        assert history.last_depth_t == t_depth
        assert history.last_dvl_t == t_dvl
        assert history.last_imu_t == t_imu
        
        # Check that control history has one entry
        assert len(history.control_history) == 1
        assert history.control_history[0].t == t_imu
        
        # Check that state history has two entries (depth and dvl)
        assert len(history.state_history) == 2
        assert t_depth in history.state_history
        assert t_dvl in history.state_history
        
        # Check that initial state is set correctly based on depth
        state_est = history.state_history[t_depth][1]
        assert_allclose(state_est.state.r_body_O()[2], depth.z, atol=1e-10)
        assert_allclose(state_est.state.v_body_O(), np.zeros(3), atol=1e-10)
    
    def test_initialization_measurements_too_far_apart(self):
        """Test that initialization fails when measurements are too far apart"""
        t_depth = 100.0
        t_dvl = 100.01
        t_imu = 101.0  # Too far apart
        
        depth = DepthInput(z=5.0, R=0.01)
        dvl = DvlInput(
            v_dvl_V=np.array([1.0, 0.5, 0.2]),
            R=np.eye(3) * 0.01
        )
        imu = EkfControl(
            odom_R_body=np.eye(3),
            a_body_B=np.array([0.1, 0.0, 0.0]),
            omega_body_B=np.array([0.0, 0.0, 0.1])
        )
        
        with pytest.raises(ValueError, match="too far apart"):
            EkfHistory.try_new(t_depth, t_dvl, t_imu, depth, dvl, imu, self.params)
    
    def test_add_imu_measurement_success(self):
        """Test successful addition of IMU measurement"""
        t_depth, t_dvl, t_imu, depth, dvl, imu = self.create_test_inputs()
        history = EkfHistory.try_new(t_depth, t_dvl, t_imu, depth, dvl, imu, self.params)
        
        # Add a new IMU measurement
        new_imu_time = t_imu + 0.05
        new_imu = EkfControl(
            odom_R_body=np.eye(3),
            a_body_B=np.array([0.2, 0.0, 0.0]),
            omega_body_B=np.array([0.0, 0.0, 0.2])
        )
        
        history.add_imu_measurement(new_imu_time, new_imu)
        
        # Check that the measurement was added
        assert len(history.control_history) == 2
        assert history.last_imu_t == new_imu_time
        assert history.control_history[-1].t == new_imu_time
    
    def test_add_imu_measurement_not_newer_than_last_imu(self):
        """Test that adding older IMU measurement is rejected"""
        t_depth, t_dvl, t_imu, depth, dvl, imu = self.create_test_inputs()
        history = EkfHistory.try_new(t_depth, t_dvl, t_imu, depth, dvl, imu, self.params)
        
        # Try to add an IMU measurement that's not newer
        old_imu_time = t_imu - 0.01
        new_imu = EkfControl(
            odom_R_body=np.eye(3),
            a_body_B=np.array([0.2, 0.0, 0.0]),
            omega_body_B=np.array([0.0, 0.0, 0.2])
        )
        
        with pytest.raises(ValueError, match="not newer than last IMU"):
            history.add_imu_measurement(old_imu_time, new_imu)
    
    def test_add_imu_measurement_not_newer_than_last_dvl(self):
        """Test that adding IMU measurement older than last DVL is rejected"""
        t_depth, t_dvl, t_imu, depth, dvl, imu = self.create_test_inputs()
        history = EkfHistory.try_new(t_depth, t_dvl, t_imu, depth, dvl, imu, self.params)
        
        # Add a new IMU measurement after the initial one
        new_imu_time = t_imu + 0.05
        history.add_imu_measurement(new_imu_time, imu)
        
        # Manually update last_dvl_t to simulate a DVL measurement
        history.last_dvl_t = new_imu_time + 0.05
        
        # Try to add an IMU measurement that's older than the new DVL time
        old_imu_time = new_imu_time + 0.025
        test_imu = EkfControl(
            odom_R_body=np.eye(3),
            a_body_B=np.array([0.2, 0.0, 0.0]),
            omega_body_B=np.array([0.0, 0.0, 0.2])
        )
        
        with pytest.raises(ValueError, match="not newer than last DVL"):
            history.add_imu_measurement(old_imu_time, test_imu)
    
    def test_add_imu_measurement_max_history_limit(self):
        """Test that control history respects maximum history length limit"""
        params = EkfParams(
            initial_position_stddev_m=0.01,
            initial_velocity_stddev_mps=0.1,
            process_noise_density_pos=0.001,
            process_noise_density_vel=0.001,
            gravity=9.81,
            history_length=2,  # Small limit for testing
            body_frame="body",
            dvl_frame="dvl",
            depth_frame="depth"
        )
        
        t_depth, t_dvl, t_imu, depth, dvl, imu = self.create_test_inputs()
        history = EkfHistory.try_new(t_depth, t_dvl, t_imu, depth, dvl, imu, params)
        
        # Add IMU measurements up to and beyond the limit
        current_time = t_imu
        for i in range(3):
            current_time = current_time + 0.05
            history.add_imu_measurement(current_time, imu)
        
        # Should only keep the last 2 measurements
        assert len(history.control_history) == 2
        assert history.control_history[0].t == t_imu + 0.1
    
    def test_add_depth_measurement_success(self):
        """Test successful addition of depth measurement"""
        t_depth, t_dvl, t_imu, depth, dvl, imu = self.create_test_inputs()
        history = EkfHistory.try_new(t_depth, t_dvl, t_imu, depth, dvl, imu, self.params)
        
        # Add an IMU measurement that's newer than the DVL to satisfy the constraint
        new_imu_time = t_dvl + 0.05
        history.add_imu_measurement(new_imu_time, imu)
        
        # Add a new depth measurement after the IMU
        new_depth_time = new_imu_time + 0.05
        new_depth = DepthInput(z=6.0, R=0.02)
        
        history.add_depth_measurement(new_depth_time, new_depth, self.ekf)
        
        # Check that the measurement was added
        assert history.last_depth_t == new_depth_time
        assert new_depth_time in history.state_history
        
        # After cleanup, only states after the oldest control input are kept
        assert len(history.state_history) == 1
        assert new_depth_time in history.state_history
    
    def test_add_depth_measurement_not_newer_than_last_depth(self):
        """Test that adding older depth measurement is rejected"""
        t_depth, t_dvl, t_imu, depth, dvl, imu = self.create_test_inputs()
        history = EkfHistory.try_new(t_depth, t_dvl, t_imu, depth, dvl, imu, self.params)
        
        # Try to add a depth measurement that's not newer
        old_depth_time = t_depth - 0.01
        new_depth = DepthInput(z=6.0, R=0.02)
        
        with pytest.raises(ValueError, match="not newer than last depth"):
            history.add_depth_measurement(old_depth_time, new_depth, self.ekf)
    
    def test_add_depth_measurement_not_newer_than_last_dvl(self):
        """Test that adding depth measurement older than last DVL is rejected"""
        t_depth, t_dvl, t_imu, depth, dvl, imu = self.create_test_inputs()
        history = EkfHistory.try_new(t_depth, t_dvl, t_imu, depth, dvl, imu, self.params)
        
        # Try to add a depth measurement that's older than the last DVL
        old_depth_time = t_dvl - 0.005
        new_depth = DepthInput(z=6.0, R=0.02)
        
        with pytest.raises(ValueError, match="not newer than last DVL"):
            history.add_depth_measurement(old_depth_time, new_depth, self.ekf)
    
    def test_find_closest_control(self):
        """Test finding closest control input by timestamp"""
        t_depth, t_dvl, t_imu, depth, dvl, imu = self.create_test_inputs()
        history = EkfHistory.try_new(t_depth, t_dvl, t_imu, depth, dvl, imu, self.params)
        
        # Add more control inputs
        t_imu2 = t_imu + 0.1
        t_imu3 = t_imu + 0.2
        history.add_imu_measurement(t_imu2, imu)
        history.add_imu_measurement(t_imu3, imu)
        
        # Test finding closest control
        query_time = t_imu + 0.13  # Closer to t_imu2 (0.1) than t_imu3 (0.2)
        closest = history._find_closest_control(query_time)
        assert abs(closest.t - t_imu2) < 1e-10  # Should be closest to t_imu2
        
        # Test exact match
        exact_match = history._find_closest_control(t_imu2)
        assert exact_match.t == t_imu2
        
        # Test before first control
        before_first = t_imu - 0.05
        first_control = history._find_closest_control(before_first)
        assert first_control.t == t_imu
        
        # Test after last control
        after_last = t_imu3 + 0.05
        last_control = history._find_closest_control(after_last)
        assert last_control.t == t_imu3
    
    def test_find_closest_control_empty_history(self):
        """Test error handling when control history is empty"""
        history = EkfHistory(max_control_history=10)
        
        query_time = 100.0
        with pytest.raises(ValueError, match="No control inputs"):
            history._find_closest_control(query_time)
    
    def test_find_latest_state_before(self):
        """Test finding latest state estimate before given time"""
        t_depth, t_dvl, t_imu, depth, dvl, imu = self.create_test_inputs()
        history = EkfHistory.try_new(t_depth, t_dvl, t_imu, depth, dvl, imu, self.params)
        
        # Add an IMU measurement that's newer than the DVL
        new_imu_time = t_dvl + 0.05
        history.add_imu_measurement(new_imu_time, imu)
        
        # Add a new depth measurement
        new_depth_time = new_imu_time + 0.05
        new_depth = DepthInput(z=6.0, R=0.02)
        history.add_depth_measurement(new_depth_time, new_depth, self.ekf)
        
        # After cleanup, the only remaining state is the new depth measurement
        # So we can only test finding states before times after the new depth measurement
        late_query_time = new_depth_time + 0.05
        late_state_time, _ = history._find_latest_state_before(late_query_time)
        assert late_state_time == new_depth_time
        
        # Test that querying before the new depth time fails (no states before it)
        early_query_time = new_depth_time - 0.01
        with pytest.raises(ValueError, match="No state estimate found"):
            history._find_latest_state_before(early_query_time)
    
    def test_get_latest_state(self):
        """Test retrieval of most recent state estimate"""
        t_depth, t_dvl, t_imu, depth, dvl, imu = self.create_test_inputs()
        history = EkfHistory.try_new(t_depth, t_dvl, t_imu, depth, dvl, imu, self.params)
        
        # Initially, latest should be dvl (since it's later than depth)
        latest_time, _ = history.get_latest_state()
        assert latest_time == t_dvl
        
        # Add an IMU measurement that's newer than the DVL
        new_imu_time = t_dvl + 0.05
        history.add_imu_measurement(new_imu_time, imu)
        
        # Add a new depth measurement
        new_depth_time = new_imu_time + 0.05
        new_depth = DepthInput(z=6.0, R=0.02)
        history.add_depth_measurement(new_depth_time, new_depth, self.ekf)
        
        # Now latest should be the new depth
        latest_time, _ = history.get_latest_state()
        assert latest_time == new_depth_time
    
    def test_state_cleanup(self):
        """Test automatic cleanup of old state estimates"""
        params = EkfParams(
            initial_position_stddev_m=0.01,
            initial_velocity_stddev_mps=0.1,
            process_noise_density_pos=0.001,
            process_noise_density_vel=0.001,
            gravity=9.81,
            history_length=2,  # Small limit for testing
            body_frame="body",
            dvl_frame="dvl",
            depth_frame="depth"
        )
        
        transforms = EkfStaticTransforms(
            r_body_depth_B=np.array([0.0, 0.0, -0.1]),
            body_T_dvl=np.eye(4)
        )
        ekf = Ekf(params, transforms)
        
        t_depth, t_dvl, t_imu, depth, dvl, imu = self.create_test_inputs()
        history = EkfHistory.try_new(t_depth, t_dvl, t_imu, depth, dvl, imu, params)
        
        # Add several IMU measurements to trigger cleanup
        current_time = t_imu
        for _ in range(5):
            current_time = current_time + 0.05
            history.add_imu_measurement(current_time, imu)
        
        # Add a depth measurement to trigger state cleanup
        new_depth_time = current_time + 0.05
        new_depth = DepthInput(z=6.0, R=0.02)
        history.add_depth_measurement(new_depth_time, new_depth, ekf)
        
        # Check that old states were cleaned up
        # Should only keep states that are after the oldest control input
        oldest_control_time = history.control_history[0].t
        for state_time in history.state_history.keys():
            assert state_time >= oldest_control_time
    
    def test_measurement_type_storage(self):
        """Test that measurement types are correctly stored and retrieved"""
        t_depth, t_dvl, t_imu, depth, dvl, imu = self.create_test_inputs()
        history = EkfHistory.try_new(t_depth, t_dvl, t_imu, depth, dvl, imu, self.params)
        
        # Check that measurement types are stored correctly
        depth_measurement, _ = history.state_history[t_depth]
        assert depth_measurement == MeasurementType.DEPTH
        
        dvl_measurement, _ = history.state_history[t_dvl]
        assert dvl_measurement == MeasurementType.DVL
        
        # Add an IMU measurement that's newer than the DVL
        new_imu_time = t_dvl + 0.05
        history.add_imu_measurement(new_imu_time, imu)
        
        # Add a new depth measurement and check its type
        new_depth_time = new_imu_time + 0.05
        new_depth = DepthInput(z=6.0, R=0.02)
        history.add_depth_measurement(new_depth_time, new_depth, self.ekf)
        
        new_depth_measurement, _ = history.state_history[new_depth_time]
        assert new_depth_measurement == MeasurementType.DEPTH
    
    def test_find_latest_state_before_no_state(self):
        """Test error handling when no state estimates exist before queried timestamp"""
        history = EkfHistory(max_control_history=10)
        
        # Query for a time before all states
        early_time = 100.0
        with pytest.raises(ValueError, match="No state estimate found"):
            history._find_latest_state_before(early_time)


if __name__ == '__main__':
    pytest.main([__file__]) 