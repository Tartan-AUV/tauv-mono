#!/usr/bin/env python3

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from dataclasses import replace

from geometry_msgs.msg import AccelStamped, WrenchStamped, Wrench, Vector3, Accel
from tauv_msgs.msg import NavigationState
from std_msgs.msg import Header

# Import the classes we want to test
from tauv_common.controller import INDIParams, INDIController


class TestINDIParams(unittest.TestCase):
    """Test cases for the INDIParams dataclass."""
    
    def test_default_params_creation(self):
        """Test that default parameters are created correctly."""
        params = INDIParams.default()
        
        # Check that control effectiveness matrix is 6x6
        self.assertEqual(params.control_effectiveness.shape, (6, 6))
        
        # Check that diagonal structure is reasonable
        # Upper 3x3 should be related to force/mass
        mass_related = params.control_effectiveness[0:3, 0:3]
        self.assertTrue(np.allclose(mass_related, np.eye(3) * (1.0/50.0), rtol=1e-10))
        
        # Lower 3x3 should be related to inertia inverse
        inertia_related = params.control_effectiveness[3:6, 3:6]
        expected_inertia_inv = np.linalg.inv(np.diag([10.0, 15.0, 12.0]))
        self.assertTrue(np.allclose(inertia_related, expected_inertia_inv, rtol=1e-10))
        
        # Off-diagonal blocks should be zero
        self.assertTrue(np.allclose(params.control_effectiveness[0:3, 3:6], 0))
        self.assertTrue(np.allclose(params.control_effectiveness[3:6, 0:3], 0))
        
    def test_default_parameter_values(self):
        """Test that default parameter values are reasonable."""
        params = INDIParams.default()
        
        self.assertEqual(params.accel_filter_alpha, 0.8)
        self.assertEqual(params.max_force, 100.0)
        self.assertEqual(params.max_torque, 50.0)
        self.assertEqual(params.kp_pos, 1.0)
        self.assertEqual(params.kd_pos, 0.5)
        
    def test_custom_params_creation(self):
        """Test creating custom parameters."""
        custom_G = np.eye(6) * 0.1
        params = INDIParams(
            control_effectiveness=custom_G,
            accel_filter_alpha=0.5,
            max_force=200.0,
            max_torque=75.0
        )
        
        np.testing.assert_array_equal(params.control_effectiveness, custom_G)
        self.assertEqual(params.accel_filter_alpha, 0.5)
        self.assertEqual(params.max_force, 200.0)
        self.assertEqual(params.max_torque, 75.0)


class TestINDIControllerLogic(unittest.TestCase):
    """Test the core logic of the INDI controller without ROS2 infrastructure."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock ROS2 node initialization
        with patch('tauv_common.controller.Node.__init__'), \
             patch('tauv_common.controller.Node.create_subscription'), \
             patch('tauv_common.controller.Node.create_publisher'), \
             patch('tauv_common.controller.Node.create_timer'), \
             patch('tauv_common.controller.Node.get_logger') as mock_logger:
            
            mock_logger.return_value = Mock()
            self.controller = INDIController()
            
        # Set up test data
        self.test_params = INDIParams.default()
        self.controller.params = self.test_params
        
    def test_controller_initialization(self):
        """Test that controller initializes properly."""
        self.assertIsNotNone(self.controller.params)
        self.assertIsNotNone(self.controller._G_inv)
        self.assertIsNone(self.controller._previous_wrench)
        self.assertIsNone(self.controller._filtered_accel)
        self.assertIsNone(self.controller._nav_state)
        self.assertIsNone(self.controller._accel_command)
        
    def test_control_effectiveness_matrix_inverse(self):
        """Test that control effectiveness matrix inverse is computed correctly."""
        G = self.controller.params.control_effectiveness
        G_inv = self.controller._G_inv
        
        # Check that G * G_inv ≈ I
        identity_test = G @ G_inv
        np.testing.assert_array_almost_equal(identity_test, np.eye(6), decimal=10)
        
    def test_singular_matrix_handling(self):
        """Test handling of singular control effectiveness matrix."""
        # Create a singular matrix
        singular_G = np.zeros((6, 6))
        
        with patch('tauv_common.controller.Node.__init__'), \
             patch('tauv_common.controller.Node.create_subscription'), \
             patch('tauv_common.controller.Node.create_publisher'), \
             patch('tauv_common.controller.Node.create_timer'), \
             patch('tauv_common.controller.Node.get_logger') as mock_logger:
            
            mock_logger.return_value = Mock()
            
            # Temporarily replace the default params
            with patch.object(INDIParams, 'default') as mock_default:
                mock_default.return_value = INDIParams(control_effectiveness=singular_G)
                controller = INDIController()
                
                # Should use pseudo-inverse for singular matrix
                self.assertIsNotNone(controller._G_inv)
                # Check that it's the pseudo-inverse
                expected_pinv = np.linalg.pinv(singular_G)
                np.testing.assert_array_almost_equal(controller._G_inv, expected_pinv)
                
    def test_apply_limits_within_bounds(self):
        """Test _apply_limits when wrench is within limits."""
        wrench_input = np.array([[10.0], [20.0], [30.0], [5.0], [10.0], [15.0]])
        
        limited_wrench = self.controller._apply_limits(wrench_input)
        
        # Should be unchanged since within limits
        np.testing.assert_array_equal(limited_wrench, wrench_input)
        
    def test_apply_limits_force_exceeded(self):
        """Test _apply_limits when force limit is exceeded."""
        # Create wrench with force magnitude > max_force
        large_force = np.array([[80.0], [60.0], [0.0], [5.0], [10.0], [15.0]])  # |F| = 100
        self.controller.params.max_force = 50.0
        
        limited_wrench = self.controller._apply_limits(large_force)
        
        # Force should be scaled down
        force_part = limited_wrench[0:3]
        force_magnitude = np.linalg.norm(force_part)
        self.assertAlmostEqual(force_magnitude, 50.0, places=5)
        
        # Direction should be preserved
        original_direction = large_force[0:3] / np.linalg.norm(large_force[0:3])
        limited_direction = force_part / np.linalg.norm(force_part)
        np.testing.assert_array_almost_equal(original_direction, limited_direction)
        
        # Torque should be unchanged
        np.testing.assert_array_equal(limited_wrench[3:6], large_force[3:6])
        
    def test_apply_limits_torque_exceeded(self):
        """Test _apply_limits when torque limit is exceeded."""
        # Create wrench with torque magnitude > max_torque
        large_torque = np.array([[10.0], [20.0], [30.0], [40.0], [30.0], [0.0]])  # |T| = 50
        self.controller.params.max_torque = 25.0
        
        limited_wrench = self.controller._apply_limits(large_torque)
        
        # Torque should be scaled down
        torque_part = limited_wrench[3:6]
        torque_magnitude = np.linalg.norm(torque_part)
        self.assertAlmostEqual(torque_magnitude, 25.0, places=5)
        
        # Direction should be preserved
        original_direction = large_torque[3:6] / np.linalg.norm(large_torque[3:6])
        limited_direction = torque_part / np.linalg.norm(torque_part)
        np.testing.assert_array_almost_equal(original_direction, limited_direction)
        
        # Force should be unchanged
        np.testing.assert_array_equal(limited_wrench[0:3], large_torque[0:3])
        
    def test_apply_limits_both_exceeded(self):
        """Test _apply_limits when both force and torque limits are exceeded."""
        # Create wrench with both force and torque magnitudes exceeding limits
        large_wrench = np.array([[80.0], [60.0], [0.0], [40.0], [30.0], [0.0]])
        self.controller.params.max_force = 50.0
        self.controller.params.max_torque = 25.0
        
        limited_wrench = self.controller._apply_limits(large_wrench)
        
        # Both force and torque should be limited
        force_magnitude = np.linalg.norm(limited_wrench[0:3])
        torque_magnitude = np.linalg.norm(limited_wrench[3:6])
        
        self.assertAlmostEqual(force_magnitude, 50.0, places=5)
        self.assertAlmostEqual(torque_magnitude, 25.0, places=5)
        
    def test_compute_angular_acceleration(self):
        """Test _compute_angular_acceleration method."""
        # Create a mock navigation state
        nav_state = Mock()
        
        result = self.controller._compute_angular_acceleration(nav_state)
        
        # Currently returns zeros as placeholder
        expected = np.zeros((3, 1))
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.shape, (3, 1))


class TestINDIControllerMessageHandling(unittest.TestCase):
    """Test message handling and conversion logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock ROS2 node initialization
        with patch('tauv_common.controller.Node.__init__'), \
             patch('tauv_common.controller.Node.create_subscription'), \
             patch('tauv_common.controller.Node.create_publisher'), \
             patch('tauv_common.controller.Node.create_timer'), \
             patch('tauv_common.controller.Node.get_logger') as mock_logger:
            
            mock_logger.return_value = Mock()
            self.controller = INDIController()
            
    def test_handle_accel_command(self):
        """Test handling of acceleration command messages."""
        # Create test acceleration command
        accel_msg = AccelStamped()
        accel_msg.header.stamp.sec = 123
        accel_msg.accel.linear.x = 1.0
        accel_msg.accel.linear.y = 2.0
        accel_msg.accel.linear.z = 3.0
        accel_msg.accel.angular.x = 0.1
        accel_msg.accel.angular.y = 0.2
        accel_msg.accel.angular.z = 0.3
        
        self.controller._handle_accel_command(accel_msg)
        
        # Check that message was stored
        self.assertEqual(self.controller._accel_command, accel_msg)
        
    @patch('tauv_common.controller.numpify')
    def test_handle_nav_state(self, mock_numpify):
        """Test handling of navigation state messages."""
        # Set up mock for numpify
        mock_linear_accel = np.array([[1.0], [2.0], [3.0]])
        mock_numpify.return_value = mock_linear_accel
        
        # Create test navigation state
        nav_state = Mock()
        nav_state.a_b = Mock()  # Mock acceleration
        
        # Call with no previous filtered acceleration
        self.controller._handle_nav_state(nav_state)
        
        # Check that navigation state was stored
        self.assertEqual(self.controller._nav_state, nav_state)
        
        # Check that filtered acceleration was initialized
        self.assertIsNotNone(self.controller._filtered_accel)
        
        # Call again to test filtering
        self.controller._handle_nav_state(nav_state)
        
    @patch('tauv_common.controller.numpify')
    def test_handle_nav_state_with_filtering(self, mock_numpify):
        """Test navigation state handling with existing filtered acceleration."""
        # Set up mock for numpify
        new_linear_accel = np.array([[2.0], [3.0], [4.0]])
        mock_numpify.return_value = new_linear_accel
        
        # Set up initial filtered acceleration
        initial_filtered = np.array([[1.0], [1.0], [1.0], [0.0], [0.0], [0.0]])
        self.controller._filtered_accel = initial_filtered.copy()
        
        # Create test navigation state
        nav_state = Mock()
        nav_state.a_b = Mock()
        
        self.controller._handle_nav_state(nav_state)
        
        # Check that filtering was applied
        # Expected: alpha * old + (1-alpha) * new
        alpha = self.controller.params.accel_filter_alpha
        expected_linear = alpha * initial_filtered[0:3] + (1-alpha) * new_linear_accel
        expected_angular = alpha * initial_filtered[3:6] + (1-alpha) * np.zeros((3, 1))
        
        np.testing.assert_array_almost_equal(
            self.controller._filtered_accel[0:3], expected_linear
        )
            
    @patch('tauv_common.controller.Node.get_clock')
    def test_publish_wrench(self, mock_get_clock):
        """Test wrench publishing logic."""
        # Mock clock
        mock_clock = Mock()
        mock_time = Mock()
        mock_time.to_msg.return_value = Mock()
        mock_clock.now.return_value = mock_time
        mock_get_clock.return_value = mock_clock
        
        # Mock publisher
        mock_publisher = Mock()
        self.controller._wrench_pub = mock_publisher
        
        # Test wrench
        test_wrench = np.array([[10.0], [20.0], [30.0], [1.0], [2.0], [3.0]])
        
        self.controller._publish_wrench(test_wrench)
        
        # Check that publish was called
        mock_publisher.publish.assert_called_once()
        
        # Check the published message
        published_msg = mock_publisher.publish.call_args[0][0]
        self.assertIsInstance(published_msg, WrenchStamped)
        self.assertEqual(published_msg.header.frame_id, "os/body")
        self.assertEqual(published_msg.wrench.force.x, 10.0)
        self.assertEqual(published_msg.wrench.force.y, 20.0)
        self.assertEqual(published_msg.wrench.force.z, 30.0)
        self.assertEqual(published_msg.wrench.torque.x, 1.0)
        self.assertEqual(published_msg.wrench.torque.y, 2.0)
        self.assertEqual(published_msg.wrench.torque.z, 3.0)


class TestINDIControlAlgorithm(unittest.TestCase):
    """Test the core INDI control algorithm."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock ROS2 node initialization
        with patch('tauv_common.controller.Node.__init__'), \
             patch('tauv_common.controller.Node.create_subscription'), \
             patch('tauv_common.controller.Node.create_publisher'), \
             patch('tauv_common.controller.Node.create_timer'), \
             patch('tauv_common.controller.Node.get_logger') as mock_logger:
            
            mock_logger.return_value = Mock()
            self.controller = INDIController()
            
        # Mock the get_logger method directly on the controller instance
        self.controller.get_logger = Mock(return_value=Mock())
        
        # Set up test scenario
        self.controller._filtered_accel = np.array([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
        self.controller._previous_wrench = np.array([[5.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
        
        # Create test acceleration command
        accel_cmd = AccelStamped()
        accel_cmd.accel.linear.x = 2.0  # Desired acceleration
        accel_cmd.accel.linear.y = 0.0
        accel_cmd.accel.linear.z = 0.0
        accel_cmd.accel.angular.x = 0.0
        accel_cmd.accel.angular.y = 0.0
        accel_cmd.accel.angular.z = 0.0
        self.controller._accel_command = accel_cmd
        
        # Create test navigation state
        nav_state = Mock()
        self.controller._nav_state = nav_state
        
    @patch('tauv_common.controller.numpify')
    @patch.object(INDIController, '_publish_wrench')
    def test_control_loop_indi_update(self, mock_publish, mock_numpify):
        """Test the core INDI control loop algorithm."""
        # Set up numpify mock to return the command values
        def numpify_side_effect(msg):
            if hasattr(msg, 'x'):  # Vector3-like
                return np.array([[msg.x], [msg.y], [msg.z]])
            return np.array([[0], [0], [0]])
        
        mock_numpify.side_effect = numpify_side_effect
        
        initial_wrench = self.controller._previous_wrench.copy()
        
        # Run control loop
        self.controller._control_loop()
        
        # Check that wrench was updated using INDI law
        # Expected: u_new = u_prev + G^(-1) * (a_desired - a_measured)
        desired_accel = np.array([[2.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
        measured_accel = self.controller._filtered_accel
        accel_error = desired_accel - measured_accel  # Should be [1.0, 0, 0, 0, 0, 0]
        
        expected_increment = self.controller._G_inv @ accel_error
        expected_wrench = initial_wrench + expected_increment
        
        # Apply limits to expected result for comparison
        expected_wrench = self.controller._apply_limits(expected_wrench)
        
        np.testing.assert_array_almost_equal(
            self.controller._previous_wrench, expected_wrench, decimal=5
        )
        
        # Check that publish was called
        mock_publish.assert_called_once()
        
    def test_control_loop_missing_inputs(self):
        """Test control loop behavior with missing inputs."""
        # Clear required inputs
        self.controller._accel_command = None
        
        with patch.object(self.controller, '_publish_wrench') as mock_publish:
            self.controller._control_loop()
            
            # Should not publish anything
            mock_publish.assert_not_called()
            
    def test_control_loop_initialization(self):
        """Test control loop with no previous wrench (initialization)."""
        self.controller._previous_wrench = None
        
        with patch('tauv_common.controller.numpify') as mock_numpify, \
             patch.object(self.controller, '_publish_wrench') as mock_publish:
            
            # Set up numpify mock
            mock_numpify.return_value = np.zeros((3, 1))
            
            self.controller._control_loop()
            
            # Should initialize previous wrench
            self.assertIsNotNone(self.controller._previous_wrench)
            self.assertEqual(self.controller._previous_wrench.shape, (6, 1))


if __name__ == '__main__':
    unittest.main() 