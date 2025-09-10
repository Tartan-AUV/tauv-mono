#!/usr/bin/env python3

import unittest

import numpy as np
from geometry_msgs.msg import Quaternion, Transform, Vector3, Wrench
from spatialmath import SE3, UnitQuaternion
from tauv_common.util.geometry import numpify, numpify_cov_6x6, wrench_msg_to_numpy


class TestNumpify(unittest.TestCase):
    """Test cases for the numpify function."""

    def test_numpify_vector3(self):
        """Test Vector3 conversion to numpy array."""
        vector = Vector3()
        vector.x = 1.0
        vector.y = 2.0
        vector.z = 3.0

        result = numpify(vector)
        expected = np.array([[1.0], [2.0], [3.0]])

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3, 1))
        np.testing.assert_array_equal(result, expected)

    def test_numpify_vector3_negative_values(self):
        """Test Vector3 conversion with negative values."""
        vector = Vector3()
        vector.x = -1.5
        vector.y = 0.0
        vector.z = -2.7

        result = numpify(vector)
        expected = np.array([[-1.5], [0.0], [-2.7]])

        np.testing.assert_array_equal(result, expected)

    def test_numpify_quaternion(self):
        """Test Quaternion conversion to UnitQuaternion."""
        quat = Quaternion()
        quat.w = 1.0
        quat.x = 0.0
        quat.y = 0.0
        quat.z = 0.0

        result = numpify(quat)

        self.assertIsInstance(result, UnitQuaternion)
        # Test identity quaternion
        np.testing.assert_allclose(result.vec, [1.0, 0.0, 0.0, 0.0])

    def test_numpify_quaternion_non_identity(self):
        """Test Quaternion conversion with non-identity quaternion."""
        # 90 degree rotation around z-axis
        quat = Quaternion()
        quat.w = np.sqrt(2) / 2
        quat.x = 0.0
        quat.y = 0.0
        quat.z = np.sqrt(2) / 2

        result = numpify(quat)

        self.assertIsInstance(result, UnitQuaternion)
        # Verify it's normalized
        self.assertAlmostEqual(np.linalg.norm(result.vec), 1.0, places=6)

    def test_numpify_transform(self):
        """Test Transform conversion to SE3."""
        transform = Transform()

        # Set translation
        transform.translation.x = 1.0
        transform.translation.y = 2.0
        transform.translation.z = 3.0

        # Set rotation (identity)
        transform.rotation.w = 1.0
        transform.rotation.x = 0.0
        transform.rotation.y = 0.0
        transform.rotation.z = 0.0

        result = numpify(transform)

        self.assertIsInstance(result, SE3)

        # Check translation
        expected_translation = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(result.t, expected_translation)

        # Check rotation (should be identity)
        np.testing.assert_allclose(result.R, np.eye(3), atol=1e-10)

    def test_numpify_transform_with_rotation(self):
        """Test Transform conversion with non-identity rotation."""
        transform = Transform()

        # Set translation
        transform.translation.x = 0.0
        transform.translation.y = 0.0
        transform.translation.z = 0.0

        # 90 degree rotation around z-axis
        transform.rotation.w = np.sqrt(2) / 2
        transform.rotation.x = 0.0
        transform.rotation.y = 0.0
        transform.rotation.z = np.sqrt(2) / 2

        result = numpify(transform)

        self.assertIsInstance(result, SE3)

        # Check that rotation matrix corresponds to 90 degree rotation around z
        # Expected rotation matrix for 90 deg around z:
        expected_R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_allclose(result.R, expected_R, atol=1e-10)

    def test_numpify_wrench(self):
        """Test Wrench conversion to stacked numpy array."""
        wrench = Wrench()

        # Set force
        wrench.force.x = 1.0
        wrench.force.y = 2.0
        wrench.force.z = 3.0

        # Set torque
        wrench.torque.x = 4.0
        wrench.torque.y = 5.0
        wrench.torque.z = 6.0

        result = numpify(wrench)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (6, 1))

        expected = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        np.testing.assert_array_equal(result, expected)

    def test_numpify_wrench_zeros(self):
        """Test Wrench conversion with zero values."""
        wrench = Wrench()
        # All values are 0.0 by default

        result = numpify(wrench)

        expected = np.zeros((6, 1))
        np.testing.assert_array_equal(result, expected)

    def test_numpify_unsupported_type(self):
        """Test that unsupported types raise TypeError."""
        unsupported_obj = "not a geometry message"

        with self.assertRaises(TypeError) as context:
            numpify(unsupported_obj)

        self.assertIn("Unsupported type for numpify", str(context.exception))
        self.assertIn("str", str(context.exception))

    def test_numpify_none_type(self):
        """Test that None type raises TypeError."""
        with self.assertRaises(TypeError) as context:
            numpify(None)

        self.assertIn("Unsupported type for numpify", str(context.exception))


class TestNumpifyCov6x6(unittest.TestCase):
    """Test cases for the numpify_cov_6x6 function."""

    def test_numpify_cov_6x6_identity(self):
        """Test conversion of identity covariance matrix."""
        # 36-element list representing 6x6 identity matrix (row-major)
        identity_cov = [1.0 if i == j else 0.0 for i in range(6) for j in range(6)]

        result = numpify_cov_6x6(identity_cov)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (6, 6))
        self.assertEqual(result.dtype, np.float64)

        expected = np.eye(6, dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_numpify_cov_6x6_zeros(self):
        """Test conversion of zero covariance matrix."""
        zero_cov = [0.0] * 36

        result = numpify_cov_6x6(zero_cov)

        expected = np.zeros((6, 6), dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_numpify_cov_6x6_custom_values(self):
        """Test conversion with custom covariance values."""
        # Create a simple 6x6 matrix with known values for testing
        test_matrix = np.arange(36, dtype=np.float64).reshape(6, 6)
        cov_list = test_matrix.flatten().tolist()

        result = numpify_cov_6x6(cov_list)

        np.testing.assert_array_equal(result, test_matrix)

    def test_numpify_cov_6x6_mixed_values(self):
        """Test conversion with mixed positive/negative values."""
        # Test with realistic covariance-like values
        cov_list = [
            0.1,
            0.01,
            0.0,
            0.0,
            0.0,
            0.0,  # Row 0
            0.01,
            0.2,
            0.0,
            0.0,
            0.0,
            0.0,  # Row 1
            0.0,
            0.0,
            0.15,
            0.0,
            0.0,
            0.0,  # Row 2
            0.0,
            0.0,
            0.0,
            0.05,
            0.0,
            0.0,  # Row 3
            0.0,
            0.0,
            0.0,
            0.0,
            0.08,
            0.0,  # Row 4
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.12,  # Row 5
        ]

        result = numpify_cov_6x6(cov_list)

        expected = np.array(cov_list, dtype=np.float64).reshape(6, 6)
        np.testing.assert_array_equal(result, expected)

    def test_numpify_cov_6x6_integer_input(self):
        """Test conversion with integer input values."""
        cov_list = list(range(36))  # [0, 1, 2, ..., 35]

        result = numpify_cov_6x6(cov_list)

        self.assertEqual(result.dtype, np.float64)
        expected = np.arange(36, dtype=np.float64).reshape(6, 6)
        np.testing.assert_array_equal(result, expected)


class TestWrenchMsgToNumpy(unittest.TestCase):
    """Test cases for the wrench_msg_to_numpy function."""

    def test_wrench_msg_to_numpy_basic(self):
        """Test basic wrench message conversion."""
        wrench = Wrench()
        wrench.force.x = 10.0
        wrench.force.y = 20.0
        wrench.force.z = 30.0
        wrench.torque.x = 1.0
        wrench.torque.y = 2.0
        wrench.torque.z = 3.0

        result = wrench_msg_to_numpy(wrench)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (6, 1))

        expected = np.array([[10.0], [20.0], [30.0], [1.0], [2.0], [3.0]])
        np.testing.assert_array_equal(result, expected)

    def test_wrench_msg_to_numpy_consistency_with_numpify(self):
        """Test that wrench_msg_to_numpy returns same result as numpify."""
        wrench = Wrench()
        wrench.force.x = -5.5
        wrench.force.y = 10.2
        wrench.force.z = 0.0
        wrench.torque.x = 3.14
        wrench.torque.y = -2.71
        wrench.torque.z = 1.41

        result_wrench_func = wrench_msg_to_numpy(wrench)
        result_numpify = numpify(wrench)

        np.testing.assert_array_equal(result_wrench_func, result_numpify)

    def test_wrench_msg_to_numpy_zero_wrench(self):
        """Test conversion of zero wrench."""
        wrench = Wrench()  # All values default to 0.0

        result = wrench_msg_to_numpy(wrench)

        expected = np.zeros((6, 1))
        np.testing.assert_array_equal(result, expected)

    def test_wrench_msg_to_numpy_negative_values(self):
        """Test conversion with negative force and torque values."""
        wrench = Wrench()
        wrench.force.x = -1.0
        wrench.force.y = -2.0
        wrench.force.z = -3.0
        wrench.torque.x = -4.0
        wrench.torque.y = -5.0
        wrench.torque.z = -6.0

        result = wrench_msg_to_numpy(wrench)

        expected = np.array([[-1.0], [-2.0], [-3.0], [-4.0], [-5.0], [-6.0]])
        np.testing.assert_array_equal(result, expected)


class TestGeometryUtilsIntegration(unittest.TestCase):
    """Integration tests for geometry utility functions."""

    def test_transform_roundtrip_consistency(self):
        """Test that Transform -> SE3 maintains mathematical consistency."""
        transform = Transform()

        # Set known translation and rotation
        transform.translation.x = 5.0
        transform.translation.y = -3.0
        transform.translation.z = 2.0

        # 45 degree rotation around x-axis
        angle = np.pi / 4
        transform.rotation.w = np.cos(angle / 2)
        transform.rotation.x = np.sin(angle / 2)
        transform.rotation.y = 0.0
        transform.rotation.z = 0.0

        se3_result = numpify(transform)

        # Verify translation
        expected_translation = np.array([5.0, -3.0, 2.0])
        np.testing.assert_allclose(se3_result.t, expected_translation)

        # Verify rotation matrix properties
        R = se3_result.R
        # Should be orthogonal: R @ R.T = I
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        # Should have determinant 1
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=10)

    def test_vector_components_consistency(self):
        """Test that vector conversion maintains component order."""
        # Test multiple vectors to ensure consistent ordering
        test_cases = [(1.0, 2.0, 3.0), (-1.0, 0.0, 5.0), (0.0, -2.5, 0.0), (100.0, -50.0, 25.0)]

        for x, y, z in test_cases:
            with self.subTest(x=x, y=y, z=z):
                vector = Vector3()
                vector.x = x
                vector.y = y
                vector.z = z

                result = numpify(vector)

                # Check component order and values
                self.assertEqual(result[0, 0], x)
                self.assertEqual(result[1, 0], y)
                self.assertEqual(result[2, 0], z)


if __name__ == '__main__':
    unittest.main()
