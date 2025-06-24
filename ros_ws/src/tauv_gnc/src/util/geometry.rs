use nalgebra as na;
use na::{Matrix3, Matrix6, Vector3, Isometry3};
use approx;

/// Extension trait to add SE(3) adjoint functionality to Isometry3
pub trait SE3Adjoint<T> {
    /// Compute the SE(3) adjoint matrix for this isometry
    ///
    /// The adjoint matrix Ad_g for SE(3) has the form:
    /// ```
    // Ad_g = [R    [t]×R]
    //        [0      R  ]
    /// ```
    /// where R is the 3×3 rotation matrix, t is the translation vector,
    /// and [t]× is the skew-symmetric matrix of t.
    fn adjoint_matrix(&self) -> Matrix6<T>;
    
    // TODO: Implement inverse adjoint
}

impl<T> SE3Adjoint<T> for Isometry3<T>
where
    T: na::RealField + Copy,
{
    fn adjoint_matrix(&self) -> Matrix6<T> {
        let rotation = self.rotation.to_rotation_matrix().matrix().to_owned();
        let translation = self.translation.vector;

        // Create the skew-symmetric matrix [t]×
        let t_skew = skew_symmetric_matrix(&translation);

        // Compute [t]×R (upper right block)
        let upper_right = t_skew * rotation;

        // Construct the 6×6 adjoint matrix
        let mut adj = Matrix6::<T>::zeros();

        // Upper left: R
        adj.fixed_view_mut::<3, 3>(0, 0).copy_from(&rotation);

        // Upper right: [t]×R
        adj.fixed_view_mut::<3, 3>(0, 3).copy_from(&upper_right);

        // Lower left: 0 (already zero)

        // Lower right: R
        adj.fixed_view_mut::<3, 3>(3, 3).copy_from(&rotation);

        adj
    }
}

/// Create a skew-symmetric matrix from a 3D vector
///
/// For vector v = [x, y, z], returns:
/// ```
// [v]× = [ 0  -z   y]
//        [ z   0  -x]
//        [-y   x   0]
/// ```
fn skew_symmetric_matrix<T>(v: &Vector3<T>) -> Matrix3<T>
where
    T: na::RealField + Copy,
{
    Matrix3::new(
        T::zero(), -v[2],     v[1],
        v[2],      T::zero(), -v[0],
        -v[1],     v[0],      T::zero()
    )
}

/// Apply the adjoint action to a se(3) element (6D twist vector)
///
/// Given an SE(3) element g and a se(3) element ξ (represented as a 6D vector),
/// computes Ad_g * ξ
pub fn adjoint_action<T>(isometry: &Isometry3<T>, twist: &na::Vector6<T>) -> na::Vector6<T>
where
    T: na::RealField + Copy,
{
    isometry.adjoint_matrix() * twist
}

#[cfg(test)]
mod tests {
    use super::*;
    use na::{Vector3, Vector6, UnitQuaternion, Translation3};
    use approx::assert_relative_eq;

    #[test]
    fn test_identity_adjoint() {
        let identity = Isometry3::<f64>::identity();
        let adj = identity.adjoint_matrix();
        let expected = Matrix6::<f64>::identity();

        assert_relative_eq!(adj, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_pure_translation_adjoint() {
        let translation = Translation3::new(1.0, 2.0, 3.0);
        let isometry = Isometry3::from_parts(translation, UnitQuaternion::identity());

        let adj = isometry.adjoint_matrix();

        // For pure translation, adjoint should be:
        // [I  [t]×]
        // [0    I ]
        let expected_upper_right = skew_symmetric_matrix(&Vector3::new(1.0, 2.0, 3.0));

        // Check structure
        assert_relative_eq!(
            adj.fixed_view::<3, 3>(0, 0).into_owned(),
            Matrix3::identity(),
            epsilon = 1e-10
        );
        assert_relative_eq!(
            adj.fixed_view::<3, 3>(0, 3).into_owned(),
            expected_upper_right,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            adj.fixed_view::<3, 3>(3, 0).into_owned(),
            Matrix3::zeros(),
            epsilon = 1e-10
        );
        assert_relative_eq!(
            adj.fixed_view::<3, 3>(3, 3).into_owned(),
            Matrix3::identity(),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_pure_rotation_adjoint() {
        let rotation = UnitQuaternion::from_axis_angle(
            &na::Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0)),
            std::f64::consts::PI / 4.0
        );
        let isometry = Isometry3::from_parts(Translation3::identity(), rotation);

        let adj = isometry.adjoint_matrix();
        let rot_matrix = rotation.to_rotation_matrix().matrix().to_owned();

        // For pure rotation, adjoint should be:
        // [R  0]
        // [0  R]
        assert_relative_eq!(
            adj.fixed_view::<3, 3>(0, 0).into_owned(),
            rot_matrix,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            adj.fixed_view::<3, 3>(0, 3).into_owned(),
            Matrix3::zeros(),
            epsilon = 1e-10
        );
        assert_relative_eq!(
            adj.fixed_view::<3, 3>(3, 0).into_owned(),
            &Matrix3::zeros(),
            epsilon = 1e-10
        );
        assert_relative_eq!(
            adj.fixed_view::<3, 3>(3, 3).into_owned(),
            rot_matrix,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_adjoint_action() {
        let translation = Translation3::new(0.1, 0.2, 0.3);
        let rotation = UnitQuaternion::from_axis_angle(
            &na::Unit::new_normalize(Vector3::new(0.0, 1.0, 0.0)),
            0.1
        );
        let isometry = Isometry3::from_parts(translation, rotation);

        let twist = Vector6::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0);
        let result = adjoint_action(&isometry, &twist);

        // The result should be a valid 6D vector
        assert_eq!(result.len(), 6);

        // Verify it matches direct matrix multiplication
        let adj = isometry.adjoint_matrix();
        let expected = adj * twist;
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_skew_symmetric_properties() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        let skew = skew_symmetric_matrix(&v);

        // Should be skew-symmetric: A^T = -A
        assert_relative_eq!(skew.transpose(), -skew, epsilon = 1e-10);

        // Diagonal should be zero
        assert_relative_eq!(skew[(0,0)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(skew[(1,1)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(skew[(2,2)], 0.0, epsilon = 1e-10);
    }
}