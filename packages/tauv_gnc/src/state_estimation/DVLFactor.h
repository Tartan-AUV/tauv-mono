//
// Created by gleb on 5/20/25.
//

#ifndef DVLFACTOR_H
#define DVLFACTOR_H

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/base/Matrix.h>

using namespace gtsam;

/**
 * DVLFactor: measures body-frame velocity using DVL
 *
 * v_b_meas ≈ R_bw * v_w
 * where R_bw = T_wb.rotation().transpose()
 */
class DVLFactor : public NoiseModelFactor2<Vector3, Pose3> {
private:
  Vector3 measuredVelocityB_;  // DVL measurement in body frame

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DVLFactor(Key velKey, Key poseKey, const Vector3& measuredVelocityB,
            const SharedNoiseModel& model)
      : NoiseModelFactor2<Vector3, Pose3>(model, velKey, poseKey),
        measuredVelocityB_(measuredVelocityB) {}

  Vector evaluateError(const Vector3& v_world, const Pose3& T_wb,
                       boost::optional<Matrix&> H_v = boost::none,
                       boost::optional<Matrix&> H_pose = boost::none) const override {
    const Rot3& R_wb = T_wb.rotation();
    Rot3 R_bw = R_wb.inverse();

    Vector3 v_body = R_bw * v_world;

    if (H_v || H_pose) {
      // Derivative w.r.t velocity
      if (H_v) {
        *H_v = R_bw.matrix();
      }

      // Derivative w.r.t pose
      if (H_pose) {
        // ∂(R_bw * v) / ∂(R_wb)
        // R_bw = R_wb^T, so ∂(R_bw*v) = -skew(R_bw*v) * ∂θ
        Matrix36 dRv_dpose;
        dRv_dpose.setZero();
        Vector3 v_body_local = v_body;
        dRv_dpose.block<1, 3>(0, 0) = -v_body_local.transpose() * skewSymmetric(Vector3::UnitX());
        dRv_dpose.block<1, 3>(1, 0) = -v_body_local.transpose() * skewSymmetric(Vector3::UnitY());
        dRv_dpose.block<1, 3>(2, 0) = -v_body_local.transpose() * skewSymmetric(Vector3::UnitZ());

        *H_pose = dRv_dpose;
      }
    }

    return v_body - measuredVelocityB_;
  }

  void print(const std::string& s,
             const KeyFormatter& keyFormatter = DefaultKeyFormatter) const override {
    std::cout << s << "DVLFactor on " << keyFormatter(this->key1())
              << " (velocity), " << keyFormatter(this->key2()) << " (pose)\n";
    std::cout << "  measured velocity (body): " << measuredVelocityB_.transpose() << std::endl;
    noiseModel_->print("  noise model: ");
  }

  bool equals(const NonlinearFactor& expected, double tol = 1e-9) const override {
    const DVLFactor* e = dynamic_cast<const DVLFactor*>(&expected);
    return e && (measuredVelocityB_ - e->measuredVelocityB_).norm() < tol &&
           NoiseModelFactor2<Vector3, Pose3>::equals(expected, tol);
  }

  NonlinearFactor::shared_ptr clone() const override {
    return boost::static_pointer_cast<NonlinearFactor>(
      NonlinearFactor::shared_ptr(new DVLFactor(*this)));
  }
};


#endif //DVLFACTOR_H
