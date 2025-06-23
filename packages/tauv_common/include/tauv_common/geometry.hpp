#pragma once

#include <manif/SE3.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

namespace tauv::geometry {

    manif::SE3d tf_to_se3(const geometry_msgs::msg::Transform tf) {
        Eigen::Quaterniond q(
            tf.rotation.w,
            tf.rotation.x,
            tf.rotation.y,
            tf.rotation.z
        );

        Eigen::Vector3d t(
            tf.translation.x,
            tf.translation.y,
            tf.translation.z
        );

        return manif::SE3d(t, q);
    }

    auto se3_to_tf(const manif::SE3d& se3) -> geometry_msgs::msg::Transform
    {
        geometry_msgs::msg::Transform tf;

        const auto& t = se3.translation();
        const auto& q = se3.quat();

        tf.translation.x = t.x();
        tf.translation.y = t.y();
        tf.translation.z = t.z();

        tf.rotation.w = q.w();
        tf.rotation.x = q.x();
        tf.rotation.y = q.y();
        tf.rotation.z = q.z();

        return tf;
    }

    auto vector3_msg_to_eigen(const geometry_msgs::msg::Vector3& msg) -> Eigen::Vector3d {
        return Eigen::Vector3d(msg.x, msg.y, msg.z);
    }
    
    auto quaternion_msg_to_eigen(const geometry_msgs::msg::Quaternion& msg) -> Eigen::Quaterniond {
        return Eigen::Quaterniond(msg.w, msg.x, msg.y, msg.z);
    }

}
