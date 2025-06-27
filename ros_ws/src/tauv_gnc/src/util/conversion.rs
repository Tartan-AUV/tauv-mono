use chrono::{NaiveDateTime, Utc};
use nalgebra as na;
use crate::util::types::Matrix3;

#[derive(Debug)]
pub enum MessageConversionError {
    QuaternionNotNormalized
}

pub trait TryFromMsg<M> {
    fn try_from_msg(msg: M) -> Result<Self, MessageConversionError>
    where 
        Self: Sized;
}

pub trait FromMsg<M> {
    fn from_msg(msg: M) -> Self
    where
        Self: Sized;
}

impl TryFromMsg<&geometry_msgs::msg::Quaternion> for na::Rotation3<f64> {
    fn try_from_msg(msg: &geometry_msgs::msg::Quaternion) -> Result<Self, MessageConversionError> {
        let q = na::Quaternion::new(msg.w, msg.x, msg.y, msg.z);
        if (q.norm() - 1.0).abs() > 1e-10 {
            Err(MessageConversionError::QuaternionNotNormalized)
        } else {
            Ok(na::Rotation3::from(na::UnitQuaternion::new_normalize(q)))
        }
    }
}

impl FromMsg<geometry_msgs::msg::Vector3> for na::Vector3<f64> {
    fn from_msg(msg: geometry_msgs::msg::Vector3) -> Self {
        na::Vector3::new(msg.x, msg.y, msg.z)
    }
}

impl FromMsg<&builtin_interfaces::msg::Time> for chrono::DateTime<Utc> {
    fn from_msg(msg: &builtin_interfaces::msg::Time) -> Self {
        chrono::DateTime::from_timestamp(msg.sec as i64, msg.nanosec).unwrap()
    }
}

impl FromMsg<&[f64; 9]> for Matrix3 {
    fn from_msg(msg: &[f64; 9]) -> Matrix3 {
        Matrix3::new(msg[0], msg[1], msg[2],
                msg[3], msg[4], msg[5],
                msg[6], msg[7], msg[8])
    }
}
