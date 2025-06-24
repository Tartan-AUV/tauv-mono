use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use rclrs::*;
use builtin_interfaces::msg::Time;
use geometry_msgs::msg::TransformStamped;
use tf2_msgs::msg::TFMessage;
use nalgebra::{Vector3, UnitQuaternion, Isometry3};

#[derive(Debug, Clone)]
pub struct Transform {
    pub isometry: Isometry3<f64>,
    pub timestamp: Time,
}

impl From<TransformStamped> for Transform {
    fn from(tf: TransformStamped) -> Self {
        let translation = Vector3::new(
            tf.transform.translation.x,
            tf.transform.translation.y,
            tf.transform.translation.z,
        );
        
        let rotation = UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
            tf.transform.rotation.w,
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
        ));

        let isometry = Isometry3::from_parts(translation.into(), rotation);

        Transform {
            isometry,
            timestamp: tf.header.stamp,
        }
    }
}

impl Transform {
    pub fn translation(&self) -> Vector3<f64> {
        self.isometry.translation.vector
    }
    
    pub fn rotation(&self) -> UnitQuaternion<f64> {
        self.isometry.rotation
    }
    
    pub fn inverse(&self) -> Transform {
        Transform {
            isometry: self.isometry.inverse(),
            timestamp: self.timestamp.clone(),
        }
    }
}

#[derive(Debug)]
pub enum TransformError {
    FrameNotFound(String),
    NoTransformAtTime,
    InvalidFrameId,
    ExtrapolationError,
}

impl std::fmt::Display for TransformError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            TransformError::FrameNotFound(frame) => write!(f, "Frame '{}' not found in transform tree", frame),
            TransformError::NoTransformAtTime => write!(f, "No transform available at requested time"),
            TransformError::InvalidFrameId => write!(f, "Invalid frame ID provided"),
            TransformError::ExtrapolationError => write!(f, "Transform extrapolation not allowed"),
        }
    }
}

impl std::error::Error for TransformError {}

type TransformKey = (String, String); // (child_frame, parent_frame)
type TransformHistory = Vec<Transform>;

pub struct TransformBuffer {
    // Map from (child_frame, parent_frame) to list of transforms
    transforms: Arc<Mutex<HashMap<TransformKey, TransformHistory>>>,
    // Static transforms (no time history)
    static_transforms: Arc<Mutex<HashMap<TransformKey, Transform>>>,
    max_history_length: usize,
    max_age_seconds: f64,
}

impl TransformBuffer {
    pub fn new() -> Self {
        Self {
            transforms: Arc::new(Mutex::new(HashMap::new())),
            static_transforms: Arc::new(Mutex::new(HashMap::new())),
            max_history_length: 100,
            max_age_seconds: 10.0,
        }
    }

    pub fn with_capacity(max_history_length: usize, max_age_seconds: f64) -> Self {
        Self {
            transforms: Arc::new(Mutex::new(HashMap::new())),
            static_transforms: Arc::new(Mutex::new(HashMap::new())),
            max_history_length,
            max_age_seconds,
        }
    }

    pub fn add_transform(&self, tf: TransformStamped, is_static: bool) {
        let transform = Transform::from(tf.clone());
        let key = (tf.child_frame_id.clone(), tf.header.frame_id.clone());

        if is_static {
            let mut static_transforms = self.static_transforms.lock().unwrap();
            static_transforms.insert(key, transform);
        } else {
            let mut transforms = self.transforms.lock().unwrap();
            let history = transforms.entry(key).or_insert_with(Vec::new);
            
            // Insert in chronological order
            // TODO: Implement proper insertion sort based on timestamp
            history.push(transform);
            
            // Maintain history length
            if history.len() > self.max_history_length {
                history.remove(0);
            }
            
            // TODO: Remove old transforms based on max_age_seconds
        }
    }

    pub fn lookup_transform(
        &self,
        target_frame: &str,
        source_frame: &str,
        time: Option<Time>,
    ) -> Result<Transform, TransformError> {
        // Handle identity transform
        if target_frame == source_frame {
            return Ok(Transform {
                isometry: Isometry3::identity(),
                timestamp: time.unwrap_or_default(),
            });
        }

        // First check static transforms
        let static_key = (source_frame.to_string(), target_frame.to_string());
        if let Some(transform) = self.static_transforms.lock().unwrap().get(&static_key) {
            return Ok(transform.clone());
        }

        // Check reverse static transform
        let reverse_static_key = (target_frame.to_string(), source_frame.to_string());
        if let Some(transform) = self.static_transforms.lock().unwrap().get(&reverse_static_key) {
            return Ok(transform.inverse());
        }

        // Check dynamic transforms
        let key = (source_frame.to_string(), target_frame.to_string());
        let transforms = self.transforms.lock().unwrap();
        
        if let Some(history) = transforms.get(&key) {
            if history.is_empty() {
                return Err(TransformError::NoTransformAtTime);
            }

            // If no specific time requested, return latest
            if time.is_none() {
                return Ok(history.last().unwrap().clone());
            }

            // TODO: Implement time-based lookup with interpolation
            // For now, return the latest transform
            return Ok(history.last().unwrap().clone());
        }

        // Check reverse transform
        let reverse_key = (target_frame.to_string(), source_frame.to_string());
        if let Some(history) = transforms.get(&reverse_key) {
            if history.is_empty() {
                return Err(TransformError::NoTransformAtTime);
            }

            return Ok(history.last().unwrap().inverse());
        }

        // TODO: Implement multi-hop transform chain resolution
        Err(TransformError::FrameNotFound(format!("{} -> {}", source_frame, target_frame)))
    }

    pub fn can_transform(
        &self,
        target_frame: &str,
        source_frame: &str,
        time: Option<Time>,
    ) -> bool {
        self.lookup_transform(target_frame, source_frame, time).is_ok()
    }

    pub fn get_frame_list(&self) -> Vec<String> {
        let mut frames = std::collections::HashSet::new();
        
        let transforms = self.transforms.lock().unwrap();
        for (child, parent) in transforms.keys() {
            frames.insert(child.clone());
            frames.insert(parent.clone());
        }
        
        let static_transforms = self.static_transforms.lock().unwrap();
        for (child, parent) in static_transforms.keys() {
            frames.insert(child.clone());
            frames.insert(parent.clone());
        }
        
        frames.into_iter().collect()
    }

    // Helper method to clear old transforms
    pub fn prune_old_transforms(&self, current_time: Time) {
        // TODO: Implement pruning based on max_age_seconds
    }
}

pub struct TransformListener {
    _tf_sub: Option<Subscription<TFMessage>>,
    _tf_static_sub: Option<Subscription<TFMessage>>,
    buffer: Arc<TransformBuffer>,
}

impl TransformListener {
    pub fn new(node: &Node, buffer: Arc<TransformBuffer>) -> Result<Self, RclrsError> {
        let buffer_clone = Arc::clone(&buffer);
        let tf_sub = node.create_subscription::<TFMessage, _>(
            "/tf",
            move |msg: TFMessage| {
                for tf in msg.transforms {
                    buffer_clone.add_transform(tf, false);
                }
            },
        )?;

        let buffer_clone = Arc::clone(&buffer);
        let tf_static_sub = node.create_subscription::<TFMessage, _>(
            "/tf_static",
            move |msg: TFMessage| {
                for tf in msg.transforms {
                    buffer_clone.add_transform(tf, true);
                }
            },
        )?;

        Ok(TransformListener {
            _tf_sub: Some(tf_sub),
            _tf_static_sub: Some(tf_static_sub),
            buffer,
        })
    }

    pub fn get_buffer(&self) -> Arc<TransformBuffer> {
        Arc::clone(&self.buffer)
    }
}

// Convenience functions for common use cases
impl TransformBuffer {
    pub fn lookup_latest_transform(
        &self,
        target_frame: &str,
        source_frame: &str,
    ) -> Result<Transform, TransformError> {
        self.lookup_transform(target_frame, source_frame, None)
    }

    pub fn lookup_transform_at_time(
        &self,
        target_frame: &str,
        source_frame: &str,
        time: Time,
    ) -> Result<Transform, TransformError> {
        self.lookup_transform(target_frame, source_frame, Some(time))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_transform() {
        let buffer = TransformBuffer::new();
        let result = buffer.lookup_transform("base_link", "base_link", None);
        assert!(result.is_ok());
        
        let transform = result.unwrap();
        assert_eq!(transform.translation(), Vector3::zeros());
        assert_eq!(transform.rotation(), UnitQuaternion::identity());
    }

    #[test]
    fn test_frame_not_found() {
        let buffer = TransformBuffer::new();
        let result = buffer.lookup_transform("frame1", "frame2", None);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            TransformError::FrameNotFound(_) => {},
            _ => panic!("Expected FrameNotFound error"),
        }
    }
} 