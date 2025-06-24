use std::sync::{Arc, Mutex};
use rclrs::*;
use tauv_msgs::msg::{Depth, DepthSensorFrame};
use std_srvs::srv::{Trigger, Trigger_Request, Trigger_Response};

pub struct DepthEstimatorNode {
    node: Node,
    depth_pub: Publisher<Depth>,
    surface_pressure: Arc<Mutex<f64>>,
    reset_triggered: Arc<Mutex<bool>>,
    water_density: f64,
    gravity: f64,
    variance: f64,
}

impl DepthEstimatorNode {
    pub fn new(executor: &Executor) -> Result<Arc<Self>, RclrsError> {
        let node = executor.create_node("depth_estimator")?;

        // Get parameters with default values
        let surface_pressure_val: f64 = node
            .declare_parameter("surface_pressure")
            .default(101325.0)
            .mandatory()?.get();

        let water_density: f64 = node
            .declare_parameter("water_density")
            .default(997.0)
            .mandatory()?.get();

        let gravity: f64 = node
            .declare_parameter("gravity")
            .default(9.81)
            .mandatory()?.get();

        let variance: f64 = node
            .declare_parameter("variance")
            .default(1.0e-4)
            .mandatory()?.get();

        let surface_pressure = Arc::new(Mutex::new(surface_pressure_val));
        let reset_triggered = Arc::new(Mutex::new(false));

        let depth_pub = node.create_publisher::<Depth>("depth")?;

        // Create the Arc<Self> now
        let node_arc = Arc::new(DepthEstimatorNode {
            node,
            depth_pub,
            surface_pressure,
            reset_triggered,
            water_density,
            gravity,
            variance,
        });

        // Subscription callback
        let node_clone = Arc::clone(&node_arc);
        let depth_sub = node_arc.node.create_subscription::<DepthSensorFrame, _>(
            "depth_sensor_frame",
            move |msg: DepthSensorFrame| {
                node_clone.handle_depth_sensor_frame(msg);
            },
        )?;

        // Service callback
        let node_clone = Arc::clone(&node_arc);
        let service = node_arc.node.create_service::<Trigger, _>(
            "reset_depth",
            move |req: Trigger_Request, info: ServiceInfo| {
                node_clone.handle_reset_service(req, info)
            },
        )?;

        Ok(node_arc)
    }

    pub fn handle_depth_sensor_frame(&self, msg: DepthSensorFrame) {
        let mut reset_triggered = self.reset_triggered.lock().unwrap();

        if *reset_triggered {
            let mut surface_pressure = self.surface_pressure.lock().unwrap();
            *surface_pressure = msg.pressure as f64;
            *reset_triggered = false;
        }

        let surface_pressure = self.surface_pressure.lock().unwrap();

        let mut depth = Depth::default();
        depth.header = msg.header;
        depth.depth = (msg.pressure as f64 - *surface_pressure) / (self.water_density * self.gravity);
        depth.variance = self.variance;

        if let Err(e) = self.depth_pub.publish(depth) {
            eprintln!("Failed to publish depth: {}", e);
        }
    }

    pub fn handle_reset_service(
        &self,
        _request: Trigger_Request,
        _info: ServiceInfo,
    ) -> Trigger_Response {
        let mut reset_triggered = self.reset_triggered.lock().unwrap();
        *reset_triggered = true;

        Trigger_Response {
            success: true,
            message: "Reset triggered".to_string(),
        }
    }
}

fn main() -> Result<(), RclrsError> {
    let context = Context::default_from_env()?;
    let mut executor = context.create_basic_executor();

    let _node = DepthEstimatorNode::new(&executor)?;

    println!("Depth estimator node started");

    executor.spin(SpinOptions::default()).first_error()?;
    Ok(())
}
