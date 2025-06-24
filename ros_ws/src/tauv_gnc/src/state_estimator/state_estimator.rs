#![allow(non_snake_case)]

use std::ops::Deref;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::thread::sleep;
use std::time::Duration;
use rclrs::*;
use sensor_msgs::msg::Imu;
use nav_msgs::msg::Odometry;
use tauv_msgs::msg::Depth as DepthMsg;
use tauv_msgs::msg::WaterlinkedDvlFrame as DvlMsg;
// Add the transform listener module
use tauv_gnc::util::transform_listener::{TransformBuffer, TransformError, TransformListener};
use tauv_gnc::util::geometry::SE3Adjoint;

use nalgebra as na;
use nalgebra::{stack, MatrixView, SVector, VectorView, U3};
use rclrs::RclrsError::RclError;

// TODO move out of here and write a macro for this shit
type Vector3 = na::Vector3<f64>;
type Vector6 = na::Vector6<f64>;
type Matrix3 = na::Matrix3<f64>;
type Matrix6 = na::Matrix6<f64>;
type Matrix3x6 = na::Matrix3x6<f64>;
type Matrix6x3 = na::Matrix6x3<f64>;
type Rotation = na::Rotation3<f64>;
type Isometry = na::Isometry3<f64>;
type Quaternion = na::UnitQuaternion<f64>;

type DvlMeasurement = Vector3;
type DepthMeasurement = f64;

struct EkfControl {
    odom_R_body: Rotation,
    a_body_B: Vector3,
    omega_body_B: Vector3,
}

struct EkfState {
    data: Vector6
}

impl EkfState {
    pub fn new(r_body_O: &Vector3, v_body_O: &Vector3) -> Self {
        Self { data: Vector6::from_iterator(r_body_O.iter().chain(v_body_O.iter()).copied()) }
    }

    pub fn zeros() -> Self {
        Self { data: SVector::zeros() }
    }

    pub fn r_body_O(&self) -> MatrixView<'_, f64, na::U3, na::U1, na::U1, na::U6> {
        self.data.fixed_rows::<3>(0)
    }

    pub fn v_body_O(&self) -> MatrixView<'_, f64, na::U3, na::U1, na::U1, na::U6> {
        self.data.fixed_rows::<3>(3)
    }
}

impl Deref for EkfState {
    type Target = Vector6;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

struct EkfCov {
    data: Matrix6
}

impl EkfCov {
    pub fn uniform(sigma_r: f64, sigma_v: f64) -> Self {
        let (var_r, var_v) = (sigma_r.powi(2), sigma_v.powi(2));
        let diag = Vector6::new(var_r, var_r, var_r, var_v, var_v, var_v);
        Self {
            data: Matrix6::from_diagonal(&diag)
        }
    }

    pub fn new(data: Matrix6) -> Self {
        Self { data }
    }
}

impl Deref for EkfCov {
    type Target = Matrix6;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

struct EkfParams {
    initial_position_stddev_m: f64,
    initial_velocity_stddev_mps: f64,
    process_noise_density_pos: f64,
    process_noise_density_vel: f64,
    gravity: f64,
    history_length: usize,

    // Frames
    body_frame: Arc<str>,
    dvl_frame: Arc<str>,
    depth_frame: Arc<str>,
}

struct Ekf {
    r_body_depth_B: Vector3,
    body_T_dvl: Isometry,
    Ad_dvl_T_body: Matrix6,
    F_dvl: Matrix3x6,

    Qc: Matrix6,
    a_g_O: Vector3,
}

impl Ekf {
    pub fn new(params: &EkfParams, tfs: &EkfStaticTransforms) -> Self{

        let Ad_dvl_T_body = tfs.body_T_dvl.inverse().adjoint_matrix();
        let F_dvl = na::stack![Matrix3::zeros(), Ad_dvl_T_body.fixed_view::<3,3>(0,0)];
        // Process noise
        let var_r = params.process_noise_density_pos.powi(2);
        let var_v = params.process_noise_density_vel.powi(2);
        let Qc = Matrix6::from_diagonal(&Vector6::new(var_r, var_r, var_r, var_v, var_v, var_v));

        // This is the specific force as seen by the IMU in the inertial (free fall) frame,
        // so pointing up
        let a_g_O = Vector3::new(0.0, 0.0, params.gravity);

        Self {
            r_body_depth_B: tfs.r_body_depth_B,
            body_T_dvl: tfs.body_T_dvl,
            Ad_dvl_T_body,
            F_dvl,
            Qc,
            a_g_O,
        }
    }

    pub fn predict(&self, xkm1: &EkfState, uk: &EkfControl, dt: f64) -> EkfState {
        let a_body_O = uk.odom_R_body * uk.a_body_B;
        let r_body_km1_body_k_O = xkm1.v_body_O() * dt + 0.5 * a_body_O * dt.powi(2);
        let r_body_k_O = xkm1.r_body_O() + r_body_km1_body_k_O;

        let v_body_k_O = xkm1.v_body_O() + a_body_O * dt;

        EkfState::new(&r_body_k_O, &v_body_k_O)
    }

    pub fn predict_cov(&self, Pkm1: &EkfCov, dt: f64) -> EkfCov {
        let I3 = Matrix3::identity();
        let dr_dv = I3 * dt;
        let F = na::stack![ I3, dr_dv;
                            0, I3 ];
        let Pk = F * **Pkm1 * F.transpose();
        EkfCov::new(Pk)
    }

    pub fn h_dvl(&self, xk: &EkfState, uk: &EkfControl) -> DvlMeasurement {
        let v_body_B = uk.odom_R_body.inverse() * xk.v_body_O();
        let xi_body_B = na::stack![ v_body_B; uk.omega_body_B ];
        let v_dvl_V = (self.Ad_dvl_T_body * xi_body_B).fixed_rows::<3>(0).into_owned();
        v_dvl_V
    }

    pub fn h_depth(&self, xk: &EkfState, uk: &EkfControl) -> DepthMeasurement {
        let r_body_depth_O = uk.odom_R_body * self.r_body_depth_B;
        let r_odom_depth_O = xk.r_body_O() + r_body_depth_O;
        r_odom_depth_O[2]
    }
}

struct EkfStaticTransforms {
    r_body_depth_B: Vector3,
    body_T_dvl: Isometry,
}

fn lookup_static_transforms(tf_buffer: &TransformBuffer, params: &EkfParams) -> Result<EkfStaticTransforms, ()> {
    let tf_depth = tf_buffer.lookup_latest_transform(&params.body_frame, &params.depth_frame);
    let tf_dvl = tf_buffer.lookup_latest_transform(&params.body_frame, &params.dvl_frame);

    if let (Ok(tf_depth), Ok(tf_dvl)) = (tf_depth, tf_dvl) {
        let r_body_depth_B = tf_depth.translation();
        let body_T_dvl = tf_dvl.isometry;
        Ok(EkfStaticTransforms { r_body_depth_B, body_T_dvl })
    } else {
        Err(())
    }
}

fn initialize_ekf_node(executor: &Executor) -> Result<(Node, EkfParams), RclrsError>  {
    let node = executor.create_node("state_estimator_ekf")?;

    // Declare parameters with defaults
    let body_frame: Arc<str> = node
        .declare_parameter("body_frame")
        .default("body".into())
        .mandatory()?.get();

    let depth_frame: Arc<str> = node
        .declare_parameter("depth_frame")
        .default("depth".into())
        .mandatory()?.get();

    let dvl_frame: Arc<str> = node
        .declare_parameter("dvl_frame")
        .default("dvl".into())
        .mandatory()?.get();

    let initial_position_stddev_m: f64 = node
        .declare_parameter("initial_position_stddev_m")
        .default(0.01)
        .mandatory()?.get();

    let initial_velocity_stddev_mps: f64 = node
        .declare_parameter("initial_velocity_stddev_mps")
        .default(0.1)
        .mandatory()?.get();

    let process_noise_density_pos: f64 = node
        .declare_parameter("process_noise_density_pos_m_per_sqrt_s")
        .default(0.001)
        .mandatory()?.get();

    let process_noise_density_vel: f64 = node
        .declare_parameter("process_noise_density_vel_mps_per_sqrt_s")
        .default(0.001)
        .mandatory()?.get();

    let gravity: f64 = node
        .declare_parameter("g")
        .default(9.79596)
        .mandatory()?.get();

    let history_length: usize = node
        .declare_parameter("history_length")
        .default(20)
        .mandatory()?.get() as usize;

    let params = EkfParams {
        initial_position_stddev_m,
        initial_velocity_stddev_mps,
        process_noise_density_pos,
        process_noise_density_vel,
        gravity,
        history_length,
        body_frame,
        dvl_frame,
        depth_frame,
    };

    Ok((node, params))
}

fn wait_for_static_transforms(node: &Node, params: &EkfParams) -> Result<EkfStaticTransforms, RclrsError> {
    let tf_buffer = Arc::new(TransformBuffer::new());
    let listener_ = TransformListener::new(node, tf_buffer.clone())?;
    loop {
        sleep(Duration::from_millis(100));
        let result = lookup_static_transforms(&*tf_buffer, params);
        match result {
            Ok(transforms) => { return Ok(transforms); }
            Err(_) => {}
        }
    }
}

fn main() -> Result<(), RclrsError> {
    let context = Context::default_from_env()?;
    let mut executor = context.create_basic_executor();

    let (node, params) = initialize_ekf_node(&executor)?;
    let static_transforms = wait_for_static_transforms(&node, &params)?;
    let ekf = Ekf::new(&params, &static_transforms);

    println!("State estimator EKF node started");

    executor.spin(SpinOptions::default()).first_error()?;
    Ok(())
}
