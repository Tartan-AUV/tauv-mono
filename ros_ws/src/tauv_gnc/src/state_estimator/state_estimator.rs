#![allow(non_snake_case)]

use std::collections::BTreeMap;
use std::ops::Deref;
use std::sync::{Arc, Mutex, mpsc};
use std::sync::mpsc::Receiver;
use std::thread;
use std::thread::sleep;
use std::time::Duration;
use chrono::{DateTime, TimeDelta, Utc};
use rclrs::*;
use sensor_msgs::msg::Imu;
use nav_msgs::msg::Odometry;
use tauv_msgs::msg::{Depth as DepthMsg, Depth};
use tauv_msgs::msg::WaterlinkedDvlFrame as DvlMsg;
use tauv_gnc::util::transform_listener::{TransformBuffer, TransformError, TransformListener};
use tauv_gnc::util::conversion::*;
use tauv_gnc::util::geometry::*;
use tauv_gnc::util::types::*;

use nalgebra as na;
use nalgebra::{stack, MatrixView, Rotation3, SVector, VectorView, U3};
use crate::EkfInput::Dvl;

struct EkfControl {
    odom_R_body: Rotation,
    a_body_B: Vector3,
    omega_body_B: Vector3,
}

impl EkfControl {
    fn try_from_msg(msg: &Imu) -> Result<Self, MessageConversionError> {
        Ok(Self {
            odom_R_body: Rotation::try_from_msg(&msg.orientation)?,
            a_body_B: Vector3::from_msg(msg.linear_acceleration.clone()),
            omega_body_B: Vector3::from_msg(msg.angular_velocity.clone()),
        })
    }
}

#[derive(Clone, Debug)]
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

use Matrix6 as EkfCov;

impl Deref for EkfState {
    type Target = Vector6;
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

struct EkfStaticTransforms {
    r_body_depth_B: Vector3,
    body_T_dvl: Isometry,
}

struct DvlInput {
    v_dvl_V: Vector3,
    R: Matrix3
}

struct DepthInput {
    z: f64,
    R: f64,
}

enum EkfInput {
    Imu(DateTime<Utc>, EkfControl),
    Dvl(DateTime<Utc>, DvlInput),
    Depth(DateTime<Utc>, DepthInput),
}

enum EkfObservation {
    Dvl(DateTime<Utc>, DvlInput),
    Depth(DateTime<Utc>, DepthInput),
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
        let Pk = F * *Pkm1 * F.transpose();
        Pk
    }

    pub fn update<const D: usize>(
        &self,
        xk_hat: &EkfState,
        Pk_hat: &EkfCov,
        zk: &na::SVector<f64, D>,
        Rk: &na::SMatrix<f64, D, D>,
        zk_hat: &na::SVector<f64, D>,
        H: &na::SMatrix<f64, D, 6>,
    ) -> Result<(EkfState, EkfCov), String>
    {
        let yk_hat = zk - zk_hat;
        let Sk = H * Pk_hat * H.transpose() + Rk;
        
        let Sk_inv = Sk.try_inverse();
        if Sk_inv.is_none() { return Err("Sk singular".to_string()); }
        let Sk_inv = Sk_inv.unwrap();
        
        let Kk = Pk_hat * H.transpose() * Sk_inv;
        let xk = xk_hat.data + Kk * yk_hat;
        let Pk = (Matrix6::identity() - Kk * H) * Pk_hat;
        Ok((EkfState { data: xk }, Pk))
    }

    pub fn h_dvl(&self, xk: &EkfState, uk: &EkfControl) -> Vector3 {
        let v_body_B = uk.odom_R_body.inverse() * xk.v_body_O();
        let xi_body_B = na::stack![ v_body_B; uk.omega_body_B ];
        let v_dvl_V = (self.Ad_dvl_T_body * xi_body_B).fixed_rows::<3>(0).into_owned();
        v_dvl_V
    }

    pub fn h_depth(&self, xk: &EkfState, uk: &EkfControl) -> f64 {
        let r_body_depth_O = uk.odom_R_body * self.r_body_depth_B;
        let r_odom_depth_O = xk.r_body_O() + r_body_depth_O;
        r_odom_depth_O[2]
    }
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

enum Event {
    Imu(EkfControl),
    Dvl(DvlInput, EkfState, EkfCov),
    Depth(DepthInput, EkfState, EkfCov),
}


struct EkfHistory {
    map: BTreeMap<DateTime<Utc>, Event>,
    last_dvl_t: DateTime<Utc>,
    last_depth_t: DateTime<Utc>,
    last_imu_t: DateTime<Utc>,
}

impl EkfHistory {
    pub fn try_new(
        t_depth: DateTime::<Utc>,
        t_dvl: DateTime::<Utc>,
        t_imu: DateTime::<Utc>,
        depth: DepthInput,
        dvl: DvlInput,
        imu: EkfControl,
        params: &EkfParams,
    ) -> Result<Self, String> {

        let max_dt= TimeDelta::from_std(Duration::from_millis(200)).unwrap();

        if max_dt < (t_depth - t_dvl).abs() || max_dt < (t_depth - t_imu).abs() ||
            max_dt < (t_imu - t_dvl).abs() {
            return Err("shit".to_string())
        }

        // TODO: use IMU and DVL inputs for state init
        let state = EkfState::new(&Vector3::new(0.0, 0.0, depth.z), &Vector3::zeros());
        let var_r = params.initial_position_stddev_m.powi(2);
        let var_v = params.initial_velocity_stddev_mps.powi(2);
        let cov = Matrix6::from_diagonal(&na::vector![var_r, var_r, depth.R, var_v, var_v, var_v]);

        let mut map = BTreeMap::new();

        map.insert(t_depth, Event::Depth(depth, state.clone(), cov));
        map.insert(t_dvl, Event::Dvl(dvl, state.clone(), cov));
        map.insert(t_imu, Event::Imu(imu));

        Ok(Self {
            map,
            last_depth_t: t_depth,
            last_dvl_t: t_dvl,
            last_imu_t: t_imu,
        })
    
    }
}

fn run_ekf(rx: Receiver<EkfInput>, node: Arc<Node>, ekf: Ekf, params: EkfParams) -> Result<(), RclrsError> {
    println!("EKF processing thread started, waiting for first depth measurement...");
    
    // Prior
    let mut depth_input_stamped = None;
    let mut dvl_input_stamped = None;
    let mut imu_input_stamped = None;

    // We wait for all inputs to come in. This has the added benefit that we won't start the EKF
    // until the DVL begins streaming.
    while depth_input_stamped.is_none() || imu_input_stamped.is_none() ||
        dvl_input_stamped.is_none() {
        match rx.recv() {
            Ok(measurement) => match measurement {
                EkfInput::Depth(t, depth_input) => {
                    depth_input_stamped = Some((t, depth_input))
                }
                EkfInput::Imu(t, imu_input) => {
                    imu_input_stamped = Some((t, imu_input));
                }
                EkfInput::Dvl(t, dvl_input) => {
                    dvl_input_stamped = Some((t, dvl_input));
                }
            }
            Err(_) => {
                println!("Channel closed while waiting for initial measurements");
                return Ok(());
            }
        }
    }

    let (t_depth, depth_input) = depth_input_stamped.unwrap();
    let (t_dvl, dvl_input) = dvl_input_stamped.unwrap();
    let (t_imu, imu_input) = imu_input_stamped.unwrap();
    let mut history = EkfHistory::try_new(
        t_depth,
        t_dvl,
        t_imu,
        depth_input,
        dvl_input,
        imu_input,
        &params,
    );
    
    // Create odometry publisher
    let odom_pub = node.create_publisher::<Odometry>("odom")?;

    Ok(())
}

fn main() -> Result<(), RclrsError> {
    let context = Context::default_from_env()?;
    let mut executor = context.create_basic_executor();

    // Init node from parameters and wait for transforms
    let (node, params) = initialize_ekf_node(&executor)?;
    let static_transforms = wait_for_static_transforms(&node, &params)?;
    let ekf = Ekf::new(&params, &static_transforms);

    // Subscriptions
    let (tx, rx) = mpsc::channel::<EkfInput>();
    let tx_clone = tx.clone();
    node.create_subscription("imu", move |msg: Imu| {
        let _ = tx_clone.send(EkfInput::Imu(
            DateTime::<Utc>::from_msg(&msg.header.stamp),
            EkfControl::try_from_msg(&msg).unwrap()
        ));
    })?;
    let tx_clone = tx.clone();
    node.create_subscription("depth", move |msg: Depth| {
        let _ = tx_clone.send(EkfInput::Depth (
            DateTime::<Utc>::from_msg(&msg.header.stamp),
            DepthInput { z: msg.depth, R: msg.variance }
        ));
    })?;
    let tx_clone = tx.clone();
    node.create_subscription("dvl", move |msg: DvlMsg| {
        let _ = tx_clone.send(EkfInput::Dvl (
            DateTime::<Utc>::from_msg(&msg.header.stamp),
            DvlInput { v_dvl_V: Vector3::new(msg.vx, msg.vy, msg.vz), R: Matrix3::from_msg(&msg.covariance)  }
        ));
    })?;

    // Spawn processing thread
    let node_arc = Arc::new(node);
    let processing_thread = thread::spawn(move || {
        run_ekf(rx, node_arc, ekf, params).unwrap();
    });

    println!("State estimator EKF node started");

    executor.spin(SpinOptions::default()).first_error()?;
    
    // Wait for processing thread to finish
    let _ = processing_thread.join();
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra as na;
    use chrono::{DateTime, Utc, TimeZone};
    use approx::assert_relative_eq;

    // Helper function to create test parameters
    fn create_test_params() -> EkfParams {
        EkfParams {
            initial_position_stddev_m: 0.01,
            initial_velocity_stddev_mps: 0.1,
            process_noise_density_pos: 0.001,
            process_noise_density_vel: 0.001,
            gravity: 9.81,
            history_length: 20,
            body_frame: "body".into(),
            dvl_frame: "dvl".into(),
            depth_frame: "depth".into(),
        }
    }

    // Helper function to create test static transforms
    fn create_test_transforms() -> EkfStaticTransforms {
        // DVL is 0.5m below and 0.2m forward of body center
        let body_T_dvl = Isometry::new(
            Vector3::new(0.2, 0.0, 0.5),
            Vector3::zeros() // No rotation for simplicity
        );
        
        // Depth sensor is 0.1m above body center
        let r_body_depth_B = Vector3::new(0.0, 0.0, -0.1);
        
        EkfStaticTransforms {
            r_body_depth_B,
            body_T_dvl,
        }
    }

    #[test]
    fn test_ekf_params_initialization() {
        let params = create_test_params();
        
        assert_eq!(params.initial_position_stddev_m, 0.01);
        assert_eq!(params.initial_velocity_stddev_mps, 0.1);
        assert_eq!(params.process_noise_density_pos, 0.001);
        assert_eq!(params.process_noise_density_vel, 0.001);
        assert_eq!(params.gravity, 9.81);
        assert_eq!(params.history_length, 20);
        assert_eq!(&*params.body_frame, "body");
        assert_eq!(&*params.dvl_frame, "dvl");
        assert_eq!(&*params.depth_frame, "depth");
    }

    #[test]
    fn test_ekf_initialization() {
        let params = create_test_params();
        let transforms = create_test_transforms();
        let ekf = Ekf::new(&params, &transforms);
        
        // Check that gravity vector is correct
        assert_relative_eq!(ekf.a_g_O, Vector3::new(0.0, 0.0, params.gravity));
        
        // Check process noise covariance matrix
        let expected_qc_diag = Vector6::new(
            params.process_noise_density_pos.powi(2),
            params.process_noise_density_pos.powi(2),
            params.process_noise_density_pos.powi(2),
            params.process_noise_density_vel.powi(2),
            params.process_noise_density_vel.powi(2),
            params.process_noise_density_vel.powi(2),
        );
        assert_relative_eq!(ekf.Qc.diagonal(), expected_qc_diag);
        
        // Check that transforms are stored correctly
        assert_relative_eq!(ekf.r_body_depth_B, transforms.r_body_depth_B);
        assert_relative_eq!(ekf.body_T_dvl, transforms.body_T_dvl);
    }

    #[test]
    fn test_ekf_predict_stationary() {
        let params = create_test_params();
        let transforms = create_test_transforms();
        let ekf = Ekf::new(&params, &transforms);
        
        // Initial state: at origin, stationary
        let x0 = EkfState::new(&Vector3::zeros(), &Vector3::zeros());
        
        // Control input: no acceleration or rotation, body aligned with odom
        let u = EkfControl {
            odom_R_body: Rotation::identity(),
            a_body_B: Vector3::zeros(),
            omega_body_B: Vector3::zeros(),
        };
        
        let dt = 0.1;
        let x1 = ekf.predict(&x0, &u, dt);
        
        // Should remain at origin with zero velocity
        assert_relative_eq!(x1.r_body_O().into_owned(), Vector3::zeros(), epsilon = 1e-10);
        assert_relative_eq!(x1.v_body_O().into_owned(), Vector3::zeros(), epsilon = 1e-10);
    }

    #[test]
    fn test_ekf_predict_constant_velocity() {
        let params = create_test_params();
        let transforms = create_test_transforms();
        let ekf = Ekf::new(&params, &transforms);
        
        // Initial state: at origin, moving at 1 m/s in x direction
        let v0 = Vector3::new(1.0, 0.0, 0.0);
        let x0 = EkfState::new(&Vector3::zeros(), &v0);
        
        // Control input: no acceleration
        let u = EkfControl {
            odom_R_body: Rotation::identity(),
            a_body_B: Vector3::zeros(),
            omega_body_B: Vector3::zeros(),
        };
        
        let dt = 0.5;
        let x1 = ekf.predict(&x0, &u, dt);
        
        // Position should be velocity * time
        assert_relative_eq!(x1.r_body_O().into_owned(), Vector3::new(0.5, 0.0, 0.0), epsilon = 1e-10);
        // Velocity should remain constant
        assert_relative_eq!(x1.v_body_O().into_owned(), v0, epsilon = 1e-10);
    }

    #[test]
    fn test_ekf_predict_with_acceleration() {
        let params = create_test_params();
        let transforms = create_test_transforms();
        let ekf = Ekf::new(&params, &transforms);
        
        // Initial state: at origin, stationary
        let x0 = EkfState::new(&Vector3::zeros(), &Vector3::zeros());
        
        // Control input: 2 m/s² acceleration in body x direction
        let u = EkfControl {
            odom_R_body: Rotation::identity(),
            a_body_B: Vector3::new(2.0, 0.0, 0.0),
            omega_body_B: Vector3::zeros(),
        };
        
        let dt = 1.0;
        let x1 = ekf.predict(&x0, &u, dt);
        
        // Position: 0.5 * a * t²
        assert_relative_eq!(x1.r_body_O().into_owned(), Vector3::new(1.0, 0.0, 0.0), epsilon = 1e-10);
        // Velocity: a * t
        assert_relative_eq!(x1.v_body_O().into_owned(), Vector3::new(2.0, 0.0, 0.0), epsilon = 1e-10);
    }

    #[test]
    fn test_h_dvl_stationary() {
        let params = create_test_params();
        let transforms = create_test_transforms();
        let ekf = Ekf::new(&params, &transforms);
        
        // State: stationary at origin
        let x = EkfState::new(&Vector3::zeros(), &Vector3::zeros());
        
        // Control: no rotation
        let u = EkfControl {
            odom_R_body: Rotation::identity(),
            a_body_B: Vector3::zeros(),
            omega_body_B: Vector3::zeros(),
        };
        
        let v_dvl = ekf.h_dvl(&x, &u);
        
        // DVL should measure zero velocity
        assert_relative_eq!(v_dvl, Vector3::zeros(), epsilon = 1e-10);
    }

    #[test]
    fn test_h_dvl_linear_motion() {
        let params = create_test_params();
        let transforms = create_test_transforms();
        let ekf = Ekf::new(&params, &transforms);
        
        // State: moving at 1 m/s in odom x direction
        let x = EkfState::new(&Vector3::zeros(), &Vector3::new(1.0, 0.0, 0.0));
        
        // Control: no rotation
        let u = EkfControl {
            odom_R_body: Rotation::identity(),
            a_body_B: Vector3::zeros(),
            omega_body_B: Vector3::zeros(),
        };
        
        let v_dvl = ekf.h_dvl(&x, &u);
        
        // Since DVL is aligned with body (no rotation in transform), 
        // it should measure the same velocity
        assert_relative_eq!(v_dvl, Vector3::new(1.0, 0.0, 0.0), epsilon = 1e-10);
    }

    #[test]
    fn test_h_dvl_with_angular_velocity() {
        let params = create_test_params();
        // Create transforms with DVL offset from body center
        let body_T_dvl = Isometry::new(
            Vector3::new(1.0, 0.0, 0.0), // DVL is 1m forward of body center
            Vector3::zeros()
        );
        let transforms = EkfStaticTransforms {
            r_body_depth_B: Vector3::zeros(),
            body_T_dvl,
        };
        let ekf = Ekf::new(&params, &transforms);
        
        // State: stationary
        let x = EkfState::new(&Vector3::zeros(), &Vector3::zeros());
        
        // Control: rotating at 1 rad/s about z axis
        let u = EkfControl {
            odom_R_body: Rotation::identity(),
            a_body_B: Vector3::zeros(),
            omega_body_B: Vector3::new(0.0, 0.0, 1.0),
        };
        
        let v_dvl = ekf.h_dvl(&x, &u);
        
        // DVL should measure tangential velocity due to rotation
        // v = ω × r = [0, 0, 1] × [1, 0, 0] = [0, 1, 0]
        assert_relative_eq!(v_dvl, Vector3::new(0.0, 1.0, 0.0), epsilon = 1e-10);
    }

    #[test]
    fn test_h_depth() {
        let params = create_test_params();
        let transforms = create_test_transforms();
        let ekf = Ekf::new(&params, &transforms);
        
        // State: at depth of 5m (positive z is down)
        let x = EkfState::new(&Vector3::new(0.0, 0.0, 5.0), &Vector3::zeros());
        
        // Control: no rotation
        let u = EkfControl {
            odom_R_body: Rotation::identity(),
            a_body_B: Vector3::zeros(),
            omega_body_B: Vector3::zeros(),
        };
        
        let depth = ekf.h_depth(&x, &u);
        
        // Depth sensor is 0.1m above body center, so it should read 4.9m
        assert_relative_eq!(depth, 4.9, epsilon = 1e-10);
    }

    #[test]
    fn test_h_depth_with_rotation() {
        let params = create_test_params();
        let transforms = create_test_transforms();
        let ekf = Ekf::new(&params, &transforms);
        
        // State: at origin
        let x = EkfState::new(&Vector3::zeros(), &Vector3::zeros());
        
        // Control: pitched 90 degrees (nose down)
        let u = EkfControl {
            odom_R_body: Rotation::from_axis_angle(&Vector3::y_axis(), std::f64::consts::FRAC_PI_2),
            a_body_B: Vector3::zeros(),
            omega_body_B: Vector3::zeros(),
        };
        
        let depth = ekf.h_depth(&x, &u);
        
        // When pitched 90 degrees nose down, the depth sensor (0.1m above body in body frame)
        // becomes 0.1m forward in odom frame, so depth should still be 0
        assert_relative_eq!(depth, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_predict_covariance() {
        let params = create_test_params();
        let transforms = create_test_transforms();
        let ekf = Ekf::new(&params, &transforms);
        
        // Initial covariance
        let P0 = Matrix6::identity();
        
        let dt = 0.1;
        let P1 = ekf.predict_cov(&P0, dt);
        
        // Check that uncertainty in position increased due to velocity uncertainty
        assert!(P1[(0, 0)] > P0[(0, 0)]);
        assert!(P1[(1, 1)] > P0[(1, 1)]);
        assert!(P1[(2, 2)] > P0[(2, 2)]);
        
        // Check cross-correlation between position and velocity
        assert_relative_eq!(P1[(0, 3)], dt, epsilon = 1e-10);
        assert_relative_eq!(P1[(1, 4)], dt, epsilon = 1e-10);
        assert_relative_eq!(P1[(2, 5)], dt, epsilon = 1e-10);
    }

    #[test]
    fn test_update_with_dvl() {
        let params = create_test_params();
        let transforms = create_test_transforms();
        let ekf = Ekf::new(&params, &transforms);
        
        // Prior state: moving at 1 m/s in x, but with high uncertainty
        let x_prior = EkfState::new(&Vector3::zeros(), &Vector3::new(1.0, 0.0, 0.0));
        let P_prior = Matrix6::identity() * 10.0; // High uncertainty
        
        // DVL measurement: 0.8 m/s in x (slightly different from prior)
        let z_dvl = Vector3::new(0.8, 0.0, 0.0);
        let R_dvl = Matrix3::identity() * 0.01; // Low measurement noise
        
        // Expected measurement based on prior
        let u = EkfControl {
            odom_R_body: Rotation::identity(),
            a_body_B: Vector3::zeros(),
            omega_body_B: Vector3::zeros(),
        };
        let z_expected = ekf.h_dvl(&x_prior, &u);
        
        // Measurement Jacobian for DVL (assuming no rotation and aligned frames)
        let H_dvl = ekf.F_dvl;
        
        let (x_post, P_post) = ekf.update(&x_prior, &P_prior, &z_dvl, &R_dvl, &z_expected, &H_dvl)
            .expect("Update should succeed");
        
        // Posterior velocity should be closer to measurement
        assert!(
            (x_post.v_body_O()[0] - 0.8).abs() < (x_prior.v_body_O()[0] - 0.8).abs()
        );
        
        // Posterior covariance should be smaller
        assert!(P_post[(3, 3)] < P_prior[(3, 3)]);
    }

    #[test]
    fn test_update_with_depth() {
        let params = create_test_params();
        let transforms = create_test_transforms();
        let ekf = Ekf::new(&params, &transforms);
        
        // Prior state: at 5m depth with high uncertainty
        let x_prior = EkfState::new(&Vector3::new(0.0, 0.0, 5.0), &Vector3::zeros());
        let P_prior = Matrix6::identity() * 10.0;
        
        // Depth measurement: 4.5m
        let z_depth = na::SVector::<f64, 1>::new(4.5);
        let R_depth = na::SMatrix::<f64, 1, 1>::new(0.01);
        
        // Expected measurement
        let u = EkfControl {
            odom_R_body: Rotation::identity(),
            a_body_B: Vector3::zeros(),
            omega_body_B: Vector3::zeros(),
        };
        let z_expected = na::SVector::<f64, 1>::new(ekf.h_depth(&x_prior, &u));
        
        // Measurement Jacobian for depth (only sensitive to z position)
        let H_depth = na::SMatrix::<f64, 1, 6>::new(0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
        
        let (x_post, P_post) = ekf.update(&x_prior, &P_prior, &z_depth, &R_depth, &z_expected, &H_depth)
            .expect("Update should succeed");
        
        // Posterior depth should be closer to measurement (accounting for sensor offset)
        // Measurement is 4.5m, sensor is 0.1m above body, so body should be at 4.6m
        assert!(
            (x_post.r_body_O()[2] - 4.6).abs() < (x_prior.r_body_O()[2] - 4.6).abs()
        );
        
        // Posterior covariance in z should be smaller
        assert!(P_post[(2, 2)] < P_prior[(2, 2)]);
    }
}
