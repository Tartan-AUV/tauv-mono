#![allow(non_snake_case)]

use std::collections::{BTreeMap, VecDeque};
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

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
struct DvlInput {
    v_dvl_V: Vector3,
    R: Matrix3
}

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
struct TimestampedControl {
    t: DateTime<Utc>,
    control: EkfControl,
}

#[derive(Clone, Debug)]
struct StateEstimate {
    state: EkfState,
    cov: EkfCov,
}

#[derive(Debug)]
enum MeasurementType {
    Dvl(DvlInput),
    Depth(DepthInput),
}

#[derive(Debug)]
struct EkfHistory {
    // Control inputs (IMU) stored in order of arrival
    control_history: VecDeque<TimestampedControl>,
    
    // State estimates indexed by time
    state_history: BTreeMap<DateTime<Utc>, (MeasurementType, StateEstimate)>,
    
    // Track last measurement times
    last_dvl_t: DateTime<Utc>,
    last_depth_t: DateTime<Utc>,
    last_imu_t: DateTime<Utc>,
    
    // Maximum number of control inputs to keep
    max_control_history: usize,
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
        let max_dt = TimeDelta::from_std(Duration::from_millis(200)).unwrap();

        if max_dt < (t_depth - t_dvl).abs() || max_dt < (t_depth - t_imu).abs() ||
            max_dt < (t_imu - t_dvl).abs() {
            return Err("Initial measurements are too far apart in time".to_string())
        }

        // Initialize state using depth measurement
        let state = EkfState::new(&Vector3::new(0.0, 0.0, depth.z), &Vector3::zeros());
        let var_r = params.initial_position_stddev_m.powi(2);
        let var_v = params.initial_velocity_stddev_mps.powi(2);
        let cov = Matrix6::from_diagonal(&na::vector![var_r, var_r, depth.R, var_v, var_v, var_v]);

        let mut control_history = VecDeque::with_capacity(params.history_length);
        control_history.push_back(TimestampedControl { t: t_imu, control: imu });

        let mut state_history = BTreeMap::new();
        state_history.insert(t_depth, (
            MeasurementType::Depth(depth),
            StateEstimate { state: state.clone(), cov }
        ));
        state_history.insert(t_dvl, (
            MeasurementType::Dvl(dvl),
            StateEstimate { state: state.clone(), cov }
        ));

        Ok(Self {
            control_history,
            state_history,
            last_depth_t: t_depth,
            last_dvl_t: t_dvl,
            last_imu_t: t_imu,
            max_control_history: params.history_length,
        })
    }

    pub fn add_imu_measurement(&mut self, t: DateTime<Utc>, imu: EkfControl) -> Result<(), String> {
        // Check that the measurement is newer than the last IMU measurement
        if t <= self.last_imu_t {
            return Err(format!("IMU measurement at {:?} is not newer than last IMU at {:?}", t, self.last_imu_t));
        }
        
        // Check that the measurement is newer than the latest DVL measurement
        if t <= self.last_dvl_t {
            return Err(format!("IMU measurement at {:?} is not newer than last DVL at {:?}", t, self.last_dvl_t));
        }
        
        // Add to control history
        self.control_history.push_back(TimestampedControl { t, control: imu });
        
        // Remove old control inputs if we exceed the maximum
        while self.control_history.len() > self.max_control_history {
            self.control_history.pop_front();
        }
        
        self.last_imu_t = t;
        Ok(())
    }
    
    pub fn add_depth_measurement(
        &mut self, 
        t: DateTime<Utc>, 
        depth: DepthInput,
        ekf: &Ekf,
    ) -> Result<(), String> {
        // Check that the measurement is newer than the last depth measurement
        if t <= self.last_depth_t {
            return Err(format!("Depth measurement at {:?} is not newer than last depth at {:?}", t, self.last_depth_t));
        }
        
        // Check that the measurement is newer than the latest DVL measurement
        if t <= self.last_dvl_t {
            return Err(format!("Depth measurement at {:?} is not newer than last DVL at {:?}", t, self.last_dvl_t));
        }
        
        // Find the closest IMU measurement
        let closest_imu = self.find_closest_control(t)?;
        
        // Find the most recent state estimate before this depth measurement
        let (state_t, state_est) = self.find_latest_state_before(t)?;
        
        // Predict from the state time to the depth measurement time using the closest IMU
        let dt = (t - state_t).num_milliseconds() as f64 / 1000.0;
        if dt < 0.0 {
            return Err("Negative time delta in prediction".to_string());
        }
        
        let predicted_state = ekf.predict(&state_est.state, &closest_imu.control, dt);
        let predicted_cov = ekf.predict_cov(&state_est.cov, dt);
        
        // Perform the depth update
        let z_depth = na::SVector::<f64, 1>::new(depth.z);
        let R_depth = na::SMatrix::<f64, 1, 1>::new(depth.R);
        let z_expected = na::SVector::<f64, 1>::new(ekf.h_depth(&predicted_state, &closest_imu.control));
        
        // Measurement Jacobian for depth (only sensitive to z position)
        let H_depth = na::SMatrix::<f64, 1, 6>::new(0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
        
        let (updated_state, updated_cov) = ekf.update(
            &predicted_state,
            &predicted_cov,
            &z_depth,
            &R_depth,
            &z_expected,
            &H_depth
        )?;
        
        // Insert the new state estimate
        self.state_history.insert(t, (
            MeasurementType::Depth(depth),
            StateEstimate { state: updated_state, cov: updated_cov }
        ));
        
        self.last_depth_t = t;
        
        // Clean up old state estimates that are before the oldest control input
        if let Some(oldest_control) = self.control_history.front() {
            self.state_history.retain(|&state_t, _| state_t >= oldest_control.t);
        }
        
        Ok(())
    }
    
    // Helper method to find the closest control input
    fn find_closest_control(&self, t: DateTime<Utc>) -> Result<&TimestampedControl, String> {
        if self.control_history.is_empty() {
            return Err("No control inputs in history".to_string());
        }
        
        // Binary search for the closest control
        let idx = match self.control_history.binary_search_by_key(&t, |tc| tc.t) {
            Ok(idx) => idx, // Exact match
            Err(idx) => {
                // Not found, idx is where it would be inserted
                if idx == 0 {
                    0 // Use first control
                } else if idx >= self.control_history.len() {
                    self.control_history.len() - 1 // Use last control
                } else {
                    // Choose between idx-1 and idx based on which is closer
                    let prev_diff = (t - self.control_history[idx - 1].t).abs();
                    let next_diff = (self.control_history[idx].t - t).abs();
                    if prev_diff <= next_diff {
                        idx - 1
                    } else {
                        idx
                    }
                }
            }
        };
        
        self.control_history.get(idx)
            .ok_or_else(|| "Failed to find control in history".to_string())
    }
    
    // Helper method to find the latest state estimate before a given time
    fn find_latest_state_before(&self, t: DateTime<Utc>) -> Result<(DateTime<Utc>, &StateEstimate), String> {
        // Use range to get all entries before time t
        self.state_history
            .range(..t)
            .rev()
            .next()
            .map(|(time, (_, state))| (*time, state))
            .ok_or_else(|| "No state estimate found before the given time".to_string())
    }
    
    // Helper method to get the current best estimate (latest state)
    pub fn get_latest_state(&self) -> Option<(DateTime<Utc>, &StateEstimate)> {
        self.state_history
            .iter()
            .rev()
            .next()
            .map(|(t, (_, state))| (*t, state))
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

    // Helper function to create test inputs
    fn create_test_depth_input() -> DepthInput {
        DepthInput { z: 5.0, R: 0.01 }
    }

    fn create_test_dvl_input() -> DvlInput {
        DvlInput {
            v_dvl_V: Vector3::new(1.0, 0.5, 0.2),
            R: Matrix3::identity() * 0.01,
        }
    }

    fn create_test_imu_control() -> EkfControl {
        EkfControl {
            odom_R_body: Rotation::identity(),
            a_body_B: Vector3::new(0.1, 0.0, 0.0),
            omega_body_B: Vector3::new(0.0, 0.0, 0.1),
        }
    }

    fn create_test_timestamps() -> (DateTime<Utc>, DateTime<Utc>, DateTime<Utc>) {
        let base_time = Utc.with_ymd_and_hms(2024, 1, 1, 12, 0, 0).unwrap();
        let t_depth = base_time;
        let t_dvl = base_time + chrono::Duration::milliseconds(10);
        let t_imu = base_time + chrono::Duration::milliseconds(20);
        (t_depth, t_dvl, t_imu)
    }

    /// Tests successful initialization of EkfHistory with valid measurements.
    /// Verifies that initial state, control history, and measurement timestamps are correctly stored.
    #[test]
    fn test_ekf_history_initialization_success() {
        let params = create_test_params();
        let (t_depth, t_dvl, t_imu) = create_test_timestamps();
        let depth = create_test_depth_input();
        let dvl = create_test_dvl_input();
        let imu = create_test_imu_control();

        let history = EkfHistory::try_new(t_depth, t_dvl, t_imu, depth.clone(), dvl, imu, &params);
        
        assert!(history.is_ok());
        let history = history.unwrap();
        
        // Check that timestamps are stored correctly
        assert_eq!(history.last_depth_t, t_depth);
        assert_eq!(history.last_dvl_t, t_dvl);
        assert_eq!(history.last_imu_t, t_imu);
        
        // Check that control history has one entry
        assert_eq!(history.control_history.len(), 1);
        assert_eq!(history.control_history[0].t, t_imu);
        
        // Check that state history has two entries (depth and dvl)
        assert_eq!(history.state_history.len(), 2);
        assert!(history.state_history.contains_key(&t_depth));
        assert!(history.state_history.contains_key(&t_dvl));
        
        // Check that initial state is set correctly based on depth
        let (_, state_est) = history.state_history.get(&t_depth).unwrap();
        assert_relative_eq!(state_est.state.r_body_O()[2], depth.z, epsilon = 1e-10);
        assert_relative_eq!(state_est.state.v_body_O().into_owned(), Vector3::zeros(), epsilon = 1e-10);
    }

    /// Tests that EkfHistory initialization fails when measurements are too far apart in time.
    /// Ensures temporal synchronization requirements are enforced.
    #[test]
    fn test_ekf_history_initialization_measurements_too_far_apart() {
        let params = create_test_params();
        let base_time = Utc.with_ymd_and_hms(2024, 1, 1, 12, 0, 0).unwrap();
        let t_depth = base_time;
        let t_dvl = base_time + chrono::Duration::milliseconds(10);
        let t_imu = base_time + chrono::Duration::seconds(1); // Too far apart
        
        let depth = create_test_depth_input();
        let dvl = create_test_dvl_input();
        let imu = create_test_imu_control();

        let history = EkfHistory::try_new(t_depth, t_dvl, t_imu, depth, dvl, imu, &params);
        
        assert!(history.is_err());
        assert!(history.unwrap_err().contains("too far apart"));
    }

    /// Tests successful addition of a new IMU measurement to the history.
    /// Verifies that control history is updated and timestamps are tracked correctly.
    #[test]
    fn test_ekf_history_add_imu_measurement_success() {
        let params = create_test_params();
        let (t_depth, t_dvl, t_imu) = create_test_timestamps();
        let depth = create_test_depth_input();
        let dvl = create_test_dvl_input();
        let imu = create_test_imu_control();

        let mut history = EkfHistory::try_new(t_depth, t_dvl, t_imu, depth, dvl, imu.clone(), &params).unwrap();
        
        // Add a new IMU measurement
        let new_imu_time = t_imu + chrono::Duration::milliseconds(50);
        let new_imu = create_test_imu_control();
        
        let result = history.add_imu_measurement(new_imu_time, new_imu);
        assert!(result.is_ok());
        
        // Check that the measurement was added
        assert_eq!(history.control_history.len(), 2);
        assert_eq!(history.last_imu_t, new_imu_time);
        assert_eq!(history.control_history.back().unwrap().t, new_imu_time);
    }

    /// Tests that adding an IMU measurement older than the last IMU measurement is rejected.
    /// Ensures temporal ordering constraints are enforced for IMU measurements.
    #[test]
    fn test_ekf_history_add_imu_measurement_not_newer_than_last_imu() {
        let params = create_test_params();
        let (t_depth, t_dvl, t_imu) = create_test_timestamps();
        let depth = create_test_depth_input();
        let dvl = create_test_dvl_input();
        let imu = create_test_imu_control();

        let mut history = EkfHistory::try_new(t_depth, t_dvl, t_imu, depth, dvl, imu.clone(), &params).unwrap();
        
        // Try to add an IMU measurement that's not newer
        let old_imu_time = t_imu - chrono::Duration::milliseconds(10);
        let new_imu = create_test_imu_control();
        
        let result = history.add_imu_measurement(old_imu_time, new_imu);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not newer than last IMU"));
    }

    /// Tests that adding an IMU measurement older than the last DVL measurement is rejected.
    /// Ensures causal ordering between different measurement types.
    #[test]
    fn test_ekf_history_add_imu_measurement_not_newer_than_last_dvl() {
        let params = create_test_params();
        let (t_depth, t_dvl, t_imu) = create_test_timestamps();
        let depth = create_test_depth_input();
        let dvl = create_test_dvl_input();
        let imu = create_test_imu_control();

        let mut history = EkfHistory::try_new(t_depth, t_dvl, t_imu, depth, dvl, imu.clone(), &params).unwrap();
        
        // Add a new IMU measurement after the initial one
        let new_imu_time = t_imu + chrono::Duration::milliseconds(50);
        history.add_imu_measurement(new_imu_time, imu.clone()).unwrap();
        
        // Now add a DVL measurement even later
        let new_dvl_time = new_imu_time + chrono::Duration::milliseconds(50);
        // Note: We need to add DVL measurement functionality to properly test this
        // For now, manually update the last_dvl_t to simulate a DVL measurement
        history.last_dvl_t = new_dvl_time;
        
        // Try to add an IMU measurement that's older than the new DVL time
        let old_imu_time = new_dvl_time - chrono::Duration::milliseconds(25);
        let test_imu = create_test_imu_control();
        
        let result = history.add_imu_measurement(old_imu_time, test_imu);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not newer than last DVL"));
    }

    /// Tests that the control history respects the maximum history length limit.
    /// Verifies that old control inputs are automatically removed when the limit is exceeded.
    #[test]
    fn test_ekf_history_add_imu_measurement_max_history_limit() {
        let mut params = create_test_params();
        params.history_length = 2; // Set small limit for testing
        
        let (t_depth, t_dvl, t_imu) = create_test_timestamps();
        let depth = create_test_depth_input();
        let dvl = create_test_dvl_input();
        let imu = create_test_imu_control();

        let mut history = EkfHistory::try_new(t_depth, t_dvl, t_imu, depth, dvl, imu.clone(), &params).unwrap();
        
        // Add IMU measurements up to and beyond the limit
        let mut current_time = t_imu;
        for i in 1..=3 {
            current_time = current_time + chrono::Duration::milliseconds(50);
            let result = history.add_imu_measurement(current_time, imu.clone());
            assert!(result.is_ok());
        }
        
        // Should only keep the last 2 measurements
        assert_eq!(history.control_history.len(), 2);
        assert_eq!(history.control_history.front().unwrap().t, 
                   t_imu + chrono::Duration::milliseconds(100));
    }

    /// Tests successful addition of a depth measurement that triggers EKF prediction and update.
    /// Verifies that state history is expanded and measurement tracking is updated.
    #[test]
    fn test_ekf_history_add_depth_measurement_success() {
        let params = create_test_params();
        let transforms = create_test_transforms();
        let ekf = Ekf::new(&params, &transforms);
        
        let (t_depth, t_dvl, t_imu) = create_test_timestamps();
        let depth = create_test_depth_input();
        let dvl = create_test_dvl_input();
        let imu = create_test_imu_control();

        let mut history = EkfHistory::try_new(t_depth, t_dvl, t_imu, depth, dvl, imu.clone(), &params).unwrap();
        
        // Add an IMU measurement that's newer than the DVL to satisfy the constraint
        let new_imu_time = t_dvl + chrono::Duration::milliseconds(50);
        history.add_imu_measurement(new_imu_time, imu.clone()).unwrap();
        
        // Add a new depth measurement after the IMU
        let new_depth_time = new_imu_time + chrono::Duration::milliseconds(50);
        let new_depth = DepthInput { z: 6.0, R: 0.02 };
        
        let result = history.add_depth_measurement(new_depth_time, new_depth, &ekf);
        assert!(result.is_ok());
        
        // Check that the measurement was added
        assert_eq!(history.last_depth_t, new_depth_time);
        assert!(history.state_history.contains_key(&new_depth_time));
        
        // After cleanup, only states after the oldest control input are kept
        // The oldest control input is now at t_imu (20ms), so we should have:
        // - DVL state at 10ms (kept because it's >= oldest control at 20ms? No, it's before)
        // - New depth state at 110ms
        // Actually, the cleanup removes states before the oldest control, so only the new depth should remain
        assert_eq!(history.state_history.len(), 1);
        assert!(history.state_history.contains_key(&new_depth_time));
    }

    /// Tests that adding a depth measurement older than the last depth measurement is rejected.
    /// Ensures temporal ordering constraints for depth measurements.
    #[test]
    fn test_ekf_history_add_depth_measurement_not_newer_than_last_depth() {
        let params = create_test_params();
        let transforms = create_test_transforms();
        let ekf = Ekf::new(&params, &transforms);
        
        let (t_depth, t_dvl, t_imu) = create_test_timestamps();
        let depth = create_test_depth_input();
        let dvl = create_test_dvl_input();
        let imu = create_test_imu_control();

        let mut history = EkfHistory::try_new(t_depth, t_dvl, t_imu, depth, dvl, imu, &params).unwrap();
        
        // Try to add a depth measurement that's not newer
        let old_depth_time = t_depth - chrono::Duration::milliseconds(10);
        let new_depth = DepthInput { z: 6.0, R: 0.02 };
        
        let result = history.add_depth_measurement(old_depth_time, new_depth, &ekf);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not newer than last depth"));
    }

    /// Tests that adding a depth measurement older than the last DVL measurement is rejected.
    /// Ensures causal ordering between depth and DVL measurements.
    #[test]
    fn test_ekf_history_add_depth_measurement_not_newer_than_last_dvl() {
        let params = create_test_params();
        let transforms = create_test_transforms();
        let ekf = Ekf::new(&params, &transforms);
        
        let (t_depth, t_dvl, t_imu) = create_test_timestamps();
        let depth = create_test_depth_input();
        let dvl = create_test_dvl_input();
        let imu = create_test_imu_control();

        let mut history = EkfHistory::try_new(t_depth, t_dvl, t_imu, depth, dvl, imu, &params).unwrap();
        
        // Try to add a depth measurement that's older than the last DVL
        let old_depth_time = t_dvl - chrono::Duration::milliseconds(5);
        let new_depth = DepthInput { z: 6.0, R: 0.02 };
        
        let result = history.add_depth_measurement(old_depth_time, new_depth, &ekf);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not newer than last DVL"));
    }

    /// Tests the binary search algorithm for finding the closest control input by timestamp.
    /// Verifies correct behavior for exact matches, interpolation, and edge cases.
    #[test]
    fn test_ekf_history_find_closest_control() {
        let params = create_test_params();
        let (t_depth, t_dvl, t_imu) = create_test_timestamps();
        let depth = create_test_depth_input();
        let dvl = create_test_dvl_input();
        let imu = create_test_imu_control();

        let mut history = EkfHistory::try_new(t_depth, t_dvl, t_imu, depth, dvl, imu.clone(), &params).unwrap();
        
        // Add more control inputs
        let t_imu2 = t_imu + chrono::Duration::milliseconds(100);
        let t_imu3 = t_imu + chrono::Duration::milliseconds(200);
        history.add_imu_measurement(t_imu2, imu.clone()).unwrap();
        history.add_imu_measurement(t_imu3, imu.clone()).unwrap();
        
        // Test finding closest control
        let query_time = t_imu + chrono::Duration::milliseconds(150);
        let closest = history.find_closest_control(query_time).unwrap();
        assert_eq!(closest.t, t_imu2); // Should be closest to t_imu2
        
        // Test exact match
        let exact_match = history.find_closest_control(t_imu2).unwrap();
        assert_eq!(exact_match.t, t_imu2);
        
        // Test before first control
        let before_first = t_imu - chrono::Duration::milliseconds(50);
        let first_control = history.find_closest_control(before_first).unwrap();
        assert_eq!(first_control.t, t_imu);
        
        // Test after last control
        let after_last = t_imu3 + chrono::Duration::milliseconds(50);
        let last_control = history.find_closest_control(after_last).unwrap();
        assert_eq!(last_control.t, t_imu3);
    }

    /// Tests error handling when attempting to find closest control with empty history.
    /// Verifies that appropriate error messages are returned.
    #[test]
    fn test_ekf_history_find_closest_control_empty_history() {
        let params = create_test_params();
        let (t_depth, t_dvl, t_imu) = create_test_timestamps();
        let depth = create_test_depth_input();
        let dvl = create_test_dvl_input();
        let imu = create_test_imu_control();

        let mut history = EkfHistory::try_new(t_depth, t_dvl, t_imu, depth, dvl, imu, &params).unwrap();
        
        // Clear the control history
        history.control_history.clear();
        
        let query_time = t_imu + chrono::Duration::milliseconds(50);
        let result = history.find_closest_control(query_time);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No control inputs"));
    }

    /// Tests finding the latest state estimate that occurred before a given timestamp.
    /// Verifies correct temporal ordering and state retrieval from the BTreeMap.
    #[test]
    fn test_ekf_history_find_latest_state_before() {
        let params = create_test_params();
        let transforms = create_test_transforms();
        let ekf = Ekf::new(&params, &transforms);
        
        let (t_depth, t_dvl, t_imu) = create_test_timestamps();
        let depth = create_test_depth_input();
        let dvl = create_test_dvl_input();
        let imu = create_test_imu_control();

        let mut history = EkfHistory::try_new(t_depth, t_dvl, t_imu, depth, dvl, imu.clone(), &params).unwrap();
        
        // Add an IMU measurement that's newer than the DVL
        let new_imu_time = t_dvl + chrono::Duration::milliseconds(50);
        history.add_imu_measurement(new_imu_time, imu.clone()).unwrap();
        
        // Add a new depth measurement
        let new_depth_time = new_imu_time + chrono::Duration::milliseconds(50);
        let new_depth = DepthInput { z: 6.0, R: 0.02 };
        history.add_depth_measurement(new_depth_time, new_depth, &ekf).unwrap();
        
        // After cleanup, the only remaining state is the new depth measurement
        // So we can only test finding states before times after the new depth measurement
        let late_query_time = new_depth_time + chrono::Duration::milliseconds(50);
        let (late_state_time, _) = history.find_latest_state_before(late_query_time).unwrap();
        assert_eq!(late_state_time, new_depth_time);
        
        // Test that querying before the new depth time fails (no states before it)
        let early_query_time = new_depth_time - chrono::Duration::milliseconds(10);
        let result = history.find_latest_state_before(early_query_time);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No state estimate found"));
    }

    /// Tests retrieval of the most recent state estimate from the history.
    /// Verifies that the latest timestamp is correctly identified as new measurements are added.
    #[test]
    fn test_ekf_history_get_latest_state() {
        let params = create_test_params();
        let transforms = create_test_transforms();
        let ekf = Ekf::new(&params, &transforms);
        
        let (t_depth, t_dvl, t_imu) = create_test_timestamps();
        let depth = create_test_depth_input();
        let dvl = create_test_dvl_input();
        let imu = create_test_imu_control();

        let mut history = EkfHistory::try_new(t_depth, t_dvl, t_imu, depth, dvl, imu.clone(), &params).unwrap();
        
        // Initially, latest should be dvl (since it's later than depth)
        let (latest_time, _) = history.get_latest_state().unwrap();
        assert_eq!(latest_time, t_dvl);
        
        // Add an IMU measurement that's newer than the DVL
        let new_imu_time = t_dvl + chrono::Duration::milliseconds(50);
        history.add_imu_measurement(new_imu_time, imu.clone()).unwrap();
        
        // Add a new depth measurement
        let new_depth_time = new_imu_time + chrono::Duration::milliseconds(50);
        let new_depth = DepthInput { z: 6.0, R: 0.02 };
        history.add_depth_measurement(new_depth_time, new_depth, &ekf).unwrap();
        
        // Now latest should be the new depth
        let (latest_time, _) = history.get_latest_state().unwrap();
        assert_eq!(latest_time, new_depth_time);
    }

    /// Tests automatic cleanup of old state estimates when control history is trimmed.
    /// Verifies that memory usage is bounded and old states are properly removed.
    #[test]
    fn test_ekf_history_state_cleanup() {
        let mut params = create_test_params();
        params.history_length = 2; // Small limit for testing
        
        let transforms = create_test_transforms();
        let ekf = Ekf::new(&params, &transforms);
        
        let (t_depth, t_dvl, t_imu) = create_test_timestamps();
        let depth = create_test_depth_input();
        let dvl = create_test_dvl_input();
        let imu = create_test_imu_control();

        let mut history = EkfHistory::try_new(t_depth, t_dvl, t_imu, depth, dvl, imu.clone(), &params).unwrap();
        
        // Add several IMU measurements to trigger cleanup
        let mut current_time = t_imu;
        for _ in 0..5 {
            current_time = current_time + chrono::Duration::milliseconds(50);
            history.add_imu_measurement(current_time, imu.clone()).unwrap();
        }
        
        // Add a depth measurement to trigger state cleanup
        let new_depth_time = current_time + chrono::Duration::milliseconds(50);
        let new_depth = DepthInput { z: 6.0, R: 0.02 };
        history.add_depth_measurement(new_depth_time, new_depth, &ekf).unwrap();
        
        // Check that old states were cleaned up
        // Should only keep states that are after the oldest control input
        let oldest_control_time = history.control_history.front().unwrap().t;
        for (state_time, _) in &history.state_history {
            assert!(*state_time >= oldest_control_time);
        }
    }

    /// Tests that measurement types (Depth vs DVL) are correctly stored and retrieved.
    /// Verifies that measurement data is preserved accurately in the state history.
    #[test]
    fn test_ekf_history_measurement_type_storage() {
        let params = create_test_params();
        let transforms = create_test_transforms();
        let ekf = Ekf::new(&params, &transforms);
        
        let (t_depth, t_dvl, t_imu) = create_test_timestamps();
        let depth = create_test_depth_input();
        let dvl = create_test_dvl_input();
        let imu = create_test_imu_control();

        let mut history = EkfHistory::try_new(t_depth, t_dvl, t_imu, depth, dvl, imu.clone(), &params).unwrap();
        
        // Check that measurement types are stored correctly
        let (depth_measurement, _) = history.state_history.get(&t_depth).unwrap();
        match depth_measurement {
            MeasurementType::Depth(_) => {}, // Expected
            _ => panic!("Expected depth measurement type"),
        }
        
        let (dvl_measurement, _) = history.state_history.get(&t_dvl).unwrap();
        match dvl_measurement {
            MeasurementType::Dvl(_) => {}, // Expected
            _ => panic!("Expected DVL measurement type"),
        }
        
        // Add an IMU measurement that's newer than the DVL
        let new_imu_time = t_dvl + chrono::Duration::milliseconds(50);
        history.add_imu_measurement(new_imu_time, imu.clone()).unwrap();
        
        // Add a new depth measurement and check its type
        let new_depth_time = new_imu_time + chrono::Duration::milliseconds(50);
        let new_depth = DepthInput { z: 6.0, R: 0.02 };
        history.add_depth_measurement(new_depth_time, new_depth, &ekf).unwrap();
        
        let (new_depth_measurement, _) = history.state_history.get(&new_depth_time).unwrap();
        match new_depth_measurement {
            MeasurementType::Depth(stored_depth) => {
                assert_relative_eq!(stored_depth.z, 6.0, epsilon = 1e-10);
                assert_relative_eq!(stored_depth.R, 0.02, epsilon = 1e-10);
            },
            _ => panic!("Expected depth measurement type"),
        }
    }

    /// Tests error handling when no state estimates exist before the queried timestamp.
    /// Verifies that appropriate error messages are returned for edge cases.
    #[test]
    fn test_ekf_history_find_latest_state_before_no_state() {
        let params = create_test_params();
        let (t_depth, t_dvl, t_imu) = create_test_timestamps();
        let depth = create_test_depth_input();
        let dvl = create_test_dvl_input();
        let imu = create_test_imu_control();

        let history = EkfHistory::try_new(t_depth, t_dvl, t_imu, depth, dvl, imu, &params).unwrap();
        
        // Query for a time before all states
        let early_time = t_depth - chrono::Duration::milliseconds(50);
        let result = history.find_latest_state_before(early_time);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No state estimate found"));
    }
}
