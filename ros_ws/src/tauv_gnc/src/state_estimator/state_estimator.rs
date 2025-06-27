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
    Imu(EkfControl, EkfState, EkfCov),
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
        map.insert(t_imu, Event::Imu(imu, state.clone(), cov));

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
