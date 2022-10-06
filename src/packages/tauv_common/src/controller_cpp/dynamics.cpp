#include "dynamics.h"

Dynamics::Dynamics() {
  this->m = 1;
  this->v = 1;
  this->d = 1;
  this->g = 9.81;
  this->rg.setZero();
  this->rb.setZero();
  this->I.setConstant(1);
  this->D1.setZero();
  this->D2.setZero();
  this->Ma.setZero();
}

void Dynamics::set_m(double m) { this->m = m; }

void Dynamics::set_v(double v) { this->v = v; }

void Dynamics::set_d(double d) { this->d = d; }

void Dynamics::set_g(double g) { this->g = g; }

void Dynamics::set_rg(double rg_x, double rg_y, double rg_z) {
  this->rg.setZero();
  this->rg << rg_x, rg_y, rg_z;
}

void Dynamics::set_rb(double rb_x, double rb_y, double rb_z) {
  this->rb.setZero();
  this->rb << rb_x, rb_y, rb_z;
}

void Dynamics::set_I(double I_x, double I_y, double I_z, double I_roll,
                     double I_pitch, double I_yaw) {
  this->I.setZero();
  this->I << I_x, I_y, I_z, I_yaw, I_pitch, I_roll;
}

void Dynamics::set_D1(double D1_x, double D1_y, double D1_z, double D1_roll,
                      double D1_pitch, double D1_yaw) {
  this->D1.setZero();
  this->D1 << D1_x, D1_y, D1_z, D1_roll, D1_pitch, D1_yaw;
}

void Dynamics::set_D2(double D2_x, double D2_y, double D2_z, double D2_roll,
                      double D2_pitch, double D2_yaw) {
  this->D2.setZero();
  this->D2 << D2_x, D2_y, D2_z, D2_roll, D2_pitch, D2_yaw;
}

void Dynamics::set_Ma(double Ma_x, double Ma_y, double Ma_z, double Ma_roll,
                      double Ma_pitch, double Ma_yaw) {
  this->Ma.setZero();
  this->Ma.diagonal() << Ma_x, Ma_y, Ma_z, Ma_roll, Ma_pitch, Ma_yaw;
}

void Dynamics::compute_tau(Eigen::Matrix<double, 6, 1> &tau,
                           const Eigen::Matrix<double, 6, 1> &x,
                           const Eigen::Matrix<double, 6, 1> &v,
                           const Eigen::Matrix<double, 6, 1> &vd) {
  Eigen::Matrix<double, 6, 6> Mrb;
  this->build_Mrb(Mrb);

  Eigen::Matrix<double, 6, 6> Ma = this->Ma;

  Eigen::Matrix<double, 6, 6> Crb;
  this->build_Crb(Crb, v);

  Eigen::Matrix<double, 6, 6> Ca;
  this->build_Ca(Ca, v);

  Eigen::Matrix<double, 6, 6> D;
  this->build_D(D, v);

  Eigen::Matrix<double, 6, 1> G;
  this->build_G(G, x);

  Eigen::Matrix<double, 6, 6> J;
  this->build_J(J, x);

  Eigen::Matrix<double, 6, 6> M = Mrb + Ma;
  Eigen::Matrix<double, 6, 6> C = Crb + Ca;

  tau.setZero();
  tau = M * vd + C * v + D * v + G;
}

void Dynamics::build_J(Eigen::Matrix<double, 6, 6> &J,
                       const Eigen::Matrix<double, 6, 1> &x) {
  double cr = cos(x(3));
  double sr = sin(x(3));
  double cp = cos(x(4));
  double sp = sin(x(4));
  double tp = tan(x(4));
  double cy = cos(x(5));
  double sy = sin(x(5));

  if (abs(tp) < 0.0001) {
    tp = tp < 0 ? -0.0001 : 0.0001;
  }

  J.setZero();

  // Fossen 2.18
  J(0, 0) = cy * cp;
  J(0, 1) = -sy * cr + cy * sp * sr;
  J(0, 2) = sy * sr + cy * cr * sp;
  J(1, 0) = sy * cp;
  J(1, 1) = cy * cr + sr * sp * sy;
  J(1, 2) = -cy * sr + sp * sy * cr;
  J(2, 0) = -sp;
  J(2, 1) = cp * sr;
  J(2, 1) = cp * cr;

  // Fossen 2.28
  J(3, 3) = 1;
  J(3, 4) = sr * tp;
  J(3, 5) = cr * tp;
  J(4, 3) = 0;
  J(4, 4) = cr;
  J(4, 5) = -sr;
  J(5, 3) = 0;
  J(5, 4) = sr / cp;
  J(5, 5) = cr / cp;
}

void Dynamics::build_G(Eigen::Matrix<double, 6, 1> &G,
                       const Eigen::Matrix<double, 6, 1> &x) {
  double rg_x = this->rg(0);
  double rg_y = this->rg(1);
  double rg_z = this->rg(2);
  double rb_x = this->rb(0);
  double rb_y = this->rb(1);
  double rb_z = this->rb(2);

  double W = this->m * this->g;
  double B = this->v * this->d * this->g;

  double cr = cos(x(3));
  double sr = sin(x(3));
  double cp = cos(x(4));
  double sp = sin(x(4));

  G.setZero();

  // Fossen 4.6
  G(0) = (W - B) * sp;
  G(1) = -1 * (W - B) * cp * sr;
  G(2) = -1 * (W - B) * cp * cr;
  G(3) = -1 * (rg_y * W - rb_y * B) * cp * cr + (rg_z * W - rb_z * B) * cp * sr;
  G(4) = (rg_z * W - rb_z * B) * sp + (rg_x * W - rb_x * B) * cp * cr;
  G(5) = -1 * (rg_x * W - rb_x * B) * cp * sr - (rg_y * W - rb_y * B) * sp;
}

void Dynamics::build_D(Eigen::Matrix<double, 6, 6> &D,
                       const Eigen::Matrix<double, 6, 1> &v) {
  Eigen::Matrix<double, 6, 1> va = v.cwiseAbs();
  D.setZero();
  D(0) = this->D1(0) + this->D2(0) * va(0);
  D(1) = this->D1(1) + this->D2(1) * va(1);
  D(2) = this->D1(2) + this->D2(2) * va(2);
  D(3) = this->D1(3) + this->D2(3) * va(3);
  D(4) = this->D1(4) + this->D2(4) * va(4);
  D(5) = this->D1(5) + this->D2(5) * va(5);
}

void Dynamics::build_Ca(Eigen::Matrix<double, 6, 6> &Ca,
                        const Eigen::Matrix<double, 6, 1> &v) {
  double Ma_x = this->Ma(0, 0);
  double Ma_y = this->Ma(1, 1);
  double Ma_z = this->Ma(2, 2);
  double Ma_roll = this->Ma(3, 3);
  double Ma_pitch = this->Ma(4, 4);
  double Ma_yaw = this->Ma(5, 5);

  Ca.setZero();
  // Fossen 6.54
  Ca(0, 4) = Ma_z * v(2);
  Ca(0, 5) = -Ma_y * v(1);
  Ca(1, 3) = -Ma_z * v(2);
  Ca(1, 5) = Ma_x * v(0);
  Ca(2, 3) = Ma_y * v(1);
  Ca(2, 4) = -Ma_x * v(0);
  Ca(3, 1) = Ma_z * v(2);
  Ca(3, 2) = -Ma_y * v(1);
  Ca(3, 4) = Ma_yaw * v(5);
  Ca(3, 5) = -Ma_pitch * v(4);
  Ca(4, 0) = -Ma_z * v(1);
  Ca(4, 2) = Ma_x * v(0);
  Ca(4, 3) = -Ma_yaw * v(5);
  Ca(4, 5) = Ma_roll * v(3);
  Ca(5, 0) = Ma_y * v(1);
  Ca(5, 1) = -Ma_x * v(0);
  Ca(5, 3) = Ma_pitch * v(4);
  Ca(5, 4) = -Ma_roll * v(3);
}

void Dynamics::build_Crb(Eigen::Matrix<double, 6, 6> &Crb,
                         const Eigen::Matrix<double, 6, 1> &v) {
  double m = this->m;
  double rg_x = this->rg(0);
  double rg_y = this->rg(1);
  double rg_z = this->rg(2);
  double Ixx = this->I(0);
  double Iyy = this->I(1);
  double Izz = this->I(2);
  double Ixy = this->I(3);
  double Ixz = this->I(4);
  double Iyz = this->I(5);

  Eigen::Matrix<double, 3, 3> Crb12;
  Crb12.setZero();
  Crb12(0, 0) = m * (rg_y * v(4) + rg_z * v(5));
  Crb12(0, 1) = -m * (rg_x * v(4) - v(2));
  Crb12(0, 2) = -m * (rg_x * v(5) + v(1));
  Crb12(1, 0) = -m * (rg_y * v(3) + v(2));
  Crb12(1, 1) = m * (rg_z * v(5) + rg_x * v(3));
  Crb12(1, 2) = -m * (rg_y * v(5) - v(0));
  Crb12(2, 0) = -m * (rg_z * v(3) - v(1));
  Crb12(2, 1) = -m * (rg_z * v(4) + v(0));
  Crb12(2, 2) = m * (rg_x * v(3) + rg_y * v(4));

  Eigen::Matrix<double, 3, 3> Crb22;
  Crb22.setZero();
  Crb22(0, 1) = -Iyz * v(4) - Ixz * v(3) + Izz * v(5);
  Crb22(0, 2) = Iyz * v(5) + Ixy * v(3) - Iyy * v(4);
  Crb22(1, 0) = Ixy * v(4) + Ixz * v(3) - Izz * v(5);
  Crb22(1, 2) = Ixy * v(5) + Ixy * v(4) + Ixx * v(3);
  Crb22(2, 0) = -Iyz * v(5) - Ixy * v(3) - Iyy * v(4);
  Crb22(2, 1) = Ixz * v(5) + Ixy * v(4) - Ixx * v(3);

  Crb.setZero();
  Crb.block<3, 3>(0, 3) = Crb12;
  Crb.block<3, 3>(3, 0) = -1 * Crb12.transpose();
  Crb.block<3, 3>(3, 3) = Crb22;
}

void Dynamics::build_Mrb(Eigen::Matrix<double, 6, 6> &Mrb) {
  double m = this->m;
  double rg_x = this->rg(0);
  double rg_y = this->rg(1);
  double rg_z = this->rg(2);
  double Ixx = this->I(0);
  double Iyy = this->I(1);
  double Izz = this->I(2);
  double Ixy = this->I(3);
  double Ixz = this->I(4);
  double Iyz = this->I(5);

  Mrb.setZero();
  Mrb(0, 0) = m;
  Mrb(0, 4) = m * rg_z;
  Mrb(0, 5) = -m * rg_y;
  Mrb(1, 1) = m;
  Mrb(1, 3) = -m * rg_z;
  Mrb(1, 5) = m * rg_x;
  Mrb(2, 2) = m;
  Mrb(2, 3) = m * rg_y;
  Mrb(2, 4) = -m * rg_x;
  Mrb(3, 1) = -m * rg_z;
  Mrb(3, 2) = m * rg_y;
  Mrb(3, 3) = Ixx;
  Mrb(3, 4) = -Ixy;
  Mrb(3, 5) = -Ixz;
  Mrb(4, 0) = m * rg_z;
  Mrb(4, 2) = -m * rg_x;
  Mrb(4, 3) = -Ixy;
  Mrb(4, 4) = Iyy;
  Mrb(4, 5) = -Iyz;
  Mrb(5, 0) = -m * rg_y;
  Mrb(5, 1) = m * rg_x;
  Mrb(5, 3) = -Ixz;
  Mrb(5, 4) = -Iyz;
  Mrb(5, 5) = Izz;
}