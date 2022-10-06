#include <eigen3/Eigen/Dense>
#include <numeric>

#pragma once

class Dynamics {
public:
  Dynamics();

  void set_m(double m);
  void set_v(double v);
  void set_d(double d);
  void set_g(double g);
  void set_rg(double rg_x, double rg_y, double rg_z);
  void set_rb(double rb_x, double rb_y, double rb_z);
  void set_I(double I_x, double I_y, double I_z, double I_roll, double I_pitch,
             double I_yaw);
  void set_D1(double D_x, double D_y, double D_z, double D_roll, double D_pitch,
              double D_yaw);
  void set_D2(double D2_x, double D2_y, double D2_z, double D2_roll,
              double D2_pitch, double D2_yaw);
  void set_Ma(double Ma_x, double Ma_y, double Ma_z, double Ma_roll,
              double Ma_pitch, double Ma_yaw);

  void compute_tau(Eigen::Matrix<double, 6, 1> &tau,
                   const Eigen::Matrix<double, 6, 1> &x,
                   const Eigen::Matrix<double, 6, 1> &v,
                   const Eigen::Matrix<double, 6, 1> &vd);

private:
  double m;
  double v;
  double d;
  double g;
  Eigen::Vector3d rg;
  Eigen::Vector3d rb;
  Eigen::Matrix<double, 6, 1> I;
  Eigen::Matrix<double, 6, 1> D1;
  Eigen::Matrix<double, 6, 1> D2;
  Eigen::Matrix<double, 6, 6> Ma;

  void build_J(Eigen::Matrix<double, 6, 6> &J,
               const Eigen::Matrix<double, 6, 1> &x);

  void build_G(Eigen::Matrix<double, 6, 1> &G,
               const Eigen::Matrix<double, 6, 1> &x);

  void build_D(Eigen::Matrix<double, 6, 6> &D,
               const Eigen::Matrix<double, 6, 1> &v);

  void build_Mrb(Eigen::Matrix<double, 6, 6> &Mrb);

  void build_Ca(Eigen::Matrix<double, 6, 6> &Ca,
                const Eigen::Matrix<double, 6, 1> &v);

  void build_Crb(Eigen::Matrix<double, 6, 6> &Crb,
                 const Eigen::Matrix<double, 6, 1> &v);
};