/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    constrainedExample.h
 * @brief   Simple constrained optimization scenarios.
 * @author  Yetong Zhang
 * @date    Aug 3, 2024
 */

#pragma once

// #include <gtsam/constrained/ConstrainedOptProblem.h>
#include <gtsam/nonlinear/expressions.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/constrained/NonlinearEqualityConstraint.h>
#include <gtsam/constrained/NonlinearInequalityConstraint.h>

namespace constrained_example {

using namespace gtsam;

/// Exponential function e^x.
inline double exp_func(const double& x, gtsam::OptionalJacobian<1, 1> H1 = {}) {
  double result = exp(x);
  if (H1) H1->setConstant(result);
  return result;
}

/// Exponential expression e^x.
inline Double_ exp(const Double_& x) { return Double_(exp_func, x); }

/// Pow functor used for pow function.
class PowFunctor {
 private:
  double c_;

 public:
  PowFunctor(const double& c) : c_(c) {}

  double operator()(const double& x, gtsam::OptionalJacobian<1, 1> H1 = {}) const {
    if (H1) H1->setConstant(c_ * pow(x, c_ - 1));
    return pow(x, c_);
  }
};

/// Pow function.
inline Double_ pow(const Double_& x, const double& c) {
  PowFunctor pow_functor(c);
  return Double_(pow_functor, x);
}

/// Plus between Double expression and double.
inline Double_ operator+(const Double_& x, const double& d) { return x + Double_(d); }

/// Negative sign operator.
inline Double_ operator-(const Double_& x) { return Double_(0.0) - x; }

/// Keys for creating expressions.
Symbol x1_key('x', 1);
Symbol x2_key('x', 2);
Double_ x1(x1_key), x2(x2_key);

}  // namespace constrained_example

