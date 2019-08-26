/*
 * Copyright (c) 2019, The Regents of the University of California (Regents).
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *    1. Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *    2. Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 *
 *    3. Neither the name of the copyright holder nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Please contact the author(s) of this library if you have any questions.
 * Authors: David Fridovich-Keil   ( dfk@eecs.berkeley.edu )
 */

///////////////////////////////////////////////////////////////////////////////
//
// EKF to estimate the state of the 12D quadrotor.Roughly following variable
// naming scheme of https://en.wikipedia.org/wiki/Extended_Kalman_filter.
//
///////////////////////////////////////////////////////////////////////////////

#include <quads/quadrotor14d.h>
#include <quads/state_estimator.h>

#include <quads_msgs/Control.h>
#include <quads_msgs/Output.h>
#include <quads_msgs/State.h>

#include <crazyflie_msgs/ControlStamped.h>

#include <geometry_msgs/TransformStamped.h>
#include <math.h>
#include <ros/ros.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_listener.h>
#include <Eigen/Dense>
#include <string>

namespace quads {

void StateEstimator::TimerCallback(const ros::TimerEvent& e) {
  double x, y, z, phi, theta, psi;
  tf_parser_.GetXYZRPY(&x, &y, &z, &phi, &theta, &psi);

  const double t = ros::Time::now().toSec();

  // Update all the polynomial fits.
  smoother_x_.Update(x, t);
  smoother_y_.Update(y, t);
  smoother_z_.Update(z, t);
  smoother_theta_.Update(theta, t);
  smoother_phi_.Update(phi, t);
  smoother_psi_.Update(psi, t);

  // Now handle output derivatives.
  quads_msgs::OutputDerivatives derivs_msg;
  derivs_msg.x = smoother_x_.Interpolate(t, 0);
  derivs_msg.xdot1 = smoother_x_.Interpolate(t, 1);
  derivs_msg.xdot2 = smoother_x_.Interpolate(t, 2);
  derivs_msg.xdot3 = smoother_x_.Interpolate(t, 3);

  derivs_msg.y = smoother_y_.Interpolate(t, 0);
  derivs_msg.ydot1 = smoother_y_.Interpolate(t, 1);
  derivs_msg.ydot2 = smoother_y_.Interpolate(t, 2);
  derivs_msg.ydot3 = smoother_y_.Interpolate(t, 3);

  derivs_msg.z = smoother_z_.Interpolate(t, 0);
  derivs_msg.zdot1 = smoother_z_.Interpolate(t, 1);
  derivs_msg.zdot2 = smoother_z_.Interpolate(t, 2);
  derivs_msg.zdot3 = smoother_z_.Interpolate(t, 3);

  derivs_msg.psi = smoother_psi_.Interpolate(t, 0);
  derivs_msg.psidot1 = smoother_psi_.Interpolate(t, 1);

  output_derivs_pub_.publish(derivs_msg);

  last_linear_system_state_estimate_.reset(
      new quads_msgs::OutputDerivatives(derivs_msg));

  if (!in_flight_) {
    ROS_WARN_THROTTLE(1.0,
                      "%s: Waiting to estimate state until we are in flight.",
                      name_.c_str());
    return;
  }

  // Publish the answer!
  quads_msgs::State state_msg;
  state_msg.x = smoother_x_.Interpolate(t, 0);
  state_msg.y = smoother_y_.Interpolate(t, 0);
  state_msg.z = smoother_z_.Interpolate(t, 0);
  state_msg.theta = smoother_theta_.Interpolate(t, 0);
  state_msg.phi = smoother_phi_.Interpolate(t, 0);
  state_msg.psi = smoother_psi_.Interpolate(t, 0);
  state_msg.dx = smoother_x_.Interpolate(t, 1);
  state_msg.dy = smoother_y_.Interpolate(t, 1);
  state_msg.dz = smoother_z_.Interpolate(t, 1);
  state_msg.zeta = thrust_;
  state_msg.xi = thrustdot_;
  state_msg.q = smoother_theta_.Interpolate(t, 1);
  state_msg.r = smoother_phi_.Interpolate(t, 1);
  state_msg.p = smoother_psi_.Interpolate(t, 1);

  // Handle zeta = 0.0, which should only happen upon initialization.
  if (state_msg.zeta >= 0.0) state_msg.zeta = std::max(1e-1, state_msg.zeta);
  if (state_msg.zeta <= 0.0) state_msg.zeta = std::min(-1e-1, state_msg.zeta);

  state_pub_.publish(state_msg);

  // Compute reward.
  // This is the norm of the difference between the linear system state we just
  // estimated and that of the linear system we expect.

  // Catch no linear system state.
  if (linear_system_state_.hasNaN() || reference_.hasNaN() ||
      std::isnan(last_linear_system_state_update_time_ ||
                 last_control_ == nullptr)) {
    last_linear_system_state_update_time_ = t;
    return;
  }

  // Integrate linear system state forward, trying to track the current
  // reference.
  const double dt = t - last_linear_system_state_update_time_;
  linear_system_state_ += dt * (A_ + B_ * K_) * linear_system_state_.eval();

  // Compute reward now.
  VectorXd our_linear_system_state(14);
  our_linear_system_state << derivs_msg.x, derivs_msg.xdot1, derivs_msg.xdot2,
      derivs_msg.xdot3, derivs_msg.y, derivs_msg.ydot1, derivs_msg.ydot2,
      derivs_msg.ydot3, derivs_msg.z, derivs_msg.zdot1, derivs_msg.zdot2,
      derivs_msg.zdot3, derivs_msg.psi, derivs_msg.psidot1;
  const double r =
      -(linear_system_state_ - our_linear_system_state).squaredNorm();

  /*
  quads_msgs::Transition transition_msg;
  transition_msg.x = state_msg;
  transition_msg.u = *last_control_;
  transition_msg.r = r;

  transitions_pub_.publish(transition_msg);
  */
}

bool StateEstimator::Initialize(const ros::NodeHandle& n) {
  name_ = ros::names::append(n.getNamespace(), "state_estimator");

  if (!LoadParameters(n)) {
    ROS_ERROR("%s: Failed to load parameters.", name_.c_str());
    return false;
  }

  if (!dynamics_.Initialize(n)) return false;
  if (!tf_parser_.Initialize(n)) return false;

  if (!RegisterCallbacks(n)) {
    ROS_ERROR("%s: Failed to register callbacks.", name_.c_str());
    return false;
  }

  // Rescale initial thrust to be a force not an acceleration.
  thrust_ = 9.81 * dynamics_.Mass();

  // Hard code linear feedback controller.
  K_ = MatrixXd::Zero(4, 14);
  K_.row(0).segment(0, 4) << 3.16, 6.19, 6.07, 3.48;
  K_.row(1).segment(4, 4) << 3.16, 6.19, 6.07, 3.48;
  K_.row(2).segment(8, 4) << 3.16, 6.19, 6.07, 3.48;
  K_.row(3).segment(12, 2) << 3.16, 2.51;

  // A and B of linear system.
  A_ = MatrixXd::Zero(14, 14);
  A_(0, 1) = 1.0;
  A_(1, 2) = 1.0;
  A_(2, 3) = 1.0;
  A_(4, 5) = 1.0;
  A_(5, 6) = 1.0;
  A_(6, 7) = 1.0;
  A_(8, 9) = 1.0;
  A_(9, 10) = 1.0;
  A_(10, 11) = 1.0;
  A_(12, 13) = 1.0;

  B_ = MatrixXd::Zero(14, 4);
  B_(3, 0) = 1.0;
  B_(7, 1) = 1.0;
  B_(11, 2) = 1.0;
  B_(13, 3) = 1.0;

  return true;
}

bool StateEstimator::LoadParameters(const ros::NodeHandle& n) {
  ros::NodeHandle nl(n);

  // Topics.
  if (!nl.getParam("topics/in_flight", in_flight_topic_)) return false;
  if (!nl.getParam("topics/state", state_topic_)) return false;
  if (!nl.getParam("topics/output_derivs", output_derivs_topic_)) return false;
  if (!nl.getParam("topics/control", control_topic_)) return false;
  if (!nl.getParam("topics/transitions", transitions_topic_)) return false;
  if (!nl.getParam("topics/linear_system_reset", linear_system_reset_topic_))
    return false;
  if (!nl.getParam("topics/reference", reference_topic_)) return false;

  // Time step.
  if (!nl.getParam("dt", dt_)) {
    dt_ = 0.01;
    ROS_WARN("%s: Time discretization set to %lf (s).", name_.c_str(), dt_);
  }

  return true;
}

bool StateEstimator::RegisterCallbacks(const ros::NodeHandle& n) {
  ros::NodeHandle nl(n);

  // Subscribers.
  control_sub_ = nl.subscribe(control_topic_.c_str(), 1,
                              &StateEstimator::ControlCallback, this);
  in_flight_sub_ = nl.subscribe(in_flight_topic_.c_str(), 1,
                                &StateEstimator::InFlightCallback, this);
  linear_system_reset_sub_ =
      nl.subscribe(linear_system_reset_topic_.c_str(), 1,
                   &StateEstimator::LinearSystemResetCallback, this);
  reference_sub_ = nl.subscribe(reference_topic_.c_str(), 1,
                                &StateEstimator::ReferenceCallback, this);

  // Publisher.
  state_pub_ = nl.advertise<quads_msgs::State>(state_topic_.c_str(), 1, false);
  output_derivs_pub_ = nl.advertise<quads_msgs::OutputDerivatives>(
      output_derivs_topic_.c_str(), 1, false);
  transitions_pub_ = nl.advertise<quads_msgs::Transition>(
      transitions_topic_.c_str(), 1, false);

  // Timer.
  timer_ =
      nl.createTimer(ros::Duration(dt_), &StateEstimator::TimerCallback, this);

  return true;
}

inline void StateEstimator::LinearSystemResetCallback(
    const std_msgs::Empty::ConstPtr& msg) {
  if (!last_linear_system_state_estimate_) return;
  linear_system_state_(0) = last_linear_system_state_estimate_->x;
  linear_system_state_(1) = last_linear_system_state_estimate_->xdot1;
  linear_system_state_(2) = last_linear_system_state_estimate_->xdot2;
  linear_system_state_(3) = last_linear_system_state_estimate_->xdot3;
  linear_system_state_(4) = last_linear_system_state_estimate_->y;
  linear_system_state_(5) = last_linear_system_state_estimate_->ydot1;
  linear_system_state_(6) = last_linear_system_state_estimate_->ydot2;
  linear_system_state_(7) = last_linear_system_state_estimate_->ydot3;
  linear_system_state_(8) = last_linear_system_state_estimate_->z;
  linear_system_state_(9) = last_linear_system_state_estimate_->zdot1;
  linear_system_state_(10) = last_linear_system_state_estimate_->zdot2;
  linear_system_state_(11) = last_linear_system_state_estimate_->zdot3;
  linear_system_state_(12) = last_linear_system_state_estimate_->psi;
  linear_system_state_(13) = last_linear_system_state_estimate_->psidot1;
}

void StateEstimator::ReferenceCallback(
    const quads_msgs::OutputDerivatives::ConstPtr& msg) {
  reference_(0) = msg->x;
  reference_(1) = msg->xdot1;
  reference_(2) = msg->xdot2;
  reference_(3) = msg->xdot3;
  reference_(4) = msg->y;
  reference_(5) = msg->ydot1;
  reference_(6) = msg->ydot2;
  reference_(7) = msg->ydot3;
  reference_(8) = msg->z;
  reference_(9) = msg->zdot1;
  reference_(10) = msg->zdot2;
  reference_(11) = msg->zdot3;
  reference_(12) = msg->psi;
  reference_(13) = msg->psidot1;
}

inline void StateEstimator::ControlCallback(
    const quads_msgs::Control::ConstPtr& msg) {
  const double t = ros::Time::now().toSec();

  if (!std::isnan(last_control_time_)) {
    const double dt = t - last_control_time_;
    thrust_ += thrustdot_ * dt;
    thrustdot_ += msg->thrustdot2 * dt;
  }

  // Record control.
  last_control_ = msg;

  /*
  constexpr double kExtraAccel = 3.0;
  thrust = std::max(
      9.81 - kExtraAccel, std::min(9.81 + kExtraAccel, thrust));
  */

  last_control_time_ = t;
}

}  // namespace quads
