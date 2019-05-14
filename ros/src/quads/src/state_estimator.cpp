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

#include <quads/state_estimator.h>

#include <quads_msgs/Output.h>
#include <quads_msgs/State.h>

#include <math.h>
#include <ros/ros.h>
#include <Eigen/Dense>
#include <string>

namespace quads {
namespace {
// Assumed process and observation noise covariances.
static constexpr Matrix12d kProcessNoise = 0.01 * Matrix12d::Identity();
static constexpr Matrix5d kOutputNoise = 0.01 * Matrix5d::Identity();
}  // anonymous namespace

void StateEstimator::TimerCallback(const ros::TimerEvent& e) {
  if (!control_.get() || !output_.get()) return;

  // Parse control and observation msgs into vectors.
  const Vector3d u(control_.u1, control_.u2, control_.u3);
  const Vector3d y(output_.x, output_.y, output_.z);

  // Compute F and H.
  const Matrix12d F = dynamics_.StateJacobian(x_, u_);
  const Matrix12d H = dynamics_.OutputJacobian(x_);

  // Predict step.
  const Vector12d x_predict = dt_ * dynamics_(x_, u);
  const Matrix12d P_predict = F * P_ * F.transpose() + kProcessNoise;

  // Update step.
  const Vector5d innovation = y - x_predict.head<5>();
  const Matrix5d S = H * P_predict * H.transpose() + kOutputNoise;
  const Eigen::Matrix<double, 12, 5> K =
      P_predict * H.transpose() * S.pseudoInverse();
  x_ = x_predict + K * innovation;
  P_ = (Matrix12d::Identity() - K * H) * P_predict;
}

bool StateEstimator::Initialize(const ros::NodeHandle& n) {
  name_ = ros::names::append(n.getNamespace(), "state_estimator");

  if (!LoadParameters(n)) {
    ROS_ERROR("%s: Failed to load parameters.", name_.c_str());
    return false;
  }

  if (!dynamics_.LoadParameters(n)) {
    ROS_ERROR("%s: Failed to load parameters.", name_.c_str());
    return false;
  }

  if (!RegisterCallbacks(n)) {
    ROS_ERROR("%s: Failed to register callbacks.", name_.c_str());
    return false;
  }

  return true;
}

bool StateEstimator::LoadParameters(const ros::NodeHandle& n) {
  ros::NodeHandle nl(n);

  // Topics.
  if (!nl.getParam("topics/output", output_topic_)) return false;
  if (!nl.getParam("topics/state", state_topic_)) return false;

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
  output_sub_ = nl.subscribe(output_topic_.c_str(), 1,
                             &StateEstimator::OutputCallback, this);
  control_sub_ = nl.subscribe(control_topic_.c_str(), 1,
                              &StateEstimator::ControlCallback, this);

  // Publisher.
  state_pub_ = nl.advertise<quads_msgs::State>(statetopic_.c_str(), 1, false);

  // Timer.
  timer_ =
      nl.createTimer(ros::Duration(dt_), &StateEstimator::TimerCallback, this);

  return true;
}

inline void StateEstimator::OutputCallback(
    const quads_msgs::Output::ConstPtr& msg) {
  output_ = msg;
}

inline void StateEstimator::ControlCallback(
    const quads_msgs::Output::ConstPtr& msg) {
  control_ = msg;
}

Matrix12d StateEstimator::Jacobian() const {
  const float theta = x_(kThetaIdx);
  const float phi = x_(kPhiIdx);
  const float zeta = x_(kZetaIdx);

  Matrix12d F;
  F << 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, << 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, << 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 0, << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, << 0, 0, 0,
      zeta * cos(phi) * cos(theta), -zeta * sin(phi) * sin(theta), 0, 0, 0,
      cos(phi) * sin(theta), 0, 0, 0, << 0, 0, 0, 0, -zeta * cos(phi), 0, 0, 0,
      -sin(phi), 0, 0, 0, << 0, 0, 0, -zeta * cos(phi) * sin(theta),
      -zeta * cos(theta) * sin(phi), 0, 0, 0, cos(phi) * cos(theta), 0, 0,
      0, << 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, << 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0;
  return F;
}

}  // namespace quads
