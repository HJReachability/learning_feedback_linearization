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

#include <geometry_msgs/TransformStamped.h>
#include <math.h>
#include <ros/ros.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_listener.h>
#include <Eigen/Dense>
#include <string>

namespace quads {
namespace {
// Assumed process and observation noise covariances.
static const Matrix14x14d kProcessNoise = 0.01 * Matrix14x14d::Identity();
static const Matrix6x6d kOutputNoise = 0.0001 * Matrix6x6d::Identity();
}  // anonymous namespace

void StateEstimator::TimerCallback(const ros::TimerEvent& e) {
  if (!control_.get()) return;

  // Parse control and observation msgs into vectors.
  const Vector4d u(control_->u1, control_->u2, control_->u3, control_->u4);

  double x, y, z, phi, theta, psi;
  tf_parser_.GetXYZRPY(&x, &y, &z, &phi, &theta, &psi);

  Vector6d meas;
  meas << x, y, z, phi, theta, psi;

  // Compute F and H.
  const Matrix14x14d F = dynamics_.StateJacobian(x_, u);
  const Matrix6x14d H = dynamics_.OutputJacobian(x_);

  // Predict step.
  const Vector14d x_predict = dt_ * dynamics_(x_, u);
  const Matrix14x14d P_predict = F * P_ * F.transpose() + kProcessNoise;

  // Update step.
  const Vector6d innovation = meas - x_predict.head<6>();
  const Matrix6x6d S = H * P_predict * H.transpose() + kOutputNoise;
  const Matrix14x6d K = P_predict * H.transpose() * S.inverse();
  x_ = x_predict + K * innovation;
  P_ = (Matrix14x14d::Identity() - K * H) * P_predict;

  // Publish the answer!
  quads_msgs::State msg;
  msg.x = x_(dynamics_.kXIdx);
  msg.y = x_(dynamics_.kYIdx);
  msg.z = x_(dynamics_.kZIdx);
  msg.theta = x_(dynamics_.kThetaIdx);
  msg.phi = x_(dynamics_.kPhiIdx);
  msg.psi = x_(dynamics_.kPsiIdx);
  msg.dx = x_(dynamics_.kDxIdx);
  msg.dy = x_(dynamics_.kDyIdx);
  msg.dz = x_(dynamics_.kDzIdx);
  msg.zeta = x_(dynamics_.kZetaIdx);
  msg.xi = x_(dynamics_.kXiIdx);
  msg.q = x_(dynamics_.kQIdx);
  msg.r = x_(dynamics_.kRIdx);
  msg.p = x_(dynamics_.kPIdx);

  state_pub_.publish(msg);
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

  return true;
}

bool StateEstimator::LoadParameters(const ros::NodeHandle& n) {
  ros::NodeHandle nl(n);

  // Topics.
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
  control_sub_ = nl.subscribe(control_topic_.c_str(), 1,
                              &StateEstimator::ControlCallback, this);

  // Publisher.
  state_pub_ = nl.advertise<quads_msgs::State>(state_topic_.c_str(), 1, false);

  // Timer.
  timer_ =
      nl.createTimer(ros::Duration(dt_), &StateEstimator::TimerCallback, this);

  return true;
}

inline void StateEstimator::ControlCallback(
    const quads_msgs::Control::ConstPtr& msg) {
  control_ = msg;
}

}  // namespace quads
