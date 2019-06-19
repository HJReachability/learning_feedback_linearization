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

  // Publish the answer!
  quads_msgs::State msg;
  msg.x = smoother_x_.Interpolate(t, 0);
  msg.y = smoother_y_.Interpolate(t, 0);
  msg.z = smoother_z_.Interpolate(t, 0);
  msg.theta = smoother_theta_.Interpolate(t, 0);
  msg.phi = smoother_phi_.Interpolate(t, 0);
  msg.psi = smoother_psi_.Interpolate(t, 0);
  msg.dx = smoother_x_.Interpolate(t, 1);
  msg.dy = smoother_y_.Interpolate(t, 1);
  msg.dz = smoother_z_.Interpolate(t, 1);
  msg.zeta = thrust_;
  msg.xi = thrustdot_;
  msg.q = smoother_theta_.Interpolate(t, 1);
  msg.r = smoother_phi_.Interpolate(t, 1);
  msg.p = smoother_psi_.Interpolate(t, 1);

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
  if (!nl.getParam("topics/in_flight", in_flight_topic_)) return false;
  if (!nl.getParam("topics/state", state_topic_)) return false;
  if (!nl.getParam("topics/control", control_topic_)) return false;
  if (!nl.getParam("topics/takeoff_control", takeoff_control_topic_))
    return false;

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
  takeoff_control_sub_ = nl.subscribe(takeoff_control_topic_.c_str(), 1,
                                      &StateEstimator::ControlCallback, this);
  in_flight_sub_ = nl.subscribe(in_flight_topic_.c_str(), 1,
                                &StateEstimator::InFlightCallback, this);

  // Publisher.
  state_pub_ = nl.advertise<quads_msgs::State>(state_topic_.c_str(), 1, false);

  // Timer.
  timer_ =
      nl.createTimer(ros::Duration(dt_), &StateEstimator::TimerCallback, this);

  return true;
}

inline void StateEstimator::ControlCallback(
    const quads_msgs::Control::ConstPtr& msg) {
  if (std::isnan(time_of_last_msg_)) {
    time_of_last_msg_ = ros::Time::now().toSec();
    return;
  }

  const double current_time = ros::Time::now().toSec();
  const double dt = current_time - time_of_last_msg_;
  time_of_last_msg_ = current_time;

  // Integrate stuff.
  thrustdot_ += msg->u1 * dt;
  thrust_ += thrustdot_ * dt;

  // Antiwindup.
  constexpr double kExtraAccel = 3.0;
  thrust_ = std::max(9.81 - kExtraAccel, std::min(thrust_, 9.81 + kExtraAccel));
}

}  // namespace quads
