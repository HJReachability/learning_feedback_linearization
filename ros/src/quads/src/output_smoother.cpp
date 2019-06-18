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
// Differentiate and smooth the outputs.
//
///////////////////////////////////////////////////////////////////////////////

#include <quads/output_smoother.h>

#include <quads_msgs/Control.h>
#include <quads_msgs/Output.h>
#include <quads_msgs/OutputDerivatives.h>

#include <geometry_msgs/TransformStamped.h>
#include <math.h>
#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <string>

namespace quads {

bool OutputSmoother::Initialize(const ros::NodeHandle& n) {
  name_ = ros::names::append(n.getNamespace(), "output_smoother");

  if (!LoadParameters(n)) {
    ROS_ERROR("%s: Failed to load parameters.", name_.c_str());
    return false;
  }

  if (!tf_parser_.Initialize(n)) return false;

  if (!RegisterCallbacks(n)) {
    ROS_ERROR("%s: Failed to register callbacks.", name_.c_str());
    return false;
  }

  return true;
}

bool OutputSmoother::LoadParameters(const ros::NodeHandle& n) {
  ros::NodeHandle nl(n);

  // Topics.
  if (!nl.getParam("topics/output_derivs", output_derivs_topic_)) return false;

  // Time step.
  if (!nl.getParam("dt", dt_)) {
    dt_ = 0.01;
    ROS_WARN("%s: Time discretization set to %lf (s).", name_.c_str(), dt_);
  }

  return true;
}

bool OutputSmoother::RegisterCallbacks(const ros::NodeHandle& n) {
  ros::NodeHandle nl(n);

  // Publisher.
  output_derivs_pub_ = nl.advertise<quads_msgs::OutputDerivatives>(
      output_derivs_topic_.c_str(), 1, false);

  // Timer.
  timer_ =
      nl.createTimer(ros::Duration(dt_), &OutputSmoother::TimerCallback, this);

  return true;
}

void OutputSmoother::TimerCallback(const ros::TimerEvent& e) {
  double x, y, z, phi, theta, psi;
  tf_parser_.GetXYZRPY(&x, &y, &z, &phi, &theta, &psi);

  // Get the current time.
  const double t = ros::Time::now().toSec();

  // Update filters and publish msg.
  smoother_x_.Update(x, t);
  smoother_y_.Update(y, t);
  smoother_z_.Update(z, t);
  smoother_psi_.Update(psi, t);

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
}

}  // namespace quads
