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
// Controller just to be used for takeoff. Implemented as a P controller on
// position and yaw.
//
///////////////////////////////////////////////////////////////////////////////

#include <quads/takeoff_controller.h>
#include <quads_msgs/Control.h>

#include <ros/ros.h>

namespace quads {

bool TakeoffController::Initialize(const ros::NodeHandle& n) {
  name_ = ros::names::append(n.getNamespace(), "takeoff_controller");

  if (!LoadParameters(n)) {
    ROS_ERROR("%s: Failed to load parameters.", name_.c_str());
    return false;
  }

  if (!RegisterCallbacks(n)) {
    ROS_ERROR("%s: Failed to register callbacks.", name_.c_str());
    return false;
  }

  if (!tf_parser_.Initialize(n)) return false;

  initialized_ = true;
  return true;
}

bool TakeoffController::LoadParameters(const ros::NodeHandle& n) {
  ros::NodeHandle nl(n);

  if (!nl.getParam("topics/control", control_topic_)) return false;
  if (!nl.getParam("time_step", dt_)) return false;
  if (!nl.getParam("hover/x", hover_x_)) hover_x_ = 0.0;
  if (!nl.getParam("hover/y", hover_y_)) hover_y_ = 0.0;
  if (!nl.getParam("hover/z", hover_z_)) hover_z_ = 1.0;

  return true;
}

bool TakeoffController::RegisterCallbacks(const ros::NodeHandle& n) {
  ros::NodeHandle nl(n);

  // Subscribers.
  control_pub_ =
      nl.advertise<quads_msgs::Control>(control_topic_.c_str(), 1, false);

  // Timer.
  timer_ = nl.createTimer(ros::Duration(dt_), &TakeoffController::TimerCallback,
                          this);

  return true;
}

void TakeoffController::TimerCallback(const ros::TimerEvent& e) {
  constexpr double kDxGain = 0.1;
  constexpr double kDyGain = 0.1;
  constexpr double kDzGain = 0.5;
  constexpr double kDpsiGain = 0.3;

  // Compute the current pose.
  double x, y, z, phi, theta, psi;
  tf_parser_.GetXYZRPY(&x, &y, &z, &phi, &theta, &psi);

  // Compute errors and feedback.
  quads_msgs::Control msg;
  msg.thrustdot2 = -kDzGain * (z - hover_z_);
  msg.pitchdot2 = -kDxGain * (x - hover_x_);
  msg.rolldot2 = kDyGain * (y - hover_y_);
  msg.yawdot2 = -kDpsiGain * (psi - 0.0);
  control_pub_.publish(msg);
}

}  // namespace quads
