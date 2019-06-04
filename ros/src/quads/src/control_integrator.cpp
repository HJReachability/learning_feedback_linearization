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
// Integrate control to match crazyflie inputs.
//
///////////////////////////////////////////////////////////////////////////////

#include <crazyflie_msgs/ControlStamped.h>
#include <quads/control_integrator.h>
#include <quads_msgs/Control.h>

#include <ros/ros.h>

namespace quads {

void ControlIntegrator::RawControlCallback(
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

  rolldot_ += msg->u2 * dt;
  roll_ += rolldot_ * dt;

  pitchdot_ += msg->u3 * dt;
  pitch_ += pitchdot_ * dt;

  yawdot_ += msg->u4 * dt;

  // Publish this guy.
  crazyflie_msgs::ControlStamped integrated_msg;
  integrated_msg.control.thrust = thrust_ ;
  integrated_msg.control.roll = roll_;
  integrated_msg.control.pitch = pitch_;
  integrated_msg.control.yaw_dot = yawdot_;
  crazyflie_control_pub_.publish(integrated_msg);
}

bool ControlIntegrator::Initialize(const ros::NodeHandle& n) {
  name_ = ros::names::append(n.getNamespace(), "control_integrator");

  if (!LoadParameters(n)) {
    ROS_ERROR("%s: Failed to load parameters.", name_.c_str());
    return false;
  }

  if (!RegisterCallbacks(n)) {
    ROS_ERROR("%s: Failed to register callbacks.", name_.c_str());
    return false;
  }

  return true;
}

bool ControlIntegrator::LoadParameters(const ros::NodeHandle& n) {
  ros::NodeHandle nl(n);

  // Topics.
  if (!nl.getParam("topics/raw_control", raw_control_topic_)) return false;
  if (!nl.getParam("topics/crazyflie_control", crazyflie_control_topic_))
    return false;

  return true;
}

bool ControlIntegrator::RegisterCallbacks(const ros::NodeHandle& n) {
  ros::NodeHandle nl(n);

  // Subscribers.
  raw_control_sub_ = nl.subscribe(raw_control_topic_.c_str(), 1,
                                  &ControlIntegrator::RawControlCallback, this);

  // Publisher.
  crazyflie_control_pub_ = nl.advertise<crazyflie_msgs::ControlStamped>(
      crazyflie_control_topic_.c_str(), 1, false);

  return true;
}

}  // namespace quads
