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
// Generate an infinite-horizon reference signal for the linear system to track.
//
///////////////////////////////////////////////////////////////////////////////

#include <quads/reference_generator.h>
#include <quads_msgs/OutputDerivatives.h>

#include <ros/ros.h>
#include <std_msgs/Empty.h>

namespace quads {

bool ReferenceGenerator::Initialize(const ros::NodeHandle& n) {
  name_ = ros::names::append(n.getNamespace(), "reference_generator");

  if (!LoadParameters(n)) {
    ROS_ERROR("%s: Failed to load parameters.", name_.c_str());
    return false;
  }

  if (!RegisterCallbacks(n)) {
    ROS_ERROR("%s: Failed to register callbacks.", name_.c_str());
    return false;
  }

  initialized_ = true;
  return true;
}

bool ReferenceGenerator::LoadParameters(const ros::NodeHandle& n) {
  ros::NodeHandle nl(n);

  // Topics.
  if (!nl.getParam("topics/in_flight", in_flight_topic_)) return false;
  if (!nl.getParam("topics/reference", reference_topic_)) return false;

  // Frequence of sinusoids.
  if (!nl.getParam("freq/x", x_freq_)) return false;
  if (!nl.getParam("freq/y", y_freq_)) return false;
  if (!nl.getParam("freq/z", z_freq_)) return false;
  if (!nl.getParam("freq/psi", psi_freq_)) return false;

  // Time step.
  if (!nl.getParam("dt", dt_)) {
    dt_ = 0.01;
    ROS_WARN("%s: Time discretization set to %lf (s).", name_.c_str(), dt_);
  }

  return true;
}

bool ReferenceGenerator::RegisterCallbacks(const ros::NodeHandle& n) {
  ros::NodeHandle nl(n);

  // Subscribers.
  in_flight_sub_ = nl.subscribe(in_flight_topic_.c_str(), 1,
                                &ReferenceGenerator::InFlightCallback, this);

  // Publisher.
  reference_pub_ = nl.advertise<quads_msgs::OutputDerivatives>(
      reference_topic_.c_str(), 1, false);

  // Timer.
  timer_ = nl.createTimer(ros::Duration(dt_),
                          &ReferenceGenerator::TimerCallback, this);

  return true;
}

void ReferenceGenerator::TimerCallback(const ros::TimerEvent& e) {
  quads_msgs::OutputDerivatives msg;

  constexpr double kLateralAmplitude = 1.0;
  constexpr double kAverageHeight = 1.5;
  constexpr double kPsiAmplitude = 0.25; //M_PI_2;

  if (!in_flight_) return;

  const double t = ros::Time::now().toSec() - in_flight_time_;
  msg.x = kLateralAmplitude * std::sin(x_freq_ * t);
  msg.xdot1 = kLateralAmplitude * x_freq_ * std::cos(x_freq_ * t);
  msg.xdot2 = -kLateralAmplitude * x_freq_ * x_freq_ * std::sin(x_freq_ * t);
  msg.xdot3 =
      -kLateralAmplitude * x_freq_ * x_freq_ * x_freq_ * std::cos(x_freq_ * t);

  msg.y = kLateralAmplitude * std::cos(y_freq_ * t);
  msg.ydot1 = -kLateralAmplitude * y_freq_ * std::sin(y_freq_ * t);
  msg.ydot2 = -kLateralAmplitude * y_freq_ * y_freq_ * std::cos(y_freq_ * t);
  msg.ydot3 =
      kLateralAmplitude * y_freq_ * y_freq_ * y_freq_ * std::sin(y_freq_ * t);

  msg.z = kAverageHeight - std::sin(z_freq_ * t);
  msg.zdot1 = -z_freq_ * std::cos(z_freq_ * t);
  msg.zdot2 = z_freq_ * z_freq_ * std::sin(z_freq_ * t);
  msg.zdot3 = z_freq_ * z_freq_ * z_freq_ * std::cos(z_freq_ * t);

  msg.psi = kPsiAmplitude * std::cos(psi_freq_ * t);
  msg.psidot1 = -kPsiAmplitude * psi_freq_ * std::sin(psi_freq_ * t);

  reference_pub_.publish(msg);
}

}  // namespace quads
