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
// Simulate flight of a 14D quadrotor.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef QUADS_SIMULATOR_14D_H
#define QUADS_SIMULATOR_14D_H

#include <quads/quadrotor_14d.h>
#include <crazyflie_msgs/ControlStamped.h>

#include <geometry_msgs/TransformStamped.h>
#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <string>

namespace quads {

class Simulator14D {
 public:
  ~Simulator14D() {}
  Simulator14D() : received_control_(false), initialized_(false) {}

  // Initialize this class by reading parameters and loading callbacks.
  bool Initialize(const ros::NodeHandle& n);

 private:
  // Load parameters and register callbacks.
  bool LoadParameters(const ros::NodeHandle& n);
  bool RegisterCallbacks(const ros::NodeHandle& n);

  // Callback to process new output msg.
  void TimerCallback(const ros::TimerEvent& e);

  // Update control signal.
  void ControlCallback(const crazyflie_msgs::ControlStamped::ConstPtr& msg);

  // Current state and control.
  Vector14d x_;
  Vector4d u_;
  Quadrotor14D dynamics_;

  // Flag for whether first control signal has been received.
  bool received_control_;

  // Timer.
  ros::Timer timer_;
  double dt_;
  ros::Time last_time_;

  // TF broadcasting.
  tf2_ros::TransformBroadcaster br_;

  // Publishers and subscribers.
  ros::Subscriber control_sub_;
  std::string control_topic_;

  // Frames of reference.
  std::string fixed_frame_id_;
  std::string robot_frame_id_;

  // Initialized flag and name.
  bool initialized_;
  std::string name_;
};  //\class Simulator14D

}  // namespace quads

#endif
