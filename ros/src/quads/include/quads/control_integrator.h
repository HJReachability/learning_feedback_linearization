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

#ifndef QUADS_CONTROL_INTEGRATOR_H
#define QUADS_CONTROL_INTEGRATOR_H

#include <quads/quadrotor14d.h>

#include <crazyflie_msgs/PrioritizedControlStamped.h>
#include <quads_msgs/Control.h>

#include <ros/ros.h>
#include <std_msgs/Empty.h>

namespace quads {

class ControlIntegrator {
 public:
  ~ControlIntegrator() {}
  ControlIntegrator()
      : thrust_(9.81),
        thrustdot_(0.0),
        roll_(0.0),
        rolldot_(0.0),
        pitch_(0.0),
        pitchdot_(0.0),
        yawdot_(0.0),
        prioritized_(true),
        time_of_last_msg_(std::numeric_limits<double>::quiet_NaN()),
        in_flight_(false),
        initialized_(false) {}

  // Initialize this class by reading parameters and loading callbacks.
  bool Initialize(const ros::NodeHandle& n);

 private:
  // Load parameters and register callbacks.
  bool LoadParameters(const ros::NodeHandle& n);
  bool RegisterCallbacks(const ros::NodeHandle& n);

  // Callback to process new control msg.
  void RawControlCallback(const quads_msgs::Control::ConstPtr& msg);

  // In flight?
  void InFlightCallback(const std_msgs::Empty::ConstPtr& msg) {
    in_flight_ = true;
  }

  // Callback to handle simulator restarts.
  void SimulatorRestartCallback(const std_msgs::Empty::ConstPtr& msg);

  // Keep track of integral(s) of raw control inputs.
  double thrust_, thrustdot_;
  double roll_, rolldot_;
  double pitch_, pitchdot_;
  double yawdot_;
  double time_of_last_msg_;

  // Dynamics.
  Quadrotor14D dynamics_;

  // Is this signal prioritized?
  bool prioritized_;

  // Are we in flight?
  bool in_flight_;

  // Publishers and subscribers.
  ros::Subscriber in_flight_sub_;
  ros::Subscriber raw_control_sub_;
  ros::Subscriber restart_simulator_sub_;
  ros::Publisher crazyflie_control_pub_;

  std::string in_flight_topic_;
  std::string raw_control_topic_;
  std::string crazyflie_control_topic_;
  std::string restart_simulator_topic_;

  // Initialized flag and name.
  bool initialized_;
  std::string name_;
};  //\class ControlIntegrator

}  // namespace quads

#endif
