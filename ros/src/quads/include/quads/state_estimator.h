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
// EKF to estimate 12D quadrotor state. Roughly following variable naming
// scheme of https://en.wikipedia.org/wiki/Extended_Kalman_filter.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef QUADS_STATE_ESTIMATOR_H
#define QUADS_STATE_ESTIMATOR_H

#include <quads/polynomial_fit.h>
#include <quads/quadrotor14d.h>
#include <quads/tf_parser.h>
#include <quads/types.h>
#include <quads_msgs/Control.h>
#include <quads_msgs/Output.h>

#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <Eigen/Dense>
#include <limits>
#include <string>

namespace quads {

class StateEstimator {
 public:
  ~StateEstimator() {}
  StateEstimator()
      : thrust_(9.81),
        thrustdot_(0.0),
        time_of_last_msg_(std::numeric_limits<double>::quiet_NaN()),
        in_flight_(false),
        initialized_(false) {}

  // Initialize this class by reading parameters and loading callbacks.
  bool Initialize(const ros::NodeHandle& n);

 private:
  // Load parameters and register callbacks.
  bool LoadParameters(const ros::NodeHandle& n);
  bool RegisterCallbacks(const ros::NodeHandle& n);

  // Are we in flight?
  void InFlightCallback(const std_msgs::Empty::ConstPtr& msg) {
    in_flight_ = true;
  }

  // Callback to process new control msgs.
  void ControlCallback(const quads_msgs::Control::ConstPtr& msg);

  // Timer callback and utility to compute Jacobian.
  void TimerCallback(const ros::TimerEvent& e);

  // Are we in flight?
  bool in_flight_;

  // Polynomial fits.
  PolynomialFit<3, 20> smoother_x_, smoother_y_, smoother_z_, smoother_psi_,
      smoother_theta_, smoother_phi_;

  // Keep track of integral(s) of raw control inputs.
  double thrust_, thrustdot_;
  double time_of_last_msg_;

  // Dynamics.
  Quadrotor14D dynamics_;

  // Most recent msg and time discretization (with timer).
  quads_msgs::Control control_;
  ros::Timer timer_;
  double dt_;

  // Publishers and subscribers.
  ros::Subscriber in_flight_sub_;
  ros::Subscriber takeoff_control_sub_;
  ros::Subscriber control_sub_;
  ros::Publisher state_pub_;

  std::string in_flight_topic_;
  std::string takeoff_control_topic_;
  std::string control_topic_;
  std::string state_topic_;

  // Tf parser.
  TfParser tf_parser_;

  // Initialized flag and name.
  bool initialized_;
  std::string name_;
};  //\class StateEstimator

}  // namespace quads

#endif
