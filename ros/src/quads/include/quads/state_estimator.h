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
#include <quads_msgs/OutputDerivatives.h>
#include <quads_msgs/Transition.h>

#include <crazyflie_msgs/ControlStamped.h>
#include <std_msgs/Empty.h>

#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <Eigen/Dense>
#include <limits>
#include <string>

namespace quads {

using Eigen::VectorXd;
using Eigen::MatrixXd;

class StateEstimator {
 public:
  ~StateEstimator() {}
  StateEstimator()
      : linear_system_state_(
            VectorXd::Constant(14, std::numeric_limits<double>::quiet_NaN())),
        reference_(
            VectorXd::Constant(14, std::numeric_limits<double>::quiet_NaN())),
        thrust_(9.81),
        thrustdot_(0.0),
        last_control_time_(std::numeric_limits<double>::quiet_NaN()),
        last_linear_system_state_update_time_(
            std::numeric_limits<double>::quiet_NaN()),
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

  // Callbacks for reset linear system state and references.
  void LinearSystemResetCallback(const std_msgs::Empty::ConstPtr& msg);
  void ReferenceCallback(const quads_msgs::OutputDerivatives::ConstPtr& msg);

  // Callback to process new control msgs.
  void ControlCallback(const quads_msgs::Control::ConstPtr& msg);

  // Timer callback and utility to compute Jacobian.
  void TimerCallback(const ros::TimerEvent& e);

  // Are we in flight?
  bool in_flight_;

  // Current linear system state and reference.
  VectorXd linear_system_state_;
  VectorXd reference_;
  double last_linear_system_state_update_time_;
  quads_msgs::OutputDerivatives::ConstPtr last_linear_system_state_estimate_;

  // Polynomial fits.
  PolynomialFit<4, 50> smoother_x_, smoother_y_, smoother_z_;
  PolynomialFit<2, 10> smoother_psi_, smoother_theta_, smoother_phi_;

  // Numerical integration for thrust.
  double thrust_, thrustdot_;
  double last_control_time_;
  quads_msgs::Control::ConstPtr last_control_;

  // Linear feedback controller, and A/B.
  MatrixXd K_, A_, B_;

  // Dynamics.
  Quadrotor14D dynamics_;

  // Most recent msg and time discretization (with timer).
  ros::Timer timer_;
  double dt_;

  // Publishers and subscribers.
  ros::Subscriber in_flight_sub_;
  ros::Subscriber control_sub_;
  ros::Subscriber linear_system_reset_sub_;
  ros::Subscriber reference_sub_;
  ros::Publisher state_pub_;
  ros::Publisher output_derivs_pub_;
  ros::Publisher transitions_pub_;

  std::string output_derivs_topic_;
  std::string in_flight_topic_;
  std::string control_topic_;
  std::string state_topic_;
  std::string transitions_topic_;
  std::string linear_system_reset_topic_;
  std::string reference_topic_;

  // Tf parser.
  TfParser tf_parser_;

  // Initialized flag and name.
  bool initialized_;
  std::string name_;
};  //\class StateEstimator

}  // namespace quads

#endif
