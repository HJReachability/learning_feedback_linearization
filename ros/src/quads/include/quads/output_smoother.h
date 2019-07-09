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
// Implementation of Kalman filtering accounting for unknown inputs.
// Please refer to: https://hal.archives-ouvertes.fr/hal-00143941/document
//
///////////////////////////////////////////////////////////////////////////////

#ifndef QUADS_OUTPUT_SMOOTHER_H
#define QUADS_OUTPUT_SMOOTHER_H

#include <quads/polynomial_fit.h>
#include <quads/tf_parser.h>
#include <quads_msgs/Output.h>

#include <geometry_msgs/TransformStamped.h>
#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <string>

namespace quads {

class OutputSmoother {
 public:
  ~OutputSmoother() {}
  OutputSmoother() : initialized_(false) {}

  // Initialize this class by reading parameters and loading callbacks.
  bool Initialize(const ros::NodeHandle& n);

 private:
  // Load parameters and register callbacks.
  bool LoadParameters(const ros::NodeHandle& n);
  bool RegisterCallbacks(const ros::NodeHandle& n);

  // Callback to process new output msg.
  void TimerCallback(const ros::TimerEvent& e);

  // One filter for each output channel since all are decoupled.
  PolynomialFit<4, 25> smoother_x_, smoother_y_, smoother_z_;
  PolynomialFit<3, 25> smoother_psi_;

  // Publishers and subscribers.
  ros::Publisher output_derivs_pub_;
  std::string output_derivs_topic_;

  // Timer and discretization.
  ros::Timer timer_;
  double dt_;

  // TF parser.
  TfParser tf_parser_;

  // Initialized flag and name.
  bool initialized_;
  std::string name_;
};  //\class OutputSmoother

}  // namespace quads

#endif
