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
// 12D quadrotor dynamics.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef QUADS_QUADROTOR_12D_H
#define QUADS_QUADROTOR_12D_H

#include <ros/ros.h>
#include <Eigen/Dense>
#include <string>

namespace quads {

using Eigen::Vector3d;
using Eigen::Vector5d = Eigen::Matrix<double, 5, 1>;
using Vector12d = Eigen::Matrix<double, 12, 1>;
using Matrix5d = Eigen::Matrix<double, 5, 5>;
using Matrix12d = Eigen::Matrix<double, 12, 12>;

class Quadrotor12D {
 public:
  // Initialize this class by reading parameters and loading callbacks.
  bool Initialize(const ros::NodeHandle& n);

  // Evaluate state derivative at this state/control.
  Vector12d operator()(const Vector12d& x, const Vector3d& u) const;

  // Jacobians.
  Matrix12d StateJacobian(const Vector12d& x, const Vector3d& u) const;
  Matrix5d OutputJacobian(const Vector12d& x) const;

 private:
  Quadrotor12D() : initialized_(false) {}

  // Load parameters.
  bool LoadParameters(const ros::NodeHandle& n);

  // Initialized flag and name.
  bool initialized_;
  std::string name_;
};  //\class OutputSmoother

}  // namespace quads

#endif
