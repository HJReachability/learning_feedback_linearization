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

#include <quads/types.h>

#include <ros/ros.h>
#include <Eigen/Dense>
#include <string>

namespace quads {

class Quadrotor12D {
 public:
  ~Quadrotor12D() {}
  Quadrotor12D() : m_(1.0), Ix_(1.0), Iy_(1.0), initialized_(false) {}

  // Initialize this class by reading parameters and loading callbacks.
  bool Initialize(const ros::NodeHandle& n);

  // Evaluate state derivative at this state/control.
  Vector12d operator()(const Vector12d& x, const Vector3d& u) const;

  // Jacobians.
  Matrix12x12d StateJacobian(const Vector12d& x, const Vector3d& u) const;
  Matrix5x12d OutputJacobian(const Vector12d& x) const;

  // Static state indices.
  static constexpr size_t kXIdx = 0;
  static constexpr size_t kYIdx = 1;
  static constexpr size_t kZIdx = 2;
  static constexpr size_t kThetaIdx = 3;
  static constexpr size_t kPhiIdx = 4;
  static constexpr size_t kDxIdx = 5;
  static constexpr size_t kDyIdx = 6;
  static constexpr size_t kDzIdx = 7;
  static constexpr size_t kZetaIdx = 8;
  static constexpr size_t kXiIdx = 9;
  static constexpr size_t kQIdx = 10;
  static constexpr size_t kRIdx = 11;

 private:
  // Load parameters.
  bool LoadParameters(const ros::NodeHandle& n);

  // Mass and inertia.
  double m_, Ix_, Iy_;

  // Initialized flag and name.
  bool initialized_;
  std::string name_;
};  //\class OutputSmoother

}  // namespace quads

#endif
