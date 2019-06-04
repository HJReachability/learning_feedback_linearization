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

#include <quads/quadrotor12d.h>

#include <math.h>
#include <ros/ros.h>
#include <Eigen/Dense>
#include <string>

namespace quads {

Vector12d Quadrotor12D::operator()(const Vector12d& x,
                                   const Vector3d& u) const {
  ROS_ASSERT(initialized_);

  const double gx = std::sin(x(kThetaIdx)) * std::cos(x(kPhiIdx)) / m_;
  const double gy = -std::sin(x(kPhiIdx)) / m_;
  const double gz = std::cos(x(kPhiIdx)) * std::cos(x(kThetaIdx)) / m_;

  Vector12d xdot;
  xdot << x(kDxIdx), x(kDyIdx), x(kDzIdx), x(kQIdx), x(kRIdx), gx * x(kZetaIdx),
      gy * x(kZetaIdx), gz * x(kZetaIdx) - 9.81, x(kXiIdx), u(0), u(1) / Ix_,
      u(2) / Iy_;
  return xdot;
}

Matrix12x12d Quadrotor12D::StateJacobian(const Vector12d& x,
                                         const Vector3d& u) const {
  const double theta = x(kThetaIdx);
  const double phi = x(kPhiIdx);
  const double zeta = x(kZetaIdx);

  Matrix12x12d F;
  F << 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
      zeta * std::cos(phi) * std::cos(theta),
      -zeta * std::sin(phi) * std::sin(theta), 0, 0, 0,
      std::cos(phi) * std::sin(theta), 0, 0, 0, 0, 0, 0, 0,
      -zeta * std::cos(phi), 0, 0, 0, -std::sin(phi), 0, 0, 0, 0, 0, 0,
      -zeta * std::cos(phi) * std::sin(theta),
      -zeta * std::cos(theta) * std::sin(phi), 0, 0, 0,
      std::cos(phi) * std::cos(theta), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  return F;
}

Matrix5x12d Quadrotor12D::OutputJacobian(const Vector12d& x) const {
  Matrix5x12d H(Matrix5x12d::Zero());
  H(0, 0) = 1.0;
  H(1, 1) = 1.0;
  H(2, 2) = 1.0;
  H(3, 3) = 1.0;
  H(4, 4) = 1.0;
  return H;
}

bool Quadrotor12D::Initialize(const ros::NodeHandle& n) {
  name_ = ros::names::append(n.getNamespace(), "quadrotor12d");

  if (!LoadParameters(n)) {
    ROS_ERROR("%s: Failed to load parameters.", name_.c_str());
    return false;
  }

  return true;
}

bool Quadrotor12D::LoadParameters(const ros::NodeHandle& n) {
  ros::NodeHandle nl(n);

  // Mass and inertia.
  if (!nl.getParam("dynamics/m", m_)) return false;
  if (!nl.getParam("dynamics/Ix", Ix_)) return false;
  if (!nl.getParam("dynamics/Iy", Iy_)) return false;

  return true;
}

}  // namespace quads
