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
// 14D quadrotor dynamics.
//
///////////////////////////////////////////////////////////////////////////////

#include <quads/quadrotor14d.h>
#include <quads/types.h>

#include <ros/ros.h>
#include <Eigen/Dense>
#include <string>
#include <iostream>

namespace quads {

Vector14d Quadrotor14D::operator()(const Vector14d& x,
                                   const Vector4d& u) const {
  ROS_ASSERT(initialized_);

  // Precompute sines/consines.
  const double cpsi = std::cos(x(kPsiIdx));
  const double spsi = std::sin(x(kPsiIdx));
  const double cphi = std::cos(x(kPhiIdx));
  const double sphi = std::sin(x(kPhiIdx));
  const double ctheta = std::cos(x(kThetaIdx));
  const double stheta = std::sin(x(kThetaIdx));

  const double gx = (cpsi * stheta * cphi + spsi * sphi) / m_;
  const double gy = (spsi * stheta * cphi - cpsi * sphi) / m_;
  const double gz = cphi * ctheta / m_;

  Vector14d xdot;
  xdot << x(kDxIdx), x(kDyIdx), x(kDzIdx), x(kQIdx), x(kRIdx), x(kPIdx),
      gx * x(kZetaIdx), gy * x(kZetaIdx), gz * x(kZetaIdx) - 9.81, x(kXiIdx),
      u(0), u(1) / Ix_, u(2) / Iy_, u(3) / Iz_;

  return xdot;
}

Matrix14x14d Quadrotor14D::StateJacobian(const Vector14d& x,
                                         const Vector4d& u) const {
  const double theta = x(kThetaIdx);
  const double psi = x(kPsiIdx);
  const double phi = x(kPhiIdx);
  const double zeta = x(kZetaIdx);

  Matrix14x14d F;
  F << 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
      zeta * std::cos(phi) * std::cos(psi) * std::cos(theta),
      zeta * (std::cos(phi) * std::sin(psi) -
              std::cos(psi) * std::sin(phi) * std::sin(theta)),
      zeta * (std::cos(psi) * std::sin(phi) -
              std::cos(phi) * std::sin(psi) * std::sin(theta)),
      0, 0, 0, std::sin(phi) * std::sin(psi) +
                   std::cos(phi) * std::cos(psi) * std::sin(theta),
      0, 0, 0, 0, 0, 0, 0,
      zeta * std::cos(phi) * std::cos(theta) * std::sin(psi),
      -zeta * (std::cos(phi) * std::cos(psi) +
               std::sin(phi) * std::sin(psi) * std::sin(theta)),
      zeta * (std::sin(phi) * std::sin(psi) +
              std::cos(phi) * std::cos(psi) * std::sin(theta)),
      0, 0, 0, std::cos(phi) * std::sin(psi) * std::sin(theta) -
                   std::cos(psi) * std::sin(phi),
      0, 0, 0, 0, 0, 0, 0, -zeta * std::cos(phi) * std::sin(theta),
      -zeta * std::cos(theta) * std::sin(phi), 0, 0, 0, 0,
      std::cos(phi) * std::cos(theta), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

  return F;
}

Matrix6x14d Quadrotor14D::OutputJacobian(const Vector14d& x) const {
  Matrix6x14d H(Matrix6x14d::Zero());
  H(0, 0) = 1.0;
  H(1, 1) = 1.0;
  H(2, 2) = 1.0;
  H(3, 3) = 1.0;
  H(4, 4) = 1.0;
  H(5, 5) = 1.0;
  return H;
}

bool Quadrotor14D::Initialize(const ros::NodeHandle& n) {
  name_ = ros::names::append(n.getNamespace(), "quadrotor14d");

  if (!LoadParameters(n)) {
    ROS_ERROR("%s: Failed to load parameters.", name_.c_str());
    return false;
  }

  initialized_ = true;
  return true;
}

bool Quadrotor14D::LoadParameters(const ros::NodeHandle& n) {
  ros::NodeHandle nl(n);

  // Mass and inertia.
  if (!nl.getParam("dynamics/m", m_)) return false;
  if (!nl.getParam("dynamics/Ix", Ix_)) return false;
  if (!nl.getParam("dynamics/Iy", Iy_)) return false;
  if (!nl.getParam("dynamics/Iz", Iz_)) return false;

  return true;
}

}  // namespace quads
