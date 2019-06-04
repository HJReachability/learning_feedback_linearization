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

#include <quads/scalar_output_smoother.h>
#include <quads/types.h>


#include <Eigen/QR>
#include <unsupported/Eigen/MatrixFunctions>
#include <iostream>

namespace quads {

namespace {
// Process and observation noise.
static const Matrix4d W = 0.001 * Matrix4d::Identity();
static const double V = 0.0005;
}  // anonymous namespace

void ScalarOutputSmoother::Update(double y, double dt) {
  // Discretize time.
  const Matrix4d A_dt = (A_ * dt).exp();
  const Vector4d F_dt = (0.5 * dt * (A_dt + Matrix4d::Identity())) * F_;

  std::cout << "A_dt\n" << A_dt << std::endl;
  std::cout << "F_dt\n" << F_dt << std::endl;

  // Following pp. 8 (leadup to Thm. 4.1) of reference above.
  const Matrix4d bar_P = A_dt * Px_ * A_dt.transpose() + W;
  const Vector4d bar_x = A_dt * x_;

  const Matrix4d bar_P_inv = bar_P.inverse();
  //  std::cout << "bar_P_inv\n" << bar_P_inv << std::endl;

  Matrix5d cal_P_inv;
  cal_P_inv.block<4, 4>(0, 0) = bar_P_inv + H_.transpose() * H_ / V;
  cal_P_inv.block<4, 1>(0, 4) = -bar_P_inv * F_dt;
  cal_P_inv.block<1, 4>(4, 0) = -F_dt.transpose() * bar_P_inv;
  cal_P_inv(4, 4) = F_dt.transpose() * bar_P_inv * F_dt;
  const Matrix5d cal_P = cal_P_inv.inverse();
  //  std::cout << "cal_P\n" << cal_P << std::endl;
  //  std::cout << "cal_P_inv\n" << cal_P_inv << std::endl;

  Vector5d cal_H = Vector5d::Zero();
  cal_H.head<4>() = H_.transpose() / V;
  const Vector5d cal_X =
      cal_P * E_.transpose() * bar_P_inv * bar_x + cal_P * cal_H * y;

  //  std::cout << "cal_X\n" << cal_X << std::endl;

  // Unpack cal_X and cal_P into x_ and Px_;
  x_ = cal_X.head<4>();
  Px_ = cal_P.block<4, 4>(0, 0);
}

}  // namespace quads
