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

void ScalarOutputSmoother::Update(double y, double dt) {
  // Discretize time.
  const Matrix4d A_dt = (A_ * dt).exp();

  // Predict step.
  const Vector4d x_predict = A_dt * x_;
  const Matrix4d Px_predict = A_dt * Px_ * A_dt.transpose() + W_ * dt;

  // Update step.
  const double innovation = y - H_ * x_predict;
  const double S = V_ + H_ * Px_predict * H_.transpose();
  const Vector4d K = Px_predict * H_.transpose() / S;

  x_ = x_predict + K * innovation;
  Px_ = (Matrix4d::Identity() - K * H_) * Px_predict;
}

}  // namespace quads
