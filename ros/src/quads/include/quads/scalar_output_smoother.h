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

#ifndef QUADS_SCALAR_OUTPUT_SMOOTHER_H
#define QUADS_SCALAR_OUTPUT_SMOOTHER_H

#include <quads/types.h>

namespace quads {

class ScalarOutputSmoother {
 public:
  ~ScalarOutputSmoother() {}
  ScalarOutputSmoother()
      : x_(Vector4d::Zero()), Px_(100.0 * Matrix4d::Identity()) {
    F_ = Vector4d::Zero();
    F_(3) = 1.0;

    E_ = Matrix45d::Zero();
    E_(0, 0) = 1.0;
    E_(1, 1) = 1.0;
    E_(2, 2) = 1.0;
    E_(3, 3) = 1.0;
    E_(3, 4) = -1.0;

    A_ = Matrix4d::Zero();
    A_(0, 1) = 1.0;
    A_(1, 2) = 1.0;
    A_(2, 3) = 1.0;

    H_ = Matrix14d::Zero();
    H_(0, 0) = 1.0;
  }

  // Update with a new measurement and time delta.
  void Update(double y, double dt);

  // Accessors.
  double X() const { return x_(0); }
  double XDot1() const { return x_(1); }
  double XDot2() const { return x_(2); }
  double XDot3() const { return x_(3); }
  const Matrix4d& Cov() const { return Px_; }

 private:
  // Mean and covariance estimates for this output channel.
  Vector4d x_;
  Matrix4d Px_;

  // Utilities.
  Vector4d F_;
  Matrix45d E_;
  Matrix14d H_;
  Matrix4d A_;
};  //\class ScalarOutputSmoother

}  // namespace quads

#endif
