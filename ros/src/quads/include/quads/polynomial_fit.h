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
// Fits a polynomial to the past 'k' scalar data points.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef QUADS_POLYNOMIAL_FIT_H
#define QUADS_POLYNOMIAL_FIT_H

#include <math.h>
#include <ros/ros.h>
#include <Eigen/Dense>
#include <list>

namespace quads {

namespace {
size_t factorial(size_t ii) {
  if (ii == 0 || ii == 1) return 1;
  return ii * factorial(ii - 1);
}

size_t factorial_fraction(size_t ii, size_t jj) {
  if (ii == jj) return 1;
  return ii * factorial_fraction(ii - 1, jj);
}
}

template <size_t k, size_t n>
class PolynomialFit {
 public:
  ~PolynomialFit() {}
  PolynomialFit() {}

  // Update with a new time-stamp and observation.
  void Update(double x, double t);

  // Interpolate at a specific time.
  double Interpolate(double t, size_t num_derivatives);

 private:
  // Data and time.
  std::list<double> xs_;
  std::list<double> ts_;

  // Polynomial coefficients.
  Eigen::Matrix<double, k + 1, 1> coeffs_;
};  //\class PolynomialFit

// -------------------------- Implementation ---------------------------- //

template <size_t k, size_t n>
void PolynomialFit<k, n>::Update(double x, double t) {
  xs_.push_back(x);
  ts_.push_back(t);

  // Handle case where we haven't yet seen 'n' data points.
  if (xs_.size() < n) return;

  // We've seen enough, so throw away old ones.
  if (xs_.size() > n) {
    ROS_ASSERT(xs_.size() == n + 1);
    xs_.pop_front();
    ts_.pop_front();
  }

  // Set up least squares problem.
  Eigen::Matrix<double, n, k + 1> A;
  Eigen::Matrix<double, n, 1> b;

  auto xs_iter = xs_.begin();
  auto ts_iter = ts_.begin();

  for (size_t ii = 0; ii < n; ii++, xs_iter++, ts_iter++) {
    b(ii) = *xs_iter;

    A(ii, 0) = 1.0;
    for (size_t jj = 1; jj < k + 1; jj++)
      A(ii, jj) = A(ii, jj - 1) * (*ts_iter - ts_.front());
  }

  // Solve least squares.
  coeffs_ = A.householderQr().solve(b);
}

template <size_t k, size_t n>
double PolynomialFit<k, n>::Interpolate(double t, size_t num_derivatives) {
  if (num_derivatives > k) return 0.0;

  if (xs_.size() != n) {
    ROS_WARN("Trying to interpolate a polynomial too soon.");
    return 0.0;
  }

  double time_power = 1.0;
  double total = 0.0;
  for (size_t ii = num_derivatives; ii < k + 1; ii++) {
    total += static_cast<double>(factorial(ii)) * time_power * coeffs_(ii) /
             static_cast<double>(factorial(ii - num_derivatives));
    time_power *= (t - ts_.front());
  }

  return total;
}

}  // namespace quads

#endif
