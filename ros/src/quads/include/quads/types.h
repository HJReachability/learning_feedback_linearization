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

#ifndef QUADS_TYPES_H
#define QUADS_TYPES_H

#include <Eigen/Dense>

namespace quads {

using Eigen::Matrix4d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Vector5d = Eigen::Matrix<double, 5, 1>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Vector12d = Eigen::Matrix<double, 12, 1>;
using Vector14d = Eigen::Matrix<double, 14, 1>;
using Matrix5x5d = Eigen::Matrix<double, 5, 5>;
using Matrix6x6d = Eigen::Matrix<double, 6, 6>;
using Matrix1x4d = Eigen::Matrix<double, 1, 4>;
using Matrix4x5d = Eigen::Matrix<double, 4, 5>;
using Matrix4x4d = Eigen::Matrix<double, 4, 4>;
using Matrix5x12d = Eigen::Matrix<double, 5, 12>;
using Matrix12x12d = Eigen::Matrix<double, 12, 12>;
using Matrix14x14d = Eigen::Matrix<double, 14, 14>;
using Matrix14x6d = Eigen::Matrix<double, 14, 6>;
using Matrix6x14d = Eigen::Matrix<double, 6, 14>;

}  // namespace quads

#endif
