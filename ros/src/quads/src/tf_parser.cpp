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

#include <quads/tf_parser.h>
#include <quads/types.h>
#include <crazyflie_utils/angles.h>

#include <geometry_msgs/TransformStamped.h>
#include <ros/ros.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_listener.h>
#include <Eigen/Dense>
#include <string>

namespace quads {

void TfParser::GetXYZRPY(double* x, double* y, double* z, double* phi,
                         double* theta, double* psi) const {
  geometry_msgs::TransformStamped msg;
  try {
    msg = tf_buffer_.lookupTransform(world_frame_, quad_frame_, ros::Time(0));
  } catch (tf2::TransformException& ex) {
    ROS_ERROR("%s", ex.what());
  }

  *x = msg.transform.translation.x;
  *y = msg.transform.translation.y;
  *z = msg.transform.translation.z;

  // Get roll, pitch, and yaw from quaternion.
  const Eigen::Quaterniond quat(msg.transform.rotation.w,
                                msg.transform.rotation.x,
                                msg.transform.rotation.y,
                                msg.transform.rotation.z);

  // Multiply by sign of x component to ensure quaternion giving the preferred
  // Euler transformation (here we're exploiting the fact that rot(q)=rot(-q) ).
  Eigen::Matrix3d R = quat.toRotationMatrix();
  Vector3d euler = crazyflie_utils::angles::Matrix2RPY(R);
  *phi = crazyflie_utils::angles::WrapAngleRadians(euler(0));
  *theta = crazyflie_utils::angles::WrapAngleRadians(euler(1));
  *psi = crazyflie_utils::angles::WrapAngleRadians(euler(2));

  // Catch nans.
  if (std::isnan(*x) || std::isnan(*y) || std::isnan(*z) || std::isnan(*phi) ||
      std::isnan(*theta) || std::isnan(*psi)) {
    *x = 0.0;
    *y = 0.0;
    *z = 0.0;
    *phi = 0.0;
    *theta = 0.0;
    *psi = 0.0;
  }
}

bool TfParser::Initialize(const ros::NodeHandle& n) {
  name_ = ros::names::append(n.getNamespace(), "tf_parser");

  if (!LoadParameters(n)) {
    ROS_ERROR("%s: Failed to load parameters.", name_.c_str());
    return false;
  }

  return true;
}

bool TfParser::LoadParameters(const ros::NodeHandle& n) {
  ros::NodeHandle nl(n);

  // Frames of reference.
  if (!nl.getParam("frames/world", world_frame_)) return false;
  if (!nl.getParam("frames/quad", quad_frame_)) return false;

  return true;
}

}  // namespace quads
