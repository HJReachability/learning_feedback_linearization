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
// Simulate flight of a 14D quadrotor.
//
///////////////////////////////////////////////////////////////////////////////

#include <crazyflie_msgs/ControlStamped.h>
#include <quads/quadrotor14d.h>
#include <quads/simulator_14d.h>

#include <geometry_msgs/TransformStamped.h>
#include <ros/ros.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <string>

namespace quads {

// Initialize this class by reading parameters and loading callbacks.
bool Simulator14D::Initialize(const ros::NodeHandle& n) {
  name_ = ros::names::append(n.getNamespace(), "simulator_14d");

  // Set state and control to zero initially.
  x_ = Vector14d::Zero();
  x_(dynamics_.kZetaIdx) = 9.81;

  u_ = Vector4d::Zero();

  if (!LoadParameters(n)) {
    ROS_ERROR("%s: Failed to load parameters.", name_.c_str());
    return false;
  }

  if (!RegisterCallbacks(n)) {
    ROS_ERROR("%s: Failed to register callbacks.", name_.c_str());
    return false;
  }

  if (!dynamics_.Initialize(n)) return false;

  // Set initial time.
  last_time_ = ros::Time::now();

  initialized_ = true;
  return true;
}

// Load parameters and register callbacks.
bool Simulator14D::LoadParameters(const ros::NodeHandle& n) {
  ros::NodeHandle nl(n);

  // Frames of reference.
  if (!nl.getParam("frames/fixed", fixed_frame_id_)) return false;
  if (!nl.getParam("frames/robot", robot_frame_id_)) return false;

  // Time step for reading tf.
  if (!nl.getParam("time_step", dt_)) return false;

  // Control topic.
  if (!nl.getParam("topics/control", control_topic_)) return false;

  // Get initial position.
  double init_x, init_y, init_z;
  if (!nl.getParam("init/x", init_x)) return false;
  if (!nl.getParam("init/y", init_y)) return false;
  if (!nl.getParam("init/z", init_z)) return false;

  x_(dynamics_.kXIdx) = init_x;
  x_(dynamics_.kYIdx) = init_y;
  x_(dynamics_.kZIdx) = init_z;

  return true;
}
bool Simulator14D::RegisterCallbacks(const ros::NodeHandle& n) {
  ros::NodeHandle nl(n);

  // Subscribers.
  control_sub_ = nl.subscribe(control_topic_.c_str(), 1,
                              &Simulator14D::ControlCallback, this);

  // Timer.
  timer_ =
      nl.createTimer(ros::Duration(dt_), &Simulator14D::TimerCallback, this);

  return true;
}

// Callback to process new output msg.
void Simulator14D::TimerCallback(const ros::TimerEvent& e) {
  const ros::Time now = ros::Time::now();

  // Only update state if we have received a control signal from outside.
  if (received_control_) {
    x_ += dynamics_(x_, u_) * (now.toSec() - last_time_.toSec());
  }

  std::cout << "state is: \n" << x_ << std::endl;

  // Threshold at ground!
#if 0
  if (x_(dynamics_.kZIdx) < 0.0) {
    x_(dynamics_.kZIdx) = 0.0;
    x_(dynamics_.kDzIdx) = std::max(0.0, x_(dynamics_.kDzIdx));
  }
#endif

  // Update last time.
  last_time_ = now;

  // Broadcast on tf.
  geometry_msgs::TransformStamped transform_stamped;

  transform_stamped.header.frame_id = fixed_frame_id_;
  transform_stamped.header.stamp = now;

  transform_stamped.child_frame_id = robot_frame_id_;

  transform_stamped.transform.translation.x = x_(dynamics_.kXIdx);
  transform_stamped.transform.translation.y = x_(dynamics_.kYIdx);
  transform_stamped.transform.translation.z = x_(dynamics_.kZIdx);

  // RPY to quaternion.
  const double roll = x_(dynamics_.kPhiIdx);
  const double pitch = x_(dynamics_.kThetaIdx);
  const double yaw = x_(dynamics_.kPsiIdx);
  const Eigen::Quaterniond q = Eigen::AngleAxisd(roll, Vector3d::UnitX()) *
                               Eigen::AngleAxisd(pitch, Vector3d::UnitY()) *
                               Eigen::AngleAxisd(yaw, Vector3d::UnitZ());

  /**
  Vector3d euler = q.toRotationMatrix().eulerAngles(0, 1, 2);
  euler(0) = angles::WrapAngleRadians(euler(0));
  euler(1) = angles::WrapAngleRadians(euler(1));
  euler(2) = angles::WrapAngleRadians(euler(2));

  std::cout << "roll = " << u_(0) << ", q roll = " << euler(0) << std::endl;
  std::cout << "pitch = " << u_(1) << ", q pitch = " << euler(1) << std::endl;
  std::cout << "yaw = " << u_(2) << ", q yaw = " << euler(2) << std::endl;
  **/

  transform_stamped.transform.rotation.x = q.x();
  transform_stamped.transform.rotation.y = q.y();
  transform_stamped.transform.rotation.z = q.z();
  transform_stamped.transform.rotation.w = q.w();

  br_.sendTransform(transform_stamped);
}

// Update control signal.
void Simulator14D::ControlCallback(const quads_msgs::Control::ConstPtr& msg) {
  u_(0) = msg->thrustdot2;
  u_(1) = msg->pitchdot2;
  u_(2) = msg->rolldot2;
  u_(3) = msg->yawdot1;
  received_control_ = true;
}

}  // namespace quads
