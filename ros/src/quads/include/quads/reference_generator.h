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
// Generate an infinite-horizon reference signal for the linear system to track.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef QUADS_REFERENCE_GENERATOR_H
#define QUADS_REFERENCE_GENERATOR_H

#include <quads/quadrotor14d.h>

#include <quads_msgs/OutputDerivatives.h>

#include <ros/ros.h>
#include <std_msgs/Empty.h>

namespace quads {

class ReferenceGenerator {
 public:
  ~ReferenceGenerator() {}
  ReferenceGenerator() : in_flight_(false), initialized_(false) {}

  // Initialize this class by reading parameters and loading callbacks.
  bool Initialize(const ros::NodeHandle& n);

 private:
  // Load parameters and register callbacks.
  bool LoadParameters(const ros::NodeHandle& n);
  bool RegisterCallbacks(const ros::NodeHandle& n);

  // Timer callback.
  void TimerCallback(const ros::TimerEvent& e);

  // In flight?
  void InFlightCallback(const std_msgs::Empty::ConstPtr& msg) {
    in_flight_ = true;
    in_flight_time_ = ros::Time::now().toSec();
  }

  // Are we in flight?
  bool in_flight_;
  double in_flight_time_;

  // Publishers and subscribers.
  ros::Subscriber in_flight_sub_;
  ros::Publisher reference_pub_;

  std::string in_flight_topic_;
  std::string reference_topic_;

  // Timer stuff.
  ros::Timer timer_;
  double dt_;

  // Frequency of sinusoids.
  double x_freq_, y_freq_, z_freq_, psi_freq_;

  // Initialized flag and name.
  bool initialized_;
  std::string name_;
};  //\class ReferenceGenerator

}  // namespace quads

#endif
