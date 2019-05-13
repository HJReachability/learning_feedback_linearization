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
// Differentiate and smooth the outputs.
//
///////////////////////////////////////////////////////////////////////////////

#include <quads/output_smoother.h>

#include <math.h>
#include <ros/ros.h>
#include <string>

namespace quads {

bool OutputSmoother::Initialize(const ros::NodeHandle& n) {
  name_ = ros::names::append(n.getNamespace(), "output_smoother");

  if (!LoadParameters(n)) {
    ROS_ERROR("%s: Failed to load parameters.", name_.c_str());
    return false;
  }

  if (!RegisterCallbacks(n)) {
    ROS_ERROR("%s: Failed to register callbacks.", name_.c_str());
    return false;
  }

  return true;
}

bool OutputSmoother::LoadParameters(const ros::NodeHandle& n) {
  ros::NodeHandle nl(n);

  // Topics.
  if (!nl.getParam("topics/output", output_topic_)) return false;
  if (!nl.getParam("topics/output_derivs", output_derivs_topic_)) return false;

  return true;
}

bool OutputSmoother::RegisterCallbacks(const ros::NodeHandle& n) {
  ros::NodeHandle nl(n);

  // Subscriber.
  output_sub_ = nl.subscribe(output_topic_.c_str(), 1,
                             &OutputSmoother::OutputCallback, this);

  // Publisher.
  output_derivs_pub_ = nl.advertise<quads_msgs::OutputDerivatives>(
      output_derivs_topic_.c_str(), 1, false);

  return true;
}

void OutputSmoother::OutputCallback(const quads_msgs::Output::ConstPtr& msg) {
  constexpr size_t kMaxHistorySize = 10;

  // Insert this msg into past outputs, and maybe dump an old one.
  old_outputs_.push_back(msg);
  if (old_outputs_.size() > kMaxHistorySize)
    old_outputs_.pop_front();

  //
}

}  // namespace quads
