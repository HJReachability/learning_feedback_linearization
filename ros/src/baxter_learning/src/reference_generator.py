#!/usr/bin/env python

# Python Imports
import sys
import numpy as np

# ROS imports
import rospy
from baxter_learning_msgs.msg import State, Reference
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import RobotTrajectory
from std_msgs.msg import Empty

# package imports
from path_planner import PathPlanner

class ReferenceGenerator(object):
    """docstring for ReferenceGenerator"""
    def __init__(self):
        
        if not self.load_parameters(): sys.exit(1)

        self._planner = PathPlanner('{}_arm'.format(self._arm))

        rospy.on_shutdown(self.shutdown)

        if not self.register_callbacks(): sys.exit(1)


    def load_parameters(self):

        if not rospy.has_param("~baxter/arm"):
            return False
        self._arm = rospy.get_param("~baxter/arm")

        if not rospy.has_param("~topics/ref"):
            return False
        self._ref_topic = rospy.get_param("~topics/ref")

        if not rospy.has_param("~topics/linear_system_reset"):
            return False
        self._reset_topic = rospy.get_param("~topics/linear_system_reset")

        if not rospy.has_param("~topics/integration_reset"):
            return False
        self._int_reset_topic = rospy.get_param("~topics/integration_reset")


        return True

    def register_callbacks(self):

        self._reset_sub = rospy.Subscriber(
            self._reset_topic, Empty, self.linear_system_reset_callback)

        self._ref_pub = rospy.Publisher(
            self._ref_topic, Reference)

        self._int_reset_pub = rospy.Publisher(self._int_reset_topic, Empty)

        return True

    def send_zeros(self):
        setpoint = State(np.zeros(7), np.zeros(7))
        feed_forward = np.zeros(7)
        msg = Reference(setpoint, feed_forward)
        while not rospy.is_shutdown():
            self._ref_pub.publish(msg)
            rospy.sleep(0.05)


    def random_span(self, rate = 20, number = 50):
        min_range = np.array([-0.9, -0.6, -0.9, 0.2, -1.0, 0.1, -0.8])
        max_range = np.array([0.9, 0.6, 0.9, 1.0, 1.0, 1.3, 0.8])

        r = rospy.Rate(rate)

        while not rospy.is_shutdown():
            position = (max_range - min_range)*np.random.random_sample(min_range.shape) + min_range
            setpoint = State(position, np.zeros(7))
            feed_forward = np.zeros(7)
            msg = Reference(setpoint, feed_forward)
            count = 0
            while not count > number and not rospy.is_shutdown():
                self._ref_pub.publish(msg)
                count = count + 1
                r.sleep()

    def dual_setpoints(self, rate = 20, number = 50):
        min_range = np.array([-0.9, -0.6, -0.9, 0.2, -1.0, 0.1, -0.8])
        max_range = np.array([0.9, 0.6, 0.9, 1.0, 1.0, 1.3, 0.8])

        

        r = rospy.Rate(rate)

        flag = True
        while not rospy.is_shutdown():
            if flag:
                position = (max_range - min_range)*0.2 + min_range
            else:
                position = (max_range - min_range)*0.8 + min_range
            setpoint = State(position, np.zeros(7))
            feed_forward = np.zeros(7)
            msg = Reference(setpoint, feed_forward)
            count = 0
            while not count > number and not rospy.is_shutdown():
                self._ref_pub.publish(msg)
                count = count + 1
                r.sleep()

            flag = not flag

    


    # Doesn't work
    def sinusoids(self, rate=20, period = 2):
        r = rospy.Rate(rate)

        min_range = np.array([-0.9, -0.6, -0.9, 0.2, -1.0, 0.1, -0.8])
        max_range = np.array([0.9, 0.6, 0.9, 1.0, 1.0, 1.3, 0.8])

        middle = (min_range + max_range) / 2.0
        amp = np.array([0.6, 0.4, 0.6, 0.3, 0.7, 0.3, 0.6])

        count = 0
        max_count = period*rate
        while not rospy.is_shutdown():
            position = middle + amp*np.sin(count/max_count*2*np.pi)
            setpoint = State(position, np.zeros(7))
            feed_forward = np.zeros(7)
            msg = Reference(setpoint, feed_forward)
            self._ref_pub.publish(msg)
            count = count + 1
            if count >= max_count:
                count = 0
            r.sleep()

    # Doesn't work
    def test_path(self):

        position1 = np.array([-0.7, -0.5, -0.7, 0.4, -0.9, 0.3, -0.6])
        position2 = np.array([0.7, 0.5, 0.7, 0.8, 0.9, 1.1, 0.6])

        print "moving to A via moveit"
        self._planner.set_max_velocity_scaling_factor(1.0)
        while not rospy.is_shutdown():
            try:
                plan = self._planner.plan_to_joint_pos(position1)
                # raw_input("Press enter to execute plan via moveit")
                if not self._planner.execute_plan(plan):
                    raise Exception("Execution failed")
            except Exception as e:
                pass
            else:
                break
        rospy.sleep(1)

        print "moving to B via control"
        self._planner.set_max_velocity_scaling_factor(0.5)
        plan = self._planner.plan_to_joint_pos(position2)
        # raw_input("Press enter to execute plan via control")
        self._int_reset_pub.publish(Empty())
        self.execute_path(plan)
        rospy.sleep(1)

        rospy.spin()

    # Doesn't work
    def alternate(self):
        position1 = np.array([-0.3, -0.2, -0.3, 0.7, -0.6, 0.7, -0.3])
        position2 = np.array([-0.6, -0.4, -0.5, 0.6, -0.4, 1.1, -0.5])

        while not rospy.is_shutdown():
            print "moving to A via moveit"
            self._planner.set_max_velocity_scaling_factor(1.0)
            while not rospy.is_shutdown():
                try:
                    plan = self._planner.plan_to_joint_pos(position1)
                    # raw_input("Press enter to execute plan via moveit")
                    if not self._planner.execute_plan(plan):
                        raise Exception("Execution failed")
                except Exception as e:
                    pass
                else:
                    break
            rospy.sleep(1)
            # self._planner.stop()
            rospy.sleep(0.5)

            print "moving to B via control"
            self._planner.set_max_velocity_scaling_factor(0.5)
            plan = self._planner.plan_to_joint_pos(position2)
            # raw_input("Press enter to execute plan via control")
            self._int_reset_pub.publish(Empty())
            self.execute_path(plan)
            rospy.sleep(1)
            # self._planner.stop()
            rospy.sleep(0.5)
            
            print "moving to B via moveit"
            self._planner.set_max_velocity_scaling_factor(1.0)
            while not rospy.is_shutdown():
                try:
                    plan = self._planner.plan_to_joint_pos(position2)
                    # raw_input("Press enter to execute plan via moveit")
                    if not self._planner.execute_plan(plan):
                        raise Exception("Execution failed")
                except Exception as e:
                    pass
                else:
                    break
            rospy.sleep(1)
            # self._planner.stop()
            rospy.sleep(0.5)

            print "moving to A via control"
            self._planner.set_max_velocity_scaling_factor(0.5)
            plan = self._planner.plan_to_joint_pos(position1)
            # raw_input("Press enter to execute plan via control")
            self._int_reset_pub.publish(Empty())
            self.execute_path(plan)
            rospy.sleep(1)
            # self._planner.stop()
            rospy.sleep(0.5)

    def linear_system_reset_callback(self, msg):
        rospy.sleep(0.5)


    def interpolate_path(self, path, t, current_index = 0):
        """
        interpolates over a :obj:`moveit_msgs.msg.RobotTrajectory` to produce desired
        positions, velocities, and accelerations at a specified time

        Parameters
        ----------
        path : :obj:`moveit_msgs.msg.RobotTrajectory`
        t : float
            the time from start
        current_index : int
            waypoint index from which to start search

        Returns
        -------
        target_position : 7x' or 6x' :obj:`numpy.ndarray` 
            desired positions
        target_velocity : 7x' or 6x' :obj:`numpy.ndarray` 
            desired velocities
        target_acceleration : 7x' or 6x' :obj:`numpy.ndarray` 
            desired accelerations
        current_index : int
            waypoint index at which search was terminated 
        """

        # a very small number (should be much smaller than rate)
        epsilon = 0.0001

        max_index = len(path.joint_trajectory.points)-1

        # If the time at current index is greater than the current time,
        # start looking from the beginning
        if (path.joint_trajectory.points[current_index].time_from_start.to_sec() > t):
            current_index = 0

        # Iterate forwards so that you're using the latest time
        while (
            not rospy.is_shutdown() and \
            current_index < max_index and \
            path.joint_trajectory.points[current_index+1].time_from_start.to_sec() < t+epsilon
        ):
            current_index = current_index+1

        # Perform the interpolation
        if current_index < max_index:
            time_low = path.joint_trajectory.points[current_index].time_from_start.to_sec()
            time_high = path.joint_trajectory.points[current_index+1].time_from_start.to_sec()

            target_position_low = np.array(
                path.joint_trajectory.points[current_index].positions
            )
            target_velocity_low = np.array(
                path.joint_trajectory.points[current_index].velocities
            )
            target_acceleration_low = np.array(
                path.joint_trajectory.points[current_index].accelerations
            )

            target_position_high = np.array(
                path.joint_trajectory.points[current_index+1].positions
            )
            target_velocity_high = np.array(
                path.joint_trajectory.points[current_index+1].velocities
            )
            target_acceleration_high = np.array(
                path.joint_trajectory.points[current_index+1].accelerations
            )

            target_position = target_position_low + \
                (t - time_low)/(time_high - time_low)*(target_position_high - target_position_low)
            target_velocity = target_velocity_low + \
                (t - time_low)/(time_high - time_low)*(target_velocity_high - target_velocity_low)
            target_acceleration = target_acceleration_low + \
                (t - time_low)/(time_high - time_low)*(target_acceleration_high - target_acceleration_low)

        # If you're at the last waypoint, no interpolation is needed
        else:
            target_position = np.array(path.joint_trajectory.points[current_index].positions)
            target_velocity = np.array(path.joint_trajectory.points[current_index].velocities)
            target_acceleration = np.array(path.joint_trajectory.points[current_index].velocities)

        return (target_position, target_velocity, target_acceleration, current_index)


    def execute_path(self, path, rate=20):
        """
        takes in a path and moves the baxter in order to follow the path.  

        Parameters
        ----------
        path : :obj:`moveit_msgs.msg.RobotTrajectory`
        rate : int
            This specifies the control frequency in hz.  It is important to
            use a rate and not a regular while loop because you want the
            loop to refresh at a constant rate, otherwise you would have to
            tune your PD parameters if the loop runs slower / faster

        Returns
        -------
        bool
            whether the controller completes the path or not
        """

        # For interpolation
        max_index = len(path.joint_trajectory.points)-1
        current_index = 0

        # For timing
        start_t = rospy.Time.now()
        r = rospy.Rate(rate)

        while not rospy.is_shutdown():
            # Find the time from start
            t = (rospy.Time.now() - start_t).to_sec()

            # Get the desired position, velocity, and acceleration
            (
                target_position, 
                target_velocity, 
                target_acceleration, 
                current_index
            ) = self.interpolate_path(path, t, current_index)

            # Run controller
            setpoint = State(target_position, target_velocity)
            feed_forward = target_acceleration
            msg = Reference(setpoint, feed_forward)
            self._ref_pub.publish(msg)

            # Sleep for a bit (to let robot move)
            r.sleep()

            if current_index >= max_index:
                break

        return True


    def shutdown(self):
        pass

if __name__ == '__main__':
    
    rospy.init_node("reference_generator")

    gen = ReferenceGenerator()

    rospy.sleep(5)

    # gen.send_zeros()
    # gen.alternate()
    gen.random_span() # TESTING
    # gen.test_path()
    # gen.sinusoids()
    # gen.dual_setpoints(number = 100) # TRAINING

    rospy.spin()






        





