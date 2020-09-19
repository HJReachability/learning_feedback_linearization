FROM ros:noetic-ros-core

# Updating system and installing dependencies.
RUN apt-get update && apt-get install -y cmake gcc gdb git vim emacs python3-pip
RUN sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
RUN curl -fsSL https://download.sublimetext.com/sublimehq-pub.gpg | sudo apt-key add -
RUN sudo add-apt-repository "deb https://download.sublimetext.com/ apt/stable/"
RUN sudo apt update && sudo apt install sublime-text
RUN pip3 install stable-baselines3

# Create directory structure and clone from git.
RUN git clone https://github.com/HJReachability/learning_feedback_linearization
WORKDIR /learning_feedback_linearization/ros

# Build.
WORKDIR /learning_feedback_linearization

# NOTE: you should run `catkin_make` if you plan to use ROS stuff.
