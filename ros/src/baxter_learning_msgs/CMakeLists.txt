cmake_minimum_required(VERSION 2.8.3)
project(baxter_learning_msgs)

#if (COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#endif()

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
  geometry_msgs
  quads_msgs
)

file(GLOB msg_files RELATIVE ${PROJECT_SOURCE_DIR}/msg ${PROJECT_SOURCE_DIR}/msg/*.msg)
add_message_files(DIRECTORY msg FILES ${msg_files})

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  quads_msgs
)

catkin_package(
  CATKIN_DEPENDS
    roscpp
    rospy
    std_msgs
    geometry_msgs
    quads_msgs
    message_runtime
)
