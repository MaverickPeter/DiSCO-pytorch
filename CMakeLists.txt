cmake_minimum_required(VERSION 2.8.3)
project(disco_ros)

find_package(catkin REQUIRED COMPONENTS roscpp rospy message_generation std_msgs geometry_msgs sensor_msgs)

#catkin_python_setup()

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
)

catkin_package(
  
  CATKIN_DEPENDS message_runtime   
)

