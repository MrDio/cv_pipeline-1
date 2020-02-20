cmake_minimum_required(VERSION 2.8.3)
project(cv_pipeline)

## Compile with C++11
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs sensor_msgs message_generation)

## Declaring ROS Services
add_service_files(FILES gqcnnpj.srv gqcnnsuction.srv fcgqcnnpj.srv fcgqcnnsuction.srv maskrcnn.srv)

## Generate added messages and services with any dependencies listed here
generate_messages(DEPENDENCIES std_msgs sensor_msgs)

## Declaring a Catkin package
catkin_package(CATKIN_DEPENDS message_runtime)
