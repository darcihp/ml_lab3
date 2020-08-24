cmake_minimum_required(VERSION 3.0.2)
project(ml_lab3)

find_package(OpenCV 4.3.0 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        roslib
        std_msgs
        cv_bridge
        image_transport
)

catkin_python_setup()

catkin_package(
        CATKIN_DEPENDS roscpp rospy std_msgs
)

include_directories(${catkin_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

catkin_install_python(PROGRAMS scripts/cnn.py
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
