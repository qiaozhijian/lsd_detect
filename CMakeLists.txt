cmake_minimum_required(VERSION 2.8.3)
project(test_ros)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  tf
  roscpp
  rospy
  cv_bridge
  image_transport

  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
)

find_package(OpenCV REQUIRED QUIET)

FIND_PACKAGE(yaml-cpp REQUIRED CONFIG PATHS ${YAML_PATHS})

catkin_package(
  INCLUDE_DIRS include
)
###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
    include/
  ${catkin_INCLUDE_DIRS}
		${OpenCV_INCLUDE_DIRS}
		${YAML_CPP_INCLUDE_DIRS}
		${PROJECT_SOURCE_DIR}/Thirdparty/line_descriptor/include
)

message(${PROJECT_SOURCE_DIR})

list(APPEND SELF_UTILS
		src/ExtractLineSegment.cpp
		src/config.cpp
		src/gridStructure.cpp
		src/lineIterator.cpp
		src/stereoFeatures.cpp
		src/matching.cpp
		)
  
add_library(self_utils SHARED ${SELF_UTILS} )

target_link_libraries(self_utils
		${EIGEN3_LIBS}
		${OpenCV_LIBS}
		${PROJECT_SOURCE_DIR}/Thirdparty/line_descriptor/lib/liblinedesc.so
		${YAML_CPP_LIBRARIES}
		)

add_executable(lsd_detect_node main.cpp)
target_link_libraries(lsd_detect_node ${catkin_LIBRARIES} self_utils ${OpenCV_LIBRARIES}  )
