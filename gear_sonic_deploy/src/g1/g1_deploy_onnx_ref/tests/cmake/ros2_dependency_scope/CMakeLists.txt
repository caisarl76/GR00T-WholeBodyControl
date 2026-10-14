cmake_minimum_required(VERSION 3.14)

project(ros2_dependency_scope_test LANGUAGES CXX)

find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

# Locate one installed ROS package include directory that neither rclcpp nor
# std_msgs exports.  ROS2.cmake must not add unrelated packages to its targets.
get_filename_component(ROS2_PREFIX "${rclcpp_DIR}/../../.." ABSOLUTE)
file(GLOB INSTALLED_ROS_INCLUDE_DIRS "${ROS2_PREFIX}/include/*")
set(EXPORTED_ROS_INCLUDE_DIRS ${rclcpp_INCLUDE_DIRS} ${std_msgs_INCLUDE_DIRS})
set(UNRELATED_ROS_INCLUDE_DIR "")

foreach(candidate IN LISTS INSTALLED_ROS_INCLUDE_DIRS)
  if(IS_DIRECTORY "${candidate}")
    list(FIND EXPORTED_ROS_INCLUDE_DIRS "${candidate}" exported_index)
    if(exported_index EQUAL -1)
      set(UNRELATED_ROS_INCLUDE_DIR "${candidate}")
      break()
    endif()
  endif()
endforeach()

# Keep ROS2.cmake's generated test artifacts in this fixture's build tree.
set(PROJECT_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/ROS2.cmake")

if(UNRELATED_ROS_INCLUDE_DIR)
  get_target_property(TEST_ROS2_INCLUDE_DIRS test_ros2 INCLUDE_DIRECTORIES)
  list(FIND TEST_ROS2_INCLUDE_DIRS "${UNRELATED_ROS_INCLUDE_DIR}" unrelated_index)
  if(NOT unrelated_index EQUAL -1)
    message(FATAL_ERROR
      "test_ros2 includes unrelated ROS package directory: ${UNRELATED_ROS_INCLUDE_DIR}")
  endif()
else()
  message(STATUS "No unrelated ROS package include directory is installed; scope check skipped")
endif()
