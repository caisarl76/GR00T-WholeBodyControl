# =============================================================================
# ROS2 Configuration
# =============================================================================
# This file contains ROS2 target dependency and test configuration
# Used by both main executable and test executable

if(NOT rclcpp_FOUND OR NOT std_msgs_FOUND)
  message(WARNING "ROS2.cmake included but ROS2 not found")
  return()
endif()

message(STATUS "Configuring ROS2 support...")

# =============================================================================
# ROS2 Target Configuration
# =============================================================================

find_package(ament_cmake REQUIRED)
find_package(Threads REQUIRED)

function(configure_ros2_target target_name)
  if(NOT TARGET ${target_name})
    message(FATAL_ERROR "Cannot configure missing ROS2 target: ${target_name}")
  endif()

  ament_target_dependencies(${target_name} PUBLIC rclcpp std_msgs)
  target_link_libraries(${target_name} PRIVATE Threads::Threads)
  target_compile_definitions(${target_name} PRIVATE HAS_ROS2=1)
endfunction()

# =============================================================================
# ROS2 Test Executable
# =============================================================================

set(TEST_EXECUTABLE_NAME test_ros2)
add_executable(${TEST_EXECUTABLE_NAME} tests/test_ros2.cpp)

target_include_directories(${TEST_EXECUTABLE_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/)
configure_ros2_target(${TEST_EXECUTABLE_NAME})

set_target_properties(${TEST_EXECUTABLE_NAME} PROPERTIES 
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/target/release/"
  OUTPUT_NAME ${TEST_EXECUTABLE_NAME}
)

message(STATUS "✅ ROS2 test executable configured")

# =============================================================================
# Test Configuration Files
# =============================================================================

file(MAKE_DIRECTORY "${PROJECT_SOURCE_DIR}/target/release/config")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/config/fastrtps_profile.xml"
  "${PROJECT_SOURCE_DIR}/target/release/config/fastrtps_profile.xml"
  COPYONLY
)

message(STATUS "✅ FastRTPS profile copied for test deployment")
