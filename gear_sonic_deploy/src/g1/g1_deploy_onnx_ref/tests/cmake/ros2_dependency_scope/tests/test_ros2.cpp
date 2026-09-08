#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#if !HAS_ROS2
#error "ROS2 target must enable HAS_ROS2"
#endif

int main() {
  const auto logger = rclcpp::get_logger("ros2_dependency_scope");
  std_msgs::msg::String message;
  message.data = logger.get_name();
  return message.data == "ros2_dependency_scope" ? 0 : 1;
}
