cmake_minimum_required(VERSION 3.8)
project(pacer_nav2_controller)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(pluginlib REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_util REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pacer_msgs REQUIRED)

set(DEPS
  rclcpp rclcpp_lifecycle pluginlib nav2_core nav2_costmap_2d nav2_util
  tf2 tf2_ros geometry_msgs nav_msgs pacer_msgs
)

add_library(pacer_nav2_controller SHARED
  src/pacer_controller.cpp
)
target_include_directories(pacer_nav2_controller PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(pacer_nav2_controller ${DEPS})

pluginlib_export_plugin_description_file(nav2_core pacer_nav2_controller_plugin.xml)

install(TARGETS pacer_nav2_controller
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/
  DESTINATION include
)
install(FILES pacer_nav2_controller_plugin.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_libraries(pacer_nav2_controller)
ament_export_dependencies(${DEPS})

ament_package()
