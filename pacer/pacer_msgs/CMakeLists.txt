cmake_minimum_required(VERSION 3.8)
project(pacer_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PedestrianTrack.msg"
  "msg/PedestrianTrackArray.msg"
  "msg/PedestrianPrediction.msg"
  "msg/PedestrianPredictionArray.msg"
  "msg/QTupacSnapshot.msg"
  "msg/StaticObstacle.msg"
  "srv/TrainPredictor.srv"
  "srv/RecalibrateTupac.srv"
  "srv/ComputeMpcCommand.srv"
  DEPENDENCIES std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
