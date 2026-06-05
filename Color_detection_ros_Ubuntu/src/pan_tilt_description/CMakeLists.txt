cmake_minimum_required(VERSION 3.8)
project(pan_tilt_description)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/turret_controller.py
  scripts/vision_tracker.py
  scripts/roi_detection.py
  scripts/roi_detection_simulation.py
  gui/plot.py
  gui/turret_gui.py
  gui/digital_twingui.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  meshes
  urdf
  worlds
  scripts
  gui
  DESTINATION share/${PROJECT_NAME}
)

ament_package()