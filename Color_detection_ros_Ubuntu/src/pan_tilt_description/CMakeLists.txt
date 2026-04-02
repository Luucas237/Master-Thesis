cmake_minimum_required(VERSION 3.8)
project(pan_tilt_description)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/turret_controller.py
  scripts/vision_tracker.py
  scripts/roi_detection.py
  scripts/roi_detection_simulation.py  # <--- UPEWNIJ SIĘ, ŻE TO TU JEST
  DESTINATION lib/${PROJECT_NAME}
)

# Jedna, właściwa instrukcja instalująca wszystkie foldery
install(DIRECTORY
  launch
  meshes
  urdf
  worlds
  scripts
  DESTINATION share/${PROJECT_NAME}
)

ament_package()