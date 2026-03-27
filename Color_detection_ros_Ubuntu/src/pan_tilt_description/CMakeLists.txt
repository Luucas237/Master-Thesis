cmake_minimum_required(VERSION 3.8)
project(pan_tilt_description)

# Szukamy narzędzi do budowania
find_package(ament_cmake REQUIRED)

# ODBLOKOWANIE FOLDERÓW (To, co mieliśmy dopisać)
install(DIRECTORY meshes urdf launch
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/turret_controller.py
  scripts/vision_tracker.py
  scripts/roi_detection.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()