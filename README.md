docker run -it --rm --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v ~/Robotyka/Master-Thesis/Color_detection_ros_Ubuntu:/workspace -w /workspace master_pan_tilt_env bash


colcon build --packages-select pan_tilt_description --symlink-install
source install/setup.bash
ros2 launch pan_tilt_description simulation.launch.py

