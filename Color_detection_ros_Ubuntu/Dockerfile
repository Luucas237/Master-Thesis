FROM osrf/ros:humble-desktop


RUN apt-get update && apt-get install -y \
    ros-humble-cv-bridge \
    python3-opencv \
    ros-humble-rviz2 \
    ros-humble-rviz-common \
    ros-humble-rviz-default-plugins \
    && rm -rf /var/lib/apt/lists/*