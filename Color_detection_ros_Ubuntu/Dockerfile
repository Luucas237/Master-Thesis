FROM osrf/ros:humble-desktop

RUN apt-get update && apt-get install -y \
    ros-humble-cv-bridge \
    python3-opencv \
    ros-humble-foxglove-bridge \
    ros-humble-rviz2 \
    ros-humble-rviz-common \
    ros-humble-rviz-default-plugins \
    ros-humble-xacro \
    ros-humble-ros-gz \
    \
    python3-pyqt5 \
    python3-numpy \
    \
    ros-humble-image-transport \
    ros-humble-image-transport-plugins \
    \
    ros-humble-rqt \
    ros-humble-rqt-common-plugins \
    ros-humble-rqt-image-view \
    \
    python3-pip \
    python3-serial \
    nano \
    \
    alsa-utils \
    mpg123 \
    gstreamer1.0-plugins-ugly \
    libsdl2-mixer-2.0-0 \
    && rm -rf /var/lib/apt/lists/*


RUN pip3 install pyzmq pygame pandas matplotlib "numpy<2"