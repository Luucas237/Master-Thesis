FROM osrf/ros:humble-desktop

RUN apt-get update && apt-get install -y \
    # --- TWOJE OBECNE PAKIETY ---
    ros-humble-cv-bridge \
    python3-opencv \
    ros-humble-foxglove-bridge \
    ros-humble-rviz2 \
    ros-humble-rviz-common \
    ros-humble-rviz-default-plugins \
    ros-humble-xacro \
    ros-humble-ros-gz \
    \
    # --- NOWE: Aplikacja GUI (Centrum Dowodzenia) ---
    python3-pyqt5 \
    python3-numpy \
    \
    # --- NOWE: Przepływ obrazu przez Wi-Fi (Kompresja) ---
    ros-humble-image-transport \
    ros-humble-image-transport-plugins \
    \
    # --- NOWE: Potężne debugowanie wykresów i zmiennych ---
    ros-humble-rqt \
    ros-humble-rqt-common-plugins \
    ros-humble-rqt-image-view \
    \
    # --- NOWE: Narzędzia systemowe ---
    python3-pip \
    python3-serial \
    nano \
    && rm -rf /var/lib/apt/lists/*