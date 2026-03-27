## Initial Setup & Docker Build
> **NOTE:** Run this only once on a new machine to download the repository and build the Docker image.

**Ubuntu ==> Terminal #1**
```bash
cd ~/Robotyka
git clone [https://github.com/Luucas237/Master-Thesis.git](https://github.com/Luucas237/Master-Thesis.git)
cd Master-Thesis/Color_detection_ros_Ubuntu
docker build -t magisterka_ros2 .
```

---

## ===== SIMULATION & CORE START =====
> **NOTE:** This section allows GUI applications (RViz/OpenCV), starts the main Docker container, builds the ROS 2 workspace, and launches the RViz simulation environment.

**Ubuntu ==> Terminal #1**
```bash
xhost +local:root
```

**Ubuntu ==> Terminal #2**
```bash
cd ~/Robotyka/Master-Thesis/Color_detection_ros_Ubuntu

docker run -it --rm --net=host --privileged -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):/workspace -w /workspace --device /dev/video0 magisterka_ros2 bash

colcon build --packages-select pan_tilt_description --symlink-install

source install/setup.bash

ros2 launch pan_tilt_description start.launch.py
```

---

## ===== TRACKING SCRIPTS =====
> **NOTE:** Run these scripts in a new terminal connected to the ALREADY RUNNING Docker container. Choose only ONE script at a time. You can find the `<CONTAINER_NAME>` by running `docker ps` on the host.

**Ubuntu ==> Terminal #3**
```bash
docker ps
docker exec -it <CONTAINER_NAME> bash

cd /workspace
source install/setup.bash
```

**For testing kinematics (Sinusoid movement):**
```bash
ros2 run pan_tilt_description turret_controller.py
```

**For basic color tracking (HSV Color Picker):**
```bash
ros2 run pan_tilt_description vision_tracker.py
```

**For advanced tracking (Predictive ROI + Motion Detection):**
```bash
ros2 run pan_tilt_description roi_detection.py
```
