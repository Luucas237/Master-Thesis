**SSH Login:**
* Command: `ssh lukasgrab@raspberrypi.local`
* Password: `Rasbian1234`
* ip: `192.168.0.43`

```bash
docker run -it --rm --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v ~/Robotyka/Master-Thesis/Color_detection_ros_Ubuntu:/workspace -w /workspace master_pan_tilt_env bash

colcon build --packages-select pan_tilt_description --symlink-install

source install/setup.bash

ros2 run pan_tilt_description turret_gui.py
```

## ===== Launching =====
**Rasbian **
```bash
rpicam-vid -n -t 0 --width 1280 --height 720 --framerate 30 --codec h264 --inline --listen -o tcp://0.0.0.0:5000  
```
**Rasbian ssh ubuntu**
```bash
cd /home/lukasgrab/Turret
python3 esp_test.py
```
**Ubuntu**
```bash
mpv tcp://192.168.0.43:5000 --profile=low-latency --fps=30 --untimed
```
