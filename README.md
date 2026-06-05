**SSH Login:**
* Command: `ssh lukasgrab@raspberrypi.local`
* Password: `Rasbian1234`
* ip: `192.168.0.43`

```bash
xhost +local:root

docker run -it --rm --net=host --device /dev/snd -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v ~/Robotyka/Master-Thesis/Color_detection_ros_Ubuntu:/workspace -w /workspace master_pan_tilt_env bash

colcon build --packages-select pan_tilt_description --symlink-install

source install/setup.bash

ros2 run pan_tilt_description turret_gui.py
```

## ===== Launching =====
**Rasbian **
```bash
while true; do rpicam-vid -n -t 0 --width 640 --height 480 --framerate 30 --codec mjpeg --listen -o tcp://0.0.0.0:5000; sleep 1; done

rpicam-vid -t 0 --width 640 --height 480 --framerate 30 --codec mjpeg --inline -o udp://192.168.0.255:5000
```
**Rasbian ssh ubuntu**
```bash
cd /home/lukasgrab/Turret
libcamerify python3 tracker_guiless.py
python3 esp_test.py
python3 tracker.py
```
**Ubuntu**
```bash
mpv tcp://192.168.0.43:5000 --profile=low-latency --untimed --demuxer=lavf --demuxer-lavf-format=mjpeg
```
