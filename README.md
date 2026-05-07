**SSH Login:**
* Command: `ssh lukasgrab@raspberrypi.local`
* Password: `Rasbian1234`
* ip: `192.168.0.43`

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
