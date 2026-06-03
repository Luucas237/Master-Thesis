#!/usr/bin/env python3
import cv2
import numpy as np
import serial
import time
import threading
import zmq
import csv
import datetime

class SimplePID:
    def __init__(self, kp=0.01, ki=0.0, kd=0.001):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error, dt):
        if dt <= 0: return 0.0
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

    def set_params(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0

class FastTurretTracker:
    def __init__(self):
        self.context = zmq.Context()
        self.cmd_socket = self.context.socket(zmq.SUB)
        self.cmd_socket.bind("tcp://*:5556")
        self.cmd_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.img_socket = self.context.socket(zmq.PUB)
        self.img_socket.bind("tcp://*:5555")

        try:
            self.ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.05)
            time.sleep(2)
        except Exception:
            self.ser = None

        self.latest_frame = None
        self.running = True
        self.cap = cv2.VideoCapture("tcp://127.0.0.1:5000")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        threading.Thread(target=self.grab_frames, daemon=True).start()

        self.mode = "AUTO"
        self.state = "SEARCHING"
        self.target_hsv = np.array([0, 220, 180], dtype=np.uint8)
        self.pid_pan = SimplePID(kp=0.01, ki=0.0, kd=0.001)
        self.pid_tilt = SimplePID(kp=0.01, ki=0.0, kd=0.001)
        self.pan_enabled = True
        self.tilt_enabled = True
        self.current_pan = 0.0
        self.current_tilt = 0.0
        self.laser_state = 0
        self.last_sent_pan = -999.0
        self.last_sent_tilt = -999.0
        self.last_sent_laser = -1
        self.prev_gray = None
        self.roi_rect = None
        self.lost_time = 0.0
        self.logging_started = False
        self.log_file = None
        self.csv_writer = None
        self.start_log_time = 0.0
        self.reset_kalman()
        self.frame_count = 0
        self.fps_timer = time.time()
        self.fps_filtered = 0.0
        self.last_time = time.time()
        print("Tracker initialized and ready.")

    def reset_kalman(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

    def grab_frames(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret: self.latest_frame = frame
            else: time.sleep(0.01)

    def process_zmq_commands(self):
        try:
            while True:
                msg = self.cmd_socket.recv_multipart(flags=zmq.NOBLOCK)
                topic = msg[0].decode('utf-8')
                data = msg[1].decode('utf-8')

                if topic == "MODE":
                    self.mode = data
                    print(f"Mode changed: {self.mode}")
                elif topic == "PID":
                    vals = list(map(float, data.split()))
                    if len(vals) == 6:
                        self.pid_pan.set_params(vals[0], vals[1], vals[2])
                        self.pid_tilt.set_params(vals[3], vals[4], vals[5])
                        print(f"PID Updated -> PAN: {vals[0]}/{vals[1]}/{vals[2]} | TILT: {vals[3]}/{vals[4]}/{vals[5]}")
                elif topic == "AXIS":
                    vals = list(map(int, data.split()))
                    if len(vals) == 2:
                        self.pan_enabled = bool(vals[0])
                        self.tilt_enabled = bool(vals[1])
                        print(f"Axes Status -> PAN Enabled: {self.pan_enabled} | TILT Enabled: {self.tilt_enabled}")
                elif topic == "COLOR":
                    h, s, v = map(int, data.split())
                    self.target_hsv = np.array([h, s, v], dtype=np.uint8)
                    print(f"Target color updated: {h}, {s}, {v}")
                elif topic == "MANUAL":
                    if self.mode == "MANUAL":
                        pan, tilt, laser = map(float, data.split())
                        self.current_pan = pan
                        self.current_tilt = tilt
                        self.laser_state = int(laser)
                elif topic == "LASER":
                    if self.mode == "MANUAL":
                        self.laser_state = int(data)
        except zmq.Again:
            pass

    def run(self):
        DEADBAND = 25
        morph_kernel = np.ones((9, 9), np.uint8)
        print("Main tracking loop running...")

        while self.running:
            self.process_zmq_commands()

            if self.latest_frame is None:
                time.sleep(0.01)
                continue

            frame = self.latest_frame.copy()
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            roi_w, roi_h = int(w * 0.4), int(h * 0.4)
            now = time.time()
            dt = now - self.last_time
            self.last_time = now

            mask_display = np.zeros((h, w), dtype=np.uint8)
            error_x = 0
            error_y = 0

            if self.mode == "AUTO":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                if self.state == "SEARCHING":
                    self.laser_state = 0
                    if self.prev_gray is not None:
                        diff = cv2.absdiff(self.prev_gray, gray)
                        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            c = max(contours, key=cv2.contourArea)
                            if cv2.contourArea(c) > 500:
                                M = cv2.moments(c)
                                if M["m00"] > 0:
                                    mx, my = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                                    x1 = max(0, mx - roi_w // 2)
                                    y1 = max(0, my - roi_h // 2)
                                    x2 = min(w, x1 + roi_w)
                                    y2 = min(h, y1 + roi_h)
                                    self.roi_rect = [x1, y1, x2, y2]
                                    self.state = "VERIFYING"
                                    self.reset_kalman()
                    self.prev_gray = gray

                elif self.state in ["VERIFYING", "TRACKING", "PREDICTING"]:
                    x1, y1, x2, y2 = self.roi_rect
                    if x1 >= x2 or y1 >= y2:
                        self.state = "SEARCHING"
                        self.prev_gray = None
                        continue

                    roi_frame = frame[y1:y2, x1:x2]
                    hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
                    lower = np.array([max(0, self.target_hsv[0]-15), 100, 100])
                    upper = np.array([min(179, self.target_hsv[0]+15), 255, 255])
                    mask_roi = cv2.inRange(hsv_roi, lower, upper)
                    mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, morph_kernel)
                    mask_display[y1:y2, x1:x2] = mask_roi
                    contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

                    target_found = False
                    if contours:
                        c = max(contours, key=cv2.contourArea)
                        if cv2.contourArea(c) > 300:
                            M = cv2.moments(c)
                            if M["m00"] > 0:
                                cx = int(M["m10"] / M["m00"]) + x1
                                cy = int(M["m01"] / M["m00"]) + y1
                                target_found = True

                    if target_found:
                        if not self.logging_started:
                            self.logging_started = True
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            self.filename = f"pid_log_{timestamp}.csv"
                            self.log_file = open(self.filename, mode='w', newline='')
                            self.csv_writer = csv.writer(self.log_file)
                            self.csv_writer.writerow(['Czas_s', 'Uchyb_X_px', 'Uchyb_Y_px', 'Kp_Pan', 'Kd_Pan', 'Kp_Tilt', 'Kd_Tilt'])
                            self.start_log_time = time.time()
                            print(f"Started continuous logging to {self.filename}")

                        self.state = "TRACKING"
                        self.laser_state = 1
                        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
                        self.kalman.correct(measurement)
                        error_x = center_x - cx
                        error_y = center_y - cy
                        if self.pan_enabled:
                            if abs(error_x) >= DEADBAND:
                                self.current_pan += self.pid_pan.compute(error_x, dt) * -1.0
                            else:
                                self.pid_pan.prev_error = error_x
                        if self.tilt_enabled:
                            if abs(error_y) >= DEADBAND:
                                self.current_tilt += self.pid_tilt.compute(error_y, dt)
                            else:
                                self.pid_tilt.prev_error = error_y
                        self.roi_rect = [max(0, cx - roi_w//2), max(0, cy - roi_h//2),
                                         min(w, cx + roi_w//2), min(h, cy + roi_h//2)]
                        cv2.circle(frame, (cx, cy), 15, (0, 255, 0), 2)

                    else:
                        if self.state == "VERIFYING":
                            self.state = "SEARCHING"
                        else:
                            if self.state == "TRACKING":
                                self.state = "PREDICTING"
                                self.lost_time = time.time()
                                self.laser_state = 0

                            if time.time() - self.lost_time < 2.0:
                                prediction = self.kalman.predict()
                                px, py = int(prediction[0]), int(prediction[1])
                                error_x = center_x - px
                                error_y = center_y - py
                                if self.pan_enabled:
                                    if abs(error_x) >= DEADBAND:
                                        self.current_pan += self.pid_pan.compute(error_x, dt) * -1.0
                                    else:
                                        self.pid_pan.prev_error = error_x
                                if self.tilt_enabled:
                                    if abs(error_y) >= DEADBAND:
                                        self.current_tilt += self.pid_tilt.compute(error_y, dt)
                                    else:
                                        self.pid_tilt.prev_error = error_y
                                cv2.circle(frame, (px, py), 15, (0, 0, 255), 2)
                                self.roi_rect = [max(0, px - roi_w//2), max(0, py - roi_h//2),
                                                 min(w, px + roi_w//2), min(h, py + roi_h//2)]
                            else:
                                self.current_pan = 0.0
                                self.current_tilt = 0.0
                                self.laser_state = 0
                                self.state = "SEARCHING"
                                self.prev_gray = None
                                self.reset_kalman()

                if self.logging_started and self.csv_writer:
                    current_log_time = time.time() - self.start_log_time
                    self.csv_writer.writerow([
                        f"{current_log_time:.3f}", error_x, error_y,
                        self.pid_pan.kp, self.pid_pan.kd, self.pid_tilt.kp, self.pid_tilt.kd
                    ])

                self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.prev_gray = cv2.GaussianBlur(self.prev_gray, (21, 21), 0)

            elif self.mode == "MANUAL":
                self.state = "SEARCHING"
                self.prev_gray = None

            self.current_tilt = max(-25.0, min(25.0, self.current_tilt))
            if self.ser:
                if abs(self.current_pan - self.last_sent_pan) >= 0.3 or \
                   abs(self.current_tilt - self.last_sent_tilt) >= 0.3 or \
                   self.laser_state != self.last_sent_laser:
                    cmd = f"{self.current_pan:.2f} {self.current_tilt:.2f} {self.laser_state}\n"
                    self.ser.write(cmd.encode('utf-8'))
                    self.last_sent_pan = self.current_pan
                    self.last_sent_tilt = self.current_tilt
                    self.last_sent_laser = self.laser_state

            ret_rgb, buffer_rgb = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            ret_mask, buffer_mask = cv2.imencode('.jpg', mask_display, [cv2.IMWRITE_JPEG_QUALITY, 50])
            if ret_rgb and ret_mask:
                self.img_socket.send_multipart([b"RGB", buffer_rgb.tobytes()])
                self.img_socket.send_multipart([b"MASK", buffer_mask.tobytes()])
            self.img_socket.send_multipart([b"LASER_STATE", str(self.laser_state).encode('utf-8')])

            self.frame_count += 1
            elapsed = time.time() - self.fps_timer
            if elapsed > 1.0:
                self.fps_filtered = self.frame_count / elapsed
                self.img_socket.send_multipart([b"FPS", str(int(self.fps_filtered)).encode('utf-8')])
                self.frame_count = 0
                self.fps_timer = time.time()

if __name__ == '__main__':
    tracker = FastTurretTracker()
    try:
        tracker.run()
    except KeyboardInterrupt:
        if tracker.log_file:
            tracker.log_file.close()
            print("Execution stopped. Log file safely closed.")
        pass