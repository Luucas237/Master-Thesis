#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import random

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, error, dt):
        if dt <= 0.0: return 0.0
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

class SimRoiTracker(Node):
    def __init__(self):
        super().__init__('sim_roi_tracker')
        
        self.pub_pan = self.create_publisher(Float64, '/model/pan_tilt_turret/joint/pan_joint/cmd_pos', 10)
        self.pub_tilt = self.create_publisher(Float64, '/model/pan_tilt_turret/joint/tilt_joint/cmd_pos', 10)
        self.pub_image = self.create_publisher(Image, '/camera/image_annotated', 10)
        self.pub_mask = self.create_publisher(Image, '/camera/image_mask', 10)
        
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
        self.width, self.height = 640, 480
        self.center_x, self.center_y = self.width // 2, self.height // 2

        self.color_picked = True
        self.target_hsv = np.array([0, 220, 180], dtype=np.uint8) 
        self.target_bgr = np.array([0, 0, 255], dtype=np.uint8)

        self.state = "SEARCHING"
        self.roi_size = 280
        self.roi_center = (self.center_x, self.center_y)
        self.lost_time = 0.0
        self.bg_accumulator = None 

        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.statePre = np.array([[self.center_x], [self.center_y], [0], [0]], dtype=np.float32)
        self.kalman.statePost = np.array([[self.center_x], [self.center_y], [0], [0]], dtype=np.float32)

        self.morph_kernel = np.ones((3, 3), np.uint8)
        self.current_pan, self.current_tilt = 0.0, 0.0
        self.last_time = time.time()

        # =========================================================
        # AUTOMATYCZNE STROJENIE PID (AUTO-TUNER)
        # =========================================================

        #-> NOWY REKORD! Kp:0.00285, Ki:0.000000, Kd:0.00135
        self.AUTOTUNE_ENABLED = True  # Ustaw na False, gdy już znajdziesz nastawy
        self.epoch_duration = 20    # Czas trwania jednego testu (w sekundach)
        self.epoch_max = 40           # Ile testów ma wykonać
        
        self.best_mse = float('inf')
        self.best_kp = 0.001
        self.best_ki = 0.00001
        self.best_kd = 0.0005
        
        self.current_test_kp = self.best_kp
        self.current_test_ki = self.best_ki
        self.current_test_kd = self.best_kd

        self.epoch_current = 1
        self.epoch_start_time = time.time()
        self.epoch_error_sum = 0.0
        self.epoch_frame_count = 0

        # Inicjalizacja początkowych PID
        self.pid_pan = PIDController(self.current_test_kp, self.current_test_ki, self.current_test_kd)
        self.pid_tilt = PIDController(self.current_test_kp, self.current_test_ki, self.current_test_kd)

        if self.AUTOTUNE_ENABLED:
            self.get_logger().info("!!! ROZPOCZĘTO TRENING PID (WSPINACZKOWY) !!!")

    def get_dynamic_mask(self, hsv_frame):
        h, s, v = self.target_hsv
        h_tol, s_tol, v_tol = 10, 150, 140
        lower_bound = np.array([max(0, int(h)-h_tol), max(40, int(s)-s_tol), max(20, int(v)-v_tol)], dtype=np.uint8)
        upper_bound = np.array([min(179, int(h)+h_tol), min(255, int(s)+s_tol), min(255, int(v)+v_tol)], dtype=np.uint8)
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        
        if int(h) - h_tol < 0:
            lb2 = np.array([180 + (int(h)-h_tol), max(40, int(s)-s_tol), max(20, int(v)-v_tol)], dtype=np.uint8)
            ub2 = np.array([179, min(255, int(s)+s_tol), min(255, int(v)+v_tol)], dtype=np.uint8)
            mask |= cv2.inRange(hsv_frame, lb2, ub2)
        return mask

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            return

        self.height, self.width, _ = frame.shape
        self.center_x, self.center_y = self.width // 2, self.height // 2

        self.current_hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        dt = time.time() - self.last_time
        self.last_time = time.time()
        
        ball_detected = False
        measured_x, measured_y = self.center_x, self.center_y
        current_mask_display = np.zeros((self.height, self.width), dtype=np.uint8)

        if self.bg_accumulator is None:
            self.bg_accumulator = gray_frame.copy().astype("float")

        if self.state == "SEARCHING":
            cv2.accumulateWeighted(gray_frame, self.bg_accumulator, 0.1)
            bg_frame = cv2.convertScaleAbs(self.bg_accumulator)
            diff = cv2.absdiff(bg_frame, gray_frame)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            thresh = cv2.dilate(thresh, None, iterations=2)
            current_mask_display = thresh.copy()
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 1000:
                    M = cv2.moments(c)
                    if M["m00"] > 0:
                        self.roi_center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
                        self.state = "TRACKING"
                        self.lost_time = 0.0
                        self.kalman.statePre = np.array([[self.roi_center[0]], [self.roi_center[1]], [0], [0]], dtype=np.float32)
                        self.kalman.statePost = np.array([[self.roi_center[0]], [self.roi_center[1]], [0], [0]], dtype=np.float32)

        if self.state == "TRACKING":
            rx, ry = self.roi_center
            half_s = self.roi_size // 2
            
            x1, y1 = max(0, rx - half_s), max(0, ry - half_s)
            x2, y2 = min(self.width, rx + half_s), min(self.height, ry + half_s)
            
            if x2 > x1 and y2 > y1:
                roi_bgr = frame[y1:y2, x1:x2]
                roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
                
                mask = self.get_dynamic_mask(roi_hsv)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel, iterations=1)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel, iterations=2)
                
                current_mask_display[y1:y2, x1:x2] = mask
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, "PREDICTIVE ROI", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(c) > 100: 
                        M = cv2.moments(c)
                        if M["m00"] > 0:
                            local_x = int(M["m10"] / M["m00"])
                            local_y = int(M["m01"] / M["m00"])
                            measured_x = x1 + local_x
                            measured_y = y1 + local_y
                            ball_detected = True
                            self.lost_time = 0.0

            if not ball_detected:
                if self.lost_time == 0.0:
                    self.lost_time = time.time()
                elif (time.time() - self.lost_time) > 1.0:
                    self.state = "SEARCHING"
                    self.bg_accumulator = gray_frame.copy().astype("float")

        if ball_detected:
            measurement = np.array([[np.float32(measured_x)], [np.float32(measured_y)]])
            self.kalman.correct(measurement)
            cv2.circle(frame, (measured_x, measured_y), 5, (0, 255, 255), -1)

        prediction = self.kalman.predict()
        est_x, est_y = int(prediction[0]), int(prediction[1])
        cv2.circle(frame, (est_x, est_y), 8, (0, 0, 255), -1)
        cv2.drawMarker(frame, (self.center_x, self.center_y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

        if self.state == "TRACKING":
            self.roi_center = (est_x, est_y)

        # =========================================================
        # LOGIKA AUTO-TUNERA (Obliczanie błędu i modyfikacja PID)
        # =========================================================
        if self.AUTOTUNE_ENABLED and ball_detected:
            # Oblicz kwadrat błędu odległości od środka i dodaj do sumy
            error_sq = (self.center_x - est_x)**2 + (self.center_y - est_y)**2
            self.epoch_error_sum += error_sq
            self.epoch_frame_count += 1
            
            # Sprawdź, czy epoka minęła
            current_time = time.time()
            if current_time - self.epoch_start_time >= self.epoch_duration:
                mse = self.epoch_error_sum / max(1, self.epoch_frame_count)
                
                self.get_logger().info(f"[AUTOTUNE] Epoka {self.epoch_current}/{self.epoch_max} zakonczona. MSE: {mse:.2f}")

                if mse < self.best_mse:
                    self.best_mse = mse
                    self.best_kp = self.current_test_kp
                    self.best_ki = self.current_test_ki
                    self.best_kd = self.current_test_kd
                    self.get_logger().info(f"    -> NOWY REKORD! Kp:{self.best_kp:.5f}, Ki:{self.best_ki:.6f}, Kd:{self.best_kd:.5f}")

                self.epoch_current += 1
                if self.epoch_current > self.epoch_max:
                    self.get_logger().info(f"[AUTOTUNE] KONIEC! Najlepsze nastawy: Kp:{self.best_kp:.5f}, Ki:{self.best_ki:.6f}, Kd:{self.best_kd:.5f}")
                    self.AUTOTUNE_ENABLED = False
                    # Zastosuj ostateczne, najlepsze nastawy
                    self.pid_pan = PIDController(self.best_kp, self.best_ki, self.best_kd)
                    self.pid_tilt = PIDController(self.best_kp, self.best_ki, self.best_kd)
                else:
                    # Generuj nowe parametry: Najlepsze znane + losowy szum poszukiwawczy
                    self.current_test_kp = max(0.0001, self.best_kp + random.uniform(-0.001, 0.001))
                    self.current_test_ki = max(0.0, self.best_ki + random.uniform(-0.00005, 0.00005))
                    self.current_test_kd = max(0.0, self.best_kd + random.uniform(-0.0005, 0.0005))
                    
                    self.pid_pan = PIDController(self.current_test_kp, self.current_test_ki, self.current_test_kd)
                    self.pid_tilt = PIDController(self.current_test_kp, self.current_test_ki, self.current_test_kd)

                # Reset epoki
                self.epoch_error_sum = 0.0
                self.epoch_frame_count = 0
                self.epoch_start_time = time.time()

        # =========================================================

        if ball_detected:
            delta_pan = self.pid_pan.compute(self.center_x - est_x, dt)
            delta_tilt = self.pid_tilt.compute(self.center_y - est_y, dt)
            self.current_pan = max(-3.14, min(3.14, self.current_pan + delta_pan))
            self.current_tilt = max(-1.57, min(1.57, self.current_tilt + delta_tilt))

        msg_pan, msg_tilt = Float64(), Float64()
        msg_pan.data, msg_tilt.data = self.current_pan, self.current_tilt
        self.pub_pan.publish(msg_pan)
        self.pub_tilt.publish(msg_tilt)

        fps_val = int(1.0 / dt) if dt > 0 else 0
        cv2.putText(frame, f"FPS: {fps_val}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if self.AUTOTUNE_ENABLED:
            cv2.putText(frame, f"AUTO-TUNING: EPOKA {self.epoch_current}/{self.epoch_max}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            cv2.putText(frame, f"Test: P:{self.current_test_kp:.4f} I:{self.current_test_ki:.5f} D:{self.current_test_kd:.4f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(frame, f"Pan: {self.current_pan:.2f} | Tilt: {self.current_tilt:.2f}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        status_text = "LOCKED IN" if (self.state == "TRACKING" and ball_detected) else ("SEARCHING (MOTION)" if self.state == "SEARCHING" else f"LOST... ({1.0 - (time.time() - self.lost_time):.1f}s)")
        status_color = (0, 0, 255) if (self.state == "TRACKING" and ball_detected) else ((150, 150, 150) if self.state == "SEARCHING" else (0, 165, 255))
        cv2.putText(frame, status_text, (self.width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        try:
            self.pub_image.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
            self.pub_mask.publish(self.bridge.cv2_to_imgmsg(current_mask_display, "mono8"))
        except Exception:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = SimRoiTracker()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()