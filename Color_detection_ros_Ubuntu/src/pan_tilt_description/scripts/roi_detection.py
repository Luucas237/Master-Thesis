#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import cv2
import numpy as np
import time

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

class RoiTracker(Node):
    def __init__(self):
        super().__init__('roi_tracker')
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)
        self.cap = cv2.VideoCapture(0)
        self.width, self.height = 640, 480
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.center_x, self.center_y = self.width // 2, self.height // 2

        # --- GUI ---
        self.window_main = "Kamera - Globalna"
        self.window_roi = "Powiekszone ROI"
        self.window_mask = "Maska HSV w ROI"
        cv2.namedWindow(self.window_main)
        cv2.setMouseCallback(self.window_main, self.mouse_callback)

        self.color_picked = False
        self.target_hsv = None
        self.target_bgr = None
        self.current_hsv_frame = None
        self.current_bgr_frame = None

        # --- MASZYNA STANÓW I ROI ---
        self.state = "SEARCHING"
        self.roi_size = 280      # ZWIĘKSZONE: Pozwala złapać szybki ruch ręki i piłki
        self.roi_center = (self.center_x, self.center_y)
        self.lost_time = 0.0
        self.bg_accumulator = None 

        # --- FILTRY I PID ---
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        # Transition Matrix określa uwzględnianie prędkości (dx, dy)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.pid_pan = PIDController(kp=0.0008, ki=0.0, kd=0.0004)
        self.pid_tilt = PIDController(kp=0.0008, ki=0.0, kd=0.0004)
        self.current_pan, self.current_tilt = 0.0, 0.0
        self.last_time = time.time()
        
        # ZMNIEJSZONY KERNEL: 3x3 nie usunie tak łatwo rozmazanej, szybkiej piłki
        self.morph_kernel = np.ones((3, 3), np.uint8)

        # ODBLOKOWANE FPS: 0.016s = ~60 FPS
        self.timer = self.create_timer(0.016, self.timer_callback)
        self.get_logger().info("System PREDYKCYJNEGO ROI uruchomiony!")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.current_hsv_frame is not None:
            self.target_hsv = self.current_hsv_frame[y, x]
            self.target_bgr = self.current_bgr_frame[y, x]
            self.color_picked = True
            self.state = "SEARCHING"
            # Reset Kalmana przy nowym kliknięciu (Wymuszony float32!)
            self.kalman.statePre = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self.kalman.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)

    def get_dynamic_mask(self, hsv_frame):
        h, s, v = self.target_hsv
        # ZACIŚNIĘTE FILTRY: H=10 (odcina żółty i skórę), S/V=50 (wymaga czystego koloru)
        h_tol, s_tol, v_tol = 10, 50, 50
        # Podnosimy też dolny, sztywny próg jasności z 30 na 80, żeby ignorować cienie i ciemne obiekty
        lower_bound = np.array([max(0, int(h)-h_tol), max(80, int(s)-s_tol), max(80, int(v)-v_tol)], dtype=np.uint8)
        upper_bound = np.array([min(179, int(h)+h_tol), min(255, int(s)+s_tol), min(255, int(v)+v_tol)], dtype=np.uint8)
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        
        if int(h) - h_tol < 0:
            lb2 = np.array([180 + (int(h)-h_tol), max(80, int(s)-s_tol), max(80, int(v)-v_tol)], dtype=np.uint8)
            ub2 = np.array([179, min(255, int(s)+s_tol), min(255, int(v)+v_tol)], dtype=np.uint8)
            mask |= cv2.inRange(hsv_frame, lb2, ub2)
        elif int(h) + h_tol > 179:
            lb2 = np.array([0, max(80, int(s)-s_tol), max(80, int(v)-v_tol)], dtype=np.uint8)
            ub2 = np.array([(int(h)+h_tol)-180, min(255, int(s)+s_tol), min(255, int(v)+v_tol)], dtype=np.uint8)
            mask |= cv2.inRange(hsv_frame, lb2, ub2)
        return mask

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret: return
        frame = cv2.flip(frame, 1)
        self.current_bgr_frame = frame.copy()
        self.current_hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        dt = time.time() - self.last_time
        self.last_time = time.time()

        display_roi = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        display_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        ball_detected = False
        measured_x, measured_y = self.center_x, self.center_y

        if self.bg_accumulator is None:
            self.bg_accumulator = gray_frame.copy().astype("float")

        if not self.color_picked:
            self.state = "IDLE"
            cv2.putText(frame, "KLIKNIJ ABY WYBRAC KOLOR", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.accumulateWeighted(gray_frame, self.bg_accumulator, 0.5)

        # ==========================================
        # STAN: SEARCHING (Szukanie ruchu po całości)
        # ==========================================
        elif self.state == "SEARCHING":
            cv2.accumulateWeighted(gray_frame, self.bg_accumulator, 0.1)
            bg_frame = cv2.convertScaleAbs(self.bg_accumulator)
            diff = cv2.absdiff(bg_frame, gray_frame)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 1000:
                    M = cv2.moments(c)
                    if M["m00"] > 0:
                        self.roi_center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
                        self.state = "TRACKING"
                        self.lost_time = 0.0
                        # Nadpisujemy Kalmana, żeby nie uciekł na start (Wymuszony float32!)
                        self.kalman.statePre = np.array([[self.roi_center[0]], [self.roi_center[1]], [0], [0]], dtype=np.float32)
                        self.kalman.statePost = np.array([[self.roi_center[0]], [self.roi_center[1]], [0], [0]], dtype=np.float32)

        # ==========================================
        # STAN: TRACKING (Analiza koloru tylko w ROI)
        # ==========================================
        if self.state == "TRACKING":
            rx, ry = self.roi_center
            half_s = self.roi_size // 2
            
            x1, y1 = max(0, rx - half_s), max(0, ry - half_s)
            x2, y2 = min(self.width, rx + half_s), min(self.height, ry + half_s)
            
            if x2 > x1 and y2 > y1:
                roi_bgr = frame[y1:y2, x1:x2]
                roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
                
                mask = self.get_dynamic_mask(roi_hsv)
                # TYLKO 1 ITERACJA: Delikatne usuwanie szumów, żeby zachować smugę rozmytej piłki
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel, iterations=1)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel, iterations=2)
                
                display_roi = cv2.resize(roi_bgr, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                display_mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, "PREDICTIVE ROI", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(c) > 100: # Drastycznie obniżony próg dla rozmytych smug!
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

        # --- AKTUALIZACJA KALMANA I PID ---
        if ball_detected:
            measurement = np.array([[np.float32(measured_x)], [np.float32(measured_y)]])
            self.kalman.correct(measurement)
            cv2.circle(frame, (measured_x, measured_y), 5, (0, 255, 255), -1)

        # KALMAN PRZEWIDUJE PRZYSZŁOŚĆ: Nawet jeśli w tej klatce nie wykryto piłki (zgubiona na ułamek sekundy)
        prediction = self.kalman.predict()
        est_x, est_y = int(prediction[0]), int(prediction[1])
        cv2.circle(frame, (est_x, est_y), 8, (0, 0, 255), -1)
        cv2.drawMarker(frame, (self.center_x, self.center_y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

        if self.state == "TRACKING":
            # PREDYKCYJNY ROI: Ustawiamy środek ROI na następną klatkę tam, gdzie Kalman zakłada że doleci piłka!
            self.roi_center = (est_x, est_y)

        if ball_detected:
            delta_pan = self.pid_pan.compute(self.center_x - est_x, dt)
            delta_tilt = self.pid_tilt.compute(self.center_y - est_y, dt)
            self.current_pan = max(-3.14, min(3.14, self.current_pan + delta_pan))
            self.current_tilt = max(-1.57, min(1.57, self.current_tilt + delta_tilt))

        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['pan_joint', 'tilt_joint']
        msg.position = [self.current_pan, self.current_tilt]
        self.publisher_.publish(msg)

        # --- HUD (INTERFEJS) ---
        fps_val = int(1.0 / dt) if dt > 0 else 0
        cv2.putText(frame, f"FPS: {fps_val}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if self.color_picked:
            color_to_draw = (int(self.target_bgr[0]), int(self.target_bgr[1]), int(self.target_bgr[2]))
            cv2.rectangle(frame, (10, 45), (40, 75), color_to_draw, -1)
            cv2.rectangle(frame, (10, 45), (40, 75), (255, 255, 255), 1)

        if self.state == "SEARCHING":
            status_text = "SEARCHING (MOTION)"
            status_color = (150, 150, 150)
        elif self.state == "TRACKING" and ball_detected:
            status_text = "LOCKED IN"
            status_color = (0, 0, 255)
        else:
            status_text = f"LOST... ({1.0 - (time.time() - self.lost_time):.1f}s)"
            status_color = (0, 165, 255)
            
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(frame, status_text, (self.width - text_size[0] - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        cv2.imshow(self.window_main, frame)
        cv2.imshow(self.window_roi, display_roi)
        cv2.imshow(self.window_mask, display_mask)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = RoiTracker()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.cap.release()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()