#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import cv2
import numpy as np
import time

# --- PROSTY REGULATOR PID ---
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, error, dt):
        if dt <= 0.0:
            return 0.0
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

# --- GŁÓWNY WĘZEŁ ROS 2 ---
class TurretTracker(Node):
    def __init__(self):
        super().__init__('vision_tracker')
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)
        
        self.cap = cv2.VideoCapture(0)
        
        self.width = 640
        self.height = 480
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        self.center_x = self.width // 2
        self.center_y = self.height // 2

        # --- KONFIGURACJA GUI ---
        self.window_name = "Kamera - Widok z celownikiem"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        self.color_picked = False
        self.target_hsv = None
        self.target_bgr = None  # Do wyświetlania koloru w kwadraciku
        self.current_hsv_frame = None
        self.current_bgr_frame = None

        # --- FILTR KALMANA ---
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], 
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], 
                                                 [0, 1, 0, 1], 
                                                 [0, 0, 1, 0], 
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        # --- REGULATORY PID ---
        self.pid_pan = PIDController(kp=0.0008, ki=0.0, kd=0.0004)
        self.pid_tilt = PIDController(kp=0.0008, ki=0.0, kd=0.0004)

        self.current_pan = 0.0
        self.current_tilt = 0.0
        self.last_time = time.time()
        
        # Kernel (element strukturalny) do operacji morfologicznych
        self.morph_kernel = np.ones((7, 7), np.uint8)

        self.timer = self.create_timer(0.033, self.timer_callback)
        self.get_logger().info("System trackingu z zaawansowanym HUD uruchomiony!")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_hsv_frame is not None and self.current_bgr_frame is not None:
                self.target_hsv = self.current_hsv_frame[y, x]
                self.target_bgr = self.current_bgr_frame[y, x]
                self.color_picked = True

    def get_dynamic_mask(self, hsv_frame):
        h, s, v = self.target_hsv
        h_tol, s_tol, v_tol = 15, 60, 60
        
        lower_bound = np.array([max(0, int(h) - h_tol), max(50, int(s) - s_tol), max(50, int(v) - v_tol)], dtype=np.uint8)
        upper_bound = np.array([min(179, int(h) + h_tol), min(255, int(s) + s_tol), min(255, int(v) + v_tol)], dtype=np.uint8)
        
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        
        if int(h) - h_tol < 0:
            lower_bound2 = np.array([180 + (int(h) - h_tol), max(50, int(s) - s_tol), max(50, int(v) - v_tol)], dtype=np.uint8)
            upper_bound2 = np.array([179, min(255, int(s) + s_tol), min(255, int(v) + v_tol)], dtype=np.uint8)
            mask2 = cv2.inRange(hsv_frame, lower_bound2, upper_bound2)
            mask = mask | mask2
        elif int(h) + h_tol > 179:
            lower_bound2 = np.array([0, max(50, int(s) - s_tol), max(50, int(v) - v_tol)], dtype=np.uint8)
            upper_bound2 = np.array([(int(h) + h_tol) - 180, min(255, int(s) + s_tol), min(255, int(v) + v_tol)], dtype=np.uint8)
            mask2 = cv2.inRange(hsv_frame, lower_bound2, upper_bound2)
            mask = mask | mask2
            
        return mask

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        self.current_bgr_frame = frame.copy()
        
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.current_hsv_frame = hsv_frame

        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        # --- GENEROWANIE I CZYSZCZENIE MASKI ---
        if self.color_picked:
            mask = self.get_dynamic_mask(hsv_frame)
            # 1. MORPH_OPEN: Usuwamy małe szumy (np. usta, drobne refleksy)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel, iterations=2)
            # 2. MORPH_CLOSE: Łatamy dziury wewnątrz wykrytego obiektu
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel, iterations=2)
        else:
            mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        ball_detected = False
        measured_x, measured_y = self.center_x, self.center_y

        if contours and self.color_picked:
            c = max(contours, key=cv2.contourArea)
            # Zwiększyliśmy próg z 500 na 1000, bo po morfologii szum jest ubity, szukamy konkretnych brył
            if cv2.contourArea(c) > 1000:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    measured_x = int(M["m10"] / M["m00"])
                    measured_y = int(M["m01"] / M["m00"])
                    ball_detected = True

        if ball_detected:
            measurement = np.array([[np.float32(measured_x)], [np.float32(measured_y)]])
            self.kalman.correct(measurement)
            cv2.circle(frame, (measured_x, measured_y), 5, (0, 255, 255), -1)

        prediction = self.kalman.predict()
        est_x, est_y = int(prediction[0]), int(prediction[1])

        cv2.circle(frame, (est_x, est_y), 8, (0, 0, 255), -1)
        cv2.drawMarker(frame, (self.center_x, self.center_y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

        error_x = self.center_x - est_x 
        error_y = self.center_y - est_y

        if ball_detected:
            delta_pan = self.pid_pan.compute(error_x, dt)
            delta_tilt = self.pid_tilt.compute(error_y, dt)

            self.current_pan += delta_pan
            self.current_tilt += delta_tilt

            self.current_pan = max(-3.14, min(3.14, self.current_pan))
            self.current_tilt = max(-1.57, min(1.57, self.current_tilt))

        # --- RYSUJEMY INTERFEJS (HUD) ---
        # 1. FPS (Lewy górny róg)
        fps_val = int(1.0 / dt) if dt > 0 else 0
        cv2.putText(frame, f"FPS: {fps_val}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 2. Kwadrat z pobranym kolorem (Pod FPS)
        if self.color_picked:
            color_to_draw = (int(self.target_bgr[0]), int(self.target_bgr[1]), int(self.target_bgr[2]))
            cv2.rectangle(frame, (10, 45), (40, 75), color_to_draw, -1)
            cv2.rectangle(frame, (10, 45), (40, 75), (255, 255, 255), 1) # Biała ramka wokół kwadratu

        # 3. Status SEARCHING / LOCKED IN (Prawy górny róg)
        status_text = "LOCKED IN" if ball_detected else "SEARCHING"
        status_color = (0, 0, 255) if ball_detected else (150, 150, 150)
        
        # Obliczamy szerokość tekstu, żeby równać do prawej
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = self.width - text_size[0] - 10
        cv2.putText(frame, status_text, (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Publikacja pozycji na kanał ROS 2
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['pan_joint', 'tilt_joint']
        msg.position = [self.current_pan, self.current_tilt]
        self.publisher_.publish(msg)

        cv2.imshow(self.window_name, frame)
        cv2.imshow("Maska HSV - Wykrywanie Czerwieni", mask)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = TurretTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.cap.release()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()