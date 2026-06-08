#!/usr/bin/env python3
import cv2
import numpy as np
import serial
import time
import threading
import csv
import datetime
# (self, kp=0.005, ki=0.0, kd=0.0001):
    # (self, kp=0.084, ki=0.0, kd=0.0063):
    # (self, kp=0.067, ki=0.0, kd=0.011):
    # (self, kp=0.021, ki=0.0, kd=0.0136):
class SimplePID:
    def __init__(self, kp=0.058, ki=0.0, kd=0.0019):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0  # Pamięć poprzedniej różniczki dla filtra

    def compute(self, error, dt):
        if dt <= 0: return 0.0
        
        self.integral += error * dt
        
        # Surowa różniczka
        raw_derivative = (error - self.prev_error) / dt
        self.prev_error = error
        
        # FILTR DOLNOPRZEPUSTOWY CZŁONU D (Uspokaja drgawki od szumu kamery)
        # alpha określa siłę filtra. 1.0 = brak filtra, 0.1 = bardzo silny filtr
        alpha = 0.3 
        filtered_derivative = (alpha * raw_derivative) + ((1.0 - alpha) * self.prev_derivative)
        self.prev_derivative = filtered_derivative
        
        # Obliczenie wyniku z przefiltrowaną różniczką
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * filtered_derivative)
        
        # KAGANIEC / Saturacja wyjścia
        limit = 1.5 
        if output > limit: output = limit
        elif output < -limit: output = -limit
        
        return output

    def set_params(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0

class HeadlessTurretTracker:
    def __init__(self):
        try:
            self.ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.05)
            time.sleep(2)
        except Exception:
            self.ser = None
            print("[OSTRZEŻENIE] Nie wykryto połączenia z ESP32 (ttyUSB0).")

        self.latest_frame = None
        self.running = True
        
        # Bezpośredni odczyt sprzętowy kamery (bez obracania obrazu!)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Wątek kamery oraz wątek terminala
        threading.Thread(target=self.grab_frames, daemon=True).start()
        threading.Thread(target=self.terminal_listener, daemon=True).start()

        self.mode = "AUTO"
        self.ctrl_type = "PID"
        self.state = "SEARCHING"
        self.target_hsv = np.array([0, 220, 180], dtype=np.uint8)
        self.pid_pan = SimplePID(kp=0.058, ki=0.0, kd=0.0019)
        self.pid_tilt = SimplePID(kp=0.058, ki=0.0, kd=0.0019)
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
        
        # Logowanie
        self.logging_started = False
        self.log_file = None
        self.csv_writer = None
        self.start_log_time = 0.0
        
        self.do_probe = False
        
        self.reset_kalman()
        self.frame_count = 0
        self.fps_timer = time.time()
        self.fps_filtered = 0.0
        self.last_time = time.time()
        print("\n[SYSTEM] Tracker Headless zainicjowany pomyślnie.")

    def reset_kalman(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],
                                                 [0,1,0,1],
                                                 [0,0,1,0],
                                                 [0,0,0,1]], np.float32)

        # Spokojniejsze nastawy Kalmana dla lepszej filtracji
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.003
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5.0

    def grab_frames(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret: 
                # Zostawiamy surowy odczyt, BEZ odwracania obrazu!
                self.latest_frame = frame
            else: 
                time.sleep(0.01)

    def classify_color_hsv(self, h, s, v):
        """Klasyfikacja barwy na podstawie przestrzeni HSV."""
        if v < 40: return "Czarny / Ciemny"
        if s < 30 and v > 200: return "Biały"
        if s < 40: return "Szary"

        if h < 10 or h > 165: return "Czerwony"
        elif h < 22: return "Pomarańczowy"
        elif h < 35: return "Żółty"
        elif h < 85: return "Zielony"
        elif h < 110: return "Błękitny (Cyjan)"
        elif h < 135: return "Niebieski"
        elif h < 165: return "Fioletowy / Różowy"
        return "Nieokreślony"

    def terminal_listener(self):
        print("\n=== KONSOLA STEROWANIA (Wpisz 'help' aby zobaczyć komendy) ===")
        while self.running:
            try:
                cmd_line = input("> ").strip().lower().split()
                if not cmd_line:
                    continue
                
                cmd = cmd_line[0]

                if cmd == "help":
                    print("\n--- DOSTĘPNE KOMENDY ---")
                    print("  mode auto     - Przełącza na śledzenie autonomiczne")
                    print("  mode manual   - Przełącza na sterowanie ręczne")
                    print("  ctrl pid      - Zmienia algorytm na regulator PID")
                    print("  ctrl bang     - Zmienia algorytm na trójpołożeniowy (Bang-Bang)")
                    print("  move P T L    - Ruch ręczny (np: move 15 -10 1) [Pan, Tilt, Laser]")
                    print("  color H S V   - Ręczna nastawa koloru HSV (np: color 25 200 200)")
                    print("  probe         - PIPETA: Zczytuje kolor ze środka kadru kamery")
                    print("  log start     - Rozpoczyna zapis danych uchybu do pliku CSV")
                    print("  log stop      - Kończy i zapisuje plik CSV")
                    print("  quit / exit   - Bezpieczne zamknięcie programu\n")
                    
                elif cmd == "mode":
                    if len(cmd_line) > 1 and cmd_line[1] in ["auto", "manual"]:
                        self.mode = cmd_line[1].upper()
                        print(f"[SYSTEM] Zmieniono tryb na: {self.mode}")
                    else:
                        print("Użycie: mode auto | mode manual")

                elif cmd == "ctrl":
                    if len(cmd_line) > 1 and cmd_line[1] in ["pid", "bang"]:
                        self.ctrl_type = "PID" if cmd_line[1] == "pid" else "BANG_BANG"
                        print(f"[SYSTEM] Aktywny regulator: {self.ctrl_type}")
                    else:
                        print("Użycie: ctrl pid | ctrl bang")
                        
                elif cmd == "move":
                    # if self.mode != "MANUAL":
                    #     print("[BŁĄD] Najpierw przełącz na tryb ręczny! (komenda: mode manual)")
                    #     continue
                    if len(cmd_line) == 4:
                        self.current_pan = float(cmd_line[1])
                        self.current_tilt = float(cmd_line[2])
                        self.laser_state = int(cmd_line[3])
                        print(f"[MANUAL] Jadę do: PAN={self.current_pan} | TILT={self.current_tilt} | LASER={self.laser_state}")
                    else:
                        print("Użycie: move <pan> <tilt> <laser> (np. move 45 -10 1)")
                        
                elif cmd == "color":
                    if len(cmd_line) == 4:
                        h, s, v = int(cmd_line[1]), int(cmd_line[2]), int(cmd_line[3])
                        self.target_hsv = np.array([h, s, v], dtype=np.uint8)
                        nazwa = self.classify_color_hsv(h, s, v)
                        print(f"[WIZJA] Nowy cel HSV: [{h}, {s}, {v}] -> Sklasyfikowano jako: {nazwa}")
                    else:
                        print("Użycie: color <H> <S> <V>")
                        
                elif cmd == "probe":
                    self.do_probe = True
                    print("[WIZJA] Oczekiwanie na pobranie koloru z centrum kadru...")
                    
                elif cmd == "log":
                    if len(cmd_line) > 1 and cmd_line[1] == "start":
                        if not self.logging_started:
                            self.logging_started = True
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            self.filename = f"A_PID_move_log_{timestamp}.csv"
                            self.log_file = open(self.filename, mode='w', newline='')
                            self.csv_writer = csv.writer(self.log_file)
                            self.csv_writer.writerow(['Czas_s', 'Uchyb_X_px', 'Uchyb_Y_px', 'Regulator', 'Param1_Pan', 'Param2_Pan', 'Param1_Tilt', 'Param2_Tilt', 'Deadband', 'Pole_Obiektu'])
                            self.start_log_time = time.time()
                            print(f"[LOGI] Rozpoczęto logowanie do {self.filename}")
                    elif len(cmd_line) > 1 and cmd_line[1] == "stop":
                        if self.logging_started:
                            self.logging_started = False
                            if self.log_file:
                                self.log_file.close()
                                self.log_file = None
                                self.csv_writer = None
                            print(f"[LOGI] Zakończono logowanie. Zapisano: {self.filename}")
                    else:
                        print("Użycie: log start | log stop")
                        
                elif cmd in ["quit", "exit", "q"]:
                    self.running = False
                    print("[SYSTEM] Zamykanie...")
                    break
                else:
                    print(f"Nieznana komenda: '{cmd}'. Wpisz 'help'.")
                    
            except ValueError:
                print("[BŁĄD] Podałeś literę zamiast cyfry!")
            except Exception as e:
                print(f"[BŁĄD KONSOLI] {e}")

    def run(self):
        BANG_SPEED_PAN = 1.0
        BANG_SPEED_TILT = 1.0
        morph_kernel = np.ones((9, 9), np.uint8)
        
        # Pamięć dla adaptacyjnego systemu śledzenia
        current_deadband = 80
        current_scale = 1.0
        
        while self.running:
            if self.latest_frame is None:
                time.sleep(0.01)
                continue

            frame = self.latest_frame.copy()
            self.latest_frame = None
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            roi_w, roi_h = int(w * 0.4), int(h * 0.4)
            
            now = time.time()
            dt = now - self.last_time
            self.last_time = now

            error_x = 0
            error_y = 0

            # --- PIPETA ---
            if self.do_probe:
                roi = frame[center_y-10:center_y+10, center_x-10:center_x+10]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                median_color = np.median(hsv_roi, axis=(0,1)).astype(np.uint8)
                self.target_hsv = median_color
                kolor_nazwa = self.classify_color_hsv(median_color[0], median_color[1], median_color[2])
                print(f"\n[WIZJA] Sukces! Pobrany kolor: H:{median_color[0]} S:{median_color[1]} V:{median_color[2]} ({kolor_nazwa})\n> ", end="", flush=True)
                self.do_probe = False
            # --------------

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
                    contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
                        self.state = "TRACKING"
                        self.laser_state = 1
                        self.kalman.predict()
                        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
                        self.kalman.correct(measurement)
                        error_x = center_x - cx
                        error_y = center_y - cy
                        
                        # --- ADAPTACYJNY SYSTEM (GAIN SCHEDULING) ---
                        area = cv2.contourArea(c)
                        # if area > 15000:  
                        #     # OBIEKT BLISKO: Cel ma ogromną prędkość kątową i szybko ucieka z kadru.
                        #     current_deadband = 100 # Zwiększamy nieco martwą strefę (bo środek dużej plamy lubi pływać)
                        #     current_scale = 1.0   # DOPALACZ: +30% mocy, żeby wieżyczka nadążyła za szybkim ruchem!
                            
                        # elif area < 3000: 
                        #     # OBIEKT DALEKO: Cel to kropka, minimalne prędkości kątowe.
                        #     current_deadband = 50 # Nie reagujemy na każdy szum z 10 pikseli
                        #     current_scale = 0.3   # REDUKCJA: -70% mocy silnika! Delikatne ruchy, żeby nie przestrzelić i nie oscylować.
                            
                        # else:             
                        #     # ŚREDNI DYSTANS (Optymalny)
                        #     current_deadband = 80 
                        #     current_scale = 0.7
                        urrent_deadband = 75  # (Zmień na 75 dla testu analitycznego)
                        current_scale = 1.0
                        # --------------------------------------------
                        
                        if self.ctrl_type == "PID":
                            if self.pan_enabled:
                                if abs(error_x) >= current_deadband:
                                    self.current_pan += (self.pid_pan.compute(error_x, dt) * current_scale) * -1.0
                                else:
                                    self.pid_pan.prev_error = error_x
                            if self.tilt_enabled:
                                if abs(error_y) >= current_deadband:
                                    self.current_tilt += (self.pid_tilt.compute(error_y, dt) * current_scale)
                                else:
                                    self.pid_tilt.prev_error = error_y
                                    
                        elif self.ctrl_type == "BANG_BANG":
                            if self.pan_enabled:
                                if abs(error_x) >= current_deadband:
                                    self.current_pan += np.sign(error_x) * BANG_SPEED_PAN * current_scale * -1.0
                            if self.tilt_enabled:
                                if abs(error_y) >= current_deadband:
                                    self.current_tilt += np.sign(error_y) * BANG_SPEED_TILT * current_scale
                                    
                        self.roi_rect = [max(0, cx - roi_w//2), max(0, cy - roi_h//2),
                                         min(w, cx + roi_w//2), min(h, cy + roi_h//2)]

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
                                
                                # Podczas predykcji używamy ostatnich znanych wartości ze skali
                                if self.ctrl_type == "PID":
                                    if self.pan_enabled:
                                        if abs(error_x) >= current_deadband:
                                            self.current_pan += (self.pid_pan.compute(error_x, dt) * current_scale) * -1.0
                                        else:
                                            self.pid_pan.prev_error = error_x
                                    if self.tilt_enabled:
                                        if abs(error_y) >= current_deadband:
                                            self.current_tilt += (self.pid_tilt.compute(error_y, dt) * current_scale)
                                        else:
                                            self.pid_tilt.prev_error = error_y
                                            
                                elif self.ctrl_type == "BANG_BANG":
                                    if self.pan_enabled:
                                        if abs(error_x) >= current_deadband:
                                            self.current_pan += np.sign(error_x) * BANG_SPEED_PAN * current_scale * -1.0
                                    if self.tilt_enabled:
                                        if abs(error_y) >= current_deadband:
                                            self.current_tilt += np.sign(error_y) * BANG_SPEED_TILT * current_scale
                                            
                                self.roi_rect = [max(0, px - roi_w//2), max(0, py - roi_h//2),
                                                 min(w, px + roi_w//2), min(h, py + roi_h//2)]
                            else:
                                # self.current_pan = 0.0
                                # self.current_tilt = 0.0
                                self.laser_state = 0
                                self.state = "SEARCHING"
                                self.prev_gray = None
                                self.reset_kalman()

                # Zapis do CSV
                if self.logging_started and self.csv_writer:
                    current_log_time = time.time() - self.start_log_time
                    
                    # Pobieranie odpowiednich nastaw w zależności od aktywnego regulatora
                    if self.ctrl_type == "PID":
                        p1_pan, p2_pan = self.pid_pan.kp, self.pid_pan.kd
                        p1_tilt, p2_tilt = self.pid_tilt.kp, self.pid_tilt.kd
                    else: # BANG_BANG
                        p1_pan, p2_pan = BANG_SPEED_PAN, 0.0
                        p1_tilt, p2_tilt = BANG_SPEED_TILT, 0.0
                        
                    # Zabezpieczenie na wypadek, gdyby obiekt zniknął (Kalman)
                    zapisane_pole = cv2.contourArea(c) if 'c' in locals() and self.state == "TRACKING" else 0

                    self.csv_writer.writerow([
                        f"{current_log_time:.3f}", error_x, error_y,
                        self.ctrl_type, p1_pan, p2_pan, p1_tilt, p2_tilt,
                        current_deadband, zapisane_pole
                    ])
                self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.prev_gray = cv2.GaussianBlur(self.prev_gray, (21, 21), 0)

            elif self.mode == "MANUAL":
                self.state = "SEARCHING"
                self.prev_gray = None

            # Ograniczenia mechaniczne i wysyłka do ESP32
            self.current_tilt = max(-25.0, min(25.0, self.current_tilt))
            self.current_pan = max(-90.0, min(90.0, self.current_pan))
            if self.ser:
                if abs(self.current_pan - self.last_sent_pan) >= 0.3 or \
                   abs(self.current_tilt - self.last_sent_tilt) >= 0.3 or \
                   self.laser_state != self.last_sent_laser:
                    cmd = f"{self.current_pan:.2f} {self.current_tilt:.2f} {self.laser_state}\n"
                    self.ser.write(cmd.encode('utf-8'))
                    self.last_sent_pan = self.current_pan
                    self.last_sent_tilt = self.current_tilt
                    self.last_sent_laser = self.laser_state

        # Zamknięcie programu
        self.cap.release()
        if self.log_file:
            self.log_file.close()
        if self.ser:
            self.ser.close()

if __name__ == '__main__':
    tracker = HeadlessTurretTracker()
    try:
        tracker.run()
    except KeyboardInterrupt:
        tracker.running = False
        print("\n[SYSTEM] Przerwano działanie (Ctrl+C). Zamknięto bezpiecznie.")