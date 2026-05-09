#!/usr/bin/env python3
import sys
import cv2
import numpy as np
import zmq

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, QFormLayout, QGroupBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

# Zamień na IP swojej Malinki, jeśli 'raspberrypi.local' nie zadziała
RASPBERRY_IP = "raspberrypi.local" 

class CommandCenterGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # --- 1. KONFIGURACJA SIECI (ZeroMQ) ---
        self.context = zmq.Context()
        
        # Gniazdo ODBIERAJĄCE obraz z Malinki (Subskrybent)
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://{RASPBERRY_IP}:5555")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "RGB")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "MASK")
        
        # Gniazdo WYSYŁAJĄCE komendy do Malinki (Publikator)
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.connect(f"tcp://{RASPBERRY_IP}:5556")

        # --- 2. BUDOWA GUI ---
        self.latest_rgb_bgr = None # Bufor do pipety
        self.initUI()
        
        # Timer odświeżający GUI i odbierający pakiety ZMQ (ok. 60 FPS dla płynności)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(16) 

    def initUI(self):
        self.setWindowTitle('Wieżyczka - Centrum Dowodzenia (Zero-Lag ZMQ)')
        self.resize(1000, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # === LEWA STRONA (Strumienie Wideo) ===
        left_layout = QVBoxLayout()
        
        self.lbl_rgb = QLabel("Czekam na obraz RGB po ZMQ...")
        self.lbl_rgb.setFixedSize(640, 480)
        self.lbl_rgb.setAlignment(Qt.AlignCenter)
        self.lbl_rgb.setStyleSheet("background-color: black; color: lime; border: 2px solid #333;")
        self.lbl_rgb.mousePressEvent = self.pipette_click # Nasłuchiwanie kliknięć
        left_layout.addWidget(self.lbl_rgb)

        self.lbl_mask = QLabel("Czekam na Maskę po ZMQ...")
        self.lbl_mask.setFixedSize(640, 240)
        self.lbl_mask.setAlignment(Qt.AlignCenter)
        self.lbl_mask.setStyleSheet("background-color: black; color: lime; border: 2px solid #333;")
        left_layout.addWidget(self.lbl_mask)

        main_layout.addLayout(left_layout)

        # === PRAWA STRONA (Panele Sterowania) ===
        right_layout = QVBoxLayout()
        
        # Panel PID
        pid_group = QGroupBox("Nastawy Prędkości (P-Controller)")
        pid_group.setStyleSheet("font-weight: bold;")
        pid_layout = QFormLayout()
        
        self.input_kp = QLineEdit("0.05")
        btn_apply_pid = QPushButton("Wyślij Kp")
        btn_apply_pid.setStyleSheet("background-color: #2a82da; color: white;")
        btn_apply_pid.clicked.connect(self.apply_pid)
        
        pid_layout.addRow("Kp:", self.input_kp)
        pid_layout.addRow(btn_apply_pid)
        pid_group.setLayout(pid_layout)
        right_layout.addWidget(pid_group)

        # Panel Koloru (Pipeta)
        color_group = QGroupBox("Wybrany Cel (Pipeta)")
        color_group.setStyleSheet("font-weight: bold;")
        color_layout = QVBoxLayout()
        
        self.lbl_picked_color = QLabel("Kliknij na obraz RGB, aby wybrać cel")
        self.lbl_picked_color.setAlignment(Qt.AlignCenter)
        self.lbl_color_patch = QLabel()
        self.lbl_color_patch.setFixedSize(100, 50)
        self.lbl_color_patch.setStyleSheet("background-color: gray; border: 1px solid white;")
        
        color_layout.addWidget(self.lbl_picked_color)
        color_layout.addWidget(self.lbl_color_patch, alignment=Qt.AlignCenter)
        color_group.setLayout(color_layout)
        right_layout.addWidget(color_group)

        # E-STOP
        btn_estop = QPushButton("TWARDY STOP (E-STOP)")
        btn_estop.setStyleSheet("background-color: red; color: white; font-size: 16px; height: 50px;")
        btn_estop.clicked.connect(self.send_estop)
        right_layout.addWidget(btn_estop)

        right_layout.addStretch()
        main_layout.addLayout(right_layout)

    # --- FUNKCJE OBSŁUGI ZDARZEŃ ---
    def pipette_click(self, event):
        if self.latest_rgb_bgr is not None:
            x, y = event.x(), event.y()
            try:
                b, g, r = self.latest_rgb_bgr[y, x]
                pixel_bgr = np.uint8([[[b, g, r]]])
                h, s, v = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)[0][0]
                
                self.lbl_picked_color.setText(f"Wybrano HSV: {h}, {s}, {v}")
                self.lbl_color_patch.setStyleSheet(f"background-color: rgb({r},{g},{b});")
                
                # Wysyłamy nowy cel przez ZMQ (Topic: COLOR)
                msg = f"{h} {s} {v}"
                self.pub_socket.send_multipart([b"COLOR", msg.encode('utf-8')])
                print(f"[ZMQ] Wysłano kolor: {msg}")
            except IndexError:
                pass

    def apply_pid(self):
        kp = self.input_kp.text()
        self.pub_socket.send_multipart([b"PID", kp.encode('utf-8')])
        print(f"[ZMQ] Wysłano Kp: {kp}")

    def send_estop(self):
        self.pub_socket.send_multipart([b"ESTOP", b"1"])
        print("[ZMQ] Wysłano sygnał E-STOP!")

    # --- PĘTLA GŁÓWNA ---
    def update_gui(self):
        # Asynchroniczne sprawdzanie poczty ZMQ
        try:
            while True:
                # Odbieramy wiadomości bez blokowania programu
                topic, msg = self.sub_socket.recv_multipart(flags=zmq.NOBLOCK)
                
                np_arr = np.frombuffer(msg, np.uint8)
                if topic == b"RGB":
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    self.latest_rgb_bgr = frame
                    
                    # Konwersja na format Qt i wyświetlenie
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_frame.shape
                    qimg = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
                    self.lbl_rgb.setPixmap(QPixmap.fromImage(qimg))
                    
                elif topic == b"MASK":
                    mask_frame = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
                    mask_resized = cv2.resize(mask_frame, (640, 240)) 
                    h, w = mask_resized.shape
                    qimg = QImage(mask_resized.data, w, h, w, QImage.Format_Grayscale8)
                    self.lbl_mask.setPixmap(QPixmap.fromImage(qimg))
                    
        except zmq.Again:
            # Pusta kolejka, GUI może się odświeżyć
            pass

    def closeEvent(self, event):
        self.sub_socket.close()
        self.pub_socket.close()
        self.context.term()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = CommandCenterGUI()
    gui.show()
    sys.exit(app.exec_())