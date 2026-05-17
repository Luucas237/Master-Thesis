#!/usr/bin/env python3
import sys
import cv2
import numpy as np
import zmq
import pygame  # NOWOŚĆ: Biblioteka do płynnego audio

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, QFormLayout, QGroupBox, QRadioButton)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

RASPBERRY_IP = "192.168.0.43" 

class CommandCenterGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        # --- INICJALIZACJA AUDIO ---
        pygame.mixer.init()
        self.sound_enabled = True
        self.laser_firing = False
        # Upewnij się, że plik laser.mp3 jest w tym samym folderze co ten skrypt!

        self.context = zmq.Context()

        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://{RASPBERRY_IP}:5555")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"RGB")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"MASK")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"FPS")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"LASER_STATE") # NOWOŚĆ: Nasłuchujemy lasera

        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.connect(f"tcp://{RASPBERRY_IP}:5556")

        self.latest_rgb_bgr = None 
        self.initUI()
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(16) 

    def initUI(self):
        self.setWindowTitle('Centrum Dowodzenia - Tryb Hybrydowy ZMQ')
        self.resize(1100, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_layout = QVBoxLayout()

        self.lbl_fps = QLabel("FPS: 0")
        self.lbl_fps.setStyleSheet("color: lime; font-weight: bold; font-size: 14px;")
        left_layout.addWidget(self.lbl_fps)

        self.lbl_rgb = QLabel("Czekam na obraz...")
        self.lbl_rgb.setFixedSize(640, 480)
        self.lbl_rgb.setAlignment(Qt.AlignCenter)
        self.lbl_rgb.setStyleSheet("background-color: black; border: 2px solid #333;")
        self.lbl_rgb.mousePressEvent = self.pipette_click 
        left_layout.addWidget(self.lbl_rgb)

        self.lbl_mask = QLabel("Czekam na Maskę...")
        self.lbl_mask.setFixedSize(640, 240)
        self.lbl_mask.setAlignment(Qt.AlignCenter)
        self.lbl_mask.setStyleSheet("background-color: black; border: 2px solid #333;")
        left_layout.addWidget(self.lbl_mask)
        main_layout.addLayout(left_layout)

        right_layout = QVBoxLayout()

        # --- NOWOŚĆ: MODUŁ AUDIO ---
        audio_group = QGroupBox("Efekty Dźwiękowe")
        audio_layout = QHBoxLayout()
        self.btn_audio = QPushButton("🔊 ON")
        self.btn_audio.setCheckable(True)
        self.btn_audio.setChecked(True) # Domyślnie włączony
        self.btn_audio.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold;")
        self.btn_audio.clicked.connect(self.toggle_audio)
        audio_layout.addWidget(self.btn_audio)
        audio_group.setLayout(audio_layout)
        right_layout.addWidget(audio_group)

        mode_group = QGroupBox("Tryb Pracy Systemu")
        mode_layout = QHBoxLayout()
        self.radio_auto = QRadioButton("AUTOMATYCZNY")
        self.radio_manual = QRadioButton("RĘCZNY")
        self.radio_auto.setChecked(True)
        self.radio_auto.toggled.connect(self.change_mode)
        mode_layout.addWidget(self.radio_auto)
        mode_layout.addWidget(self.radio_manual)
        mode_group.setLayout(mode_layout)
        right_layout.addWidget(mode_group)

        pid_group = QGroupBox("Nastawy Pełnego PID (Tryb Auto)")
        pid_layout = QFormLayout()
        self.input_kp = QLineEdit("0.05")
        self.input_ki = QLineEdit("0.001")
        self.input_kd = QLineEdit("0.01")
        btn_apply_pid = QPushButton("Wyślij PID")
        btn_apply_pid.clicked.connect(self.apply_pid)
        pid_layout.addRow("Kp:", self.input_kp)
        pid_layout.addRow("Ki:", self.input_ki)
        pid_layout.addRow("Kd:", self.input_kd)
        pid_layout.addRow(btn_apply_pid)
        pid_group.setLayout(pid_layout)
        right_layout.addWidget(pid_group)

        manual_group = QGroupBox("Sterowanie Ręczne (PAN TILT LASER)")
        manual_layout = QFormLayout()
        self.input_pan = QLineEdit("0")
        self.input_tilt = QLineEdit("0")
        self.input_laser = QLineEdit("0")
        btn_apply_manual = QPushButton("Wyślij Komendę Ręczną")
        btn_apply_manual.setStyleSheet("background-color: #e67e22; color: white;")
        btn_apply_manual.clicked.connect(self.apply_manual)
        manual_layout.addRow("Kąt PAN (-90 do 90):", self.input_pan)
        manual_layout.addRow("Kąt TILT (-25 do 25):", self.input_tilt)
        manual_layout.addRow("Laser (0/1):", self.input_laser)
        manual_layout.addRow(btn_apply_manual)
        manual_group.setLayout(manual_layout)
        right_layout.addWidget(manual_group)

        color_group = QGroupBox("Cel HSV (Pipeta)")
        color_layout = QVBoxLayout()
        self.lbl_color_patch = QLabel("Kliknij na RGB")
        self.lbl_color_patch.setFixedSize(150, 40)
        color_layout.addWidget(self.lbl_color_patch, alignment=Qt.AlignCenter)
        color_group.setLayout(color_layout)
        right_layout.addWidget(color_group)

        right_layout.addStretch()
        main_layout.addLayout(right_layout)

    # --- NOWOŚĆ: OBSŁUGA AUDIO ---
    def toggle_audio(self):
        self.sound_enabled = self.btn_audio.isChecked()
        if self.sound_enabled:
            self.btn_audio.setText("🔊 ON")
            self.btn_audio.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold;")
        else:
            self.btn_audio.setText("🔇 OFF")
            self.btn_audio.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold;")
            pygame.mixer.music.stop() # Wyłącz natychmiast, jeśli grał

    def change_mode(self):
        mode = "AUTO" if self.radio_auto.isChecked() else "MANUAL"
        self.pub_socket.send_multipart([b"MODE", mode.encode('utf-8')])
        print(f"[ZMQ] Zmiana trybu na: {mode}")

    def apply_pid(self):
        kp, ki, kd = self.input_kp.text(), self.input_ki.text(), self.input_kd.text()
        msg = f"{kp} {ki} {kd}"
        self.pub_socket.send_multipart([b"PID", msg.encode('utf-8')])

    def apply_manual(self):
        self.radio_manual.setChecked(True) 
        pan = self.input_pan.text()
        tilt = self.input_tilt.text()
        laser = self.input_laser.text()
        msg = f"{pan} {tilt} {laser}"
        self.pub_socket.send_multipart([b"MANUAL", msg.encode('utf-8')])

    def pipette_click(self, event):
        if self.latest_rgb_bgr is not None:
            x, y = event.x(), event.y()
            try:
                b, g, r = self.latest_rgb_bgr[y, x]
                h, s, v = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
                self.lbl_color_patch.setStyleSheet(f"background-color: rgb({r},{g},{b});")
                self.pub_socket.send_multipart([b"COLOR", f"{h} {s} {v}".encode('utf-8')])
            except IndexError:
                pass

    def update_gui(self):
        try:
            while True:
                topic, msg = self.sub_socket.recv_multipart(flags=zmq.NOBLOCK)
                
                # --- LOGIKA LASERA I DŹWIĘKU ---
                if topic == b"LASER_STATE":
                    state = int(msg.decode('utf-8'))
                    if state == 1 and not self.laser_firing:
                        self.laser_firing = True
                        if self.sound_enabled:
                            try:
                                pygame.mixer.music.load("/workspace/src/pan_tilt_description/gui/laser1.mp3")
                                pygame.mixer.music.play(-1)
                            except Exception as e:
                                print(f"Błąd odtwarzania audio: {e}")
                    
                    elif state == 0 and self.laser_firing:
                        self.laser_firing = False
                        pygame.mixer.music.stop()
                        
                elif topic == b"RGB":
                    np_arr = np.frombuffer(msg, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    self.latest_rgb_bgr = frame
                    qimg = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1], QImage.Format_BGR888)
                    self.lbl_rgb.setPixmap(QPixmap.fromImage(qimg))
                elif topic == b"MASK":
                    np_arr = np.frombuffer(msg, np.uint8)
                    mask_frame = cv2.resize(cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE), (640, 240)) 
                    qimg = QImage(mask_frame.data, mask_frame.shape[1], mask_frame.shape[0], mask_frame.shape[1], QImage.Format_Grayscale8)
                    self.lbl_mask.setPixmap(QPixmap.fromImage(qimg))
                elif topic == b"FPS":
                    fps_val = msg.decode('utf-8')
                    self.lbl_fps.setText(f"FPS: {fps_val}")
        except zmq.Again:
            pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = CommandCenterGUI()
    gui.show()
    sys.exit(app.exec_())