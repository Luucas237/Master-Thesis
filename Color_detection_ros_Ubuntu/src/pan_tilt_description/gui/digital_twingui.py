#!/usr/bin/env python3
import sys
import os
import cv2
import numpy as np
import pygame
import threading
import time
import socket

from ament_index_python.packages import get_package_share_directory
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, QFormLayout, QGroupBox, QRadioButton, QCheckBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

# ADRES RASPBERRY PI
RASPBERRY_IP = "192.168.0.43"

class DigitalTwinGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        pygame.mixer.init()
        self.sound_enabled = True
        self.laser_firing = False

        # ZMIENNE WIZYJNE DO LOKALNYCH OBLICZEŃ
        self.latest_frame = None
        self.mask_display = np.zeros((480, 640), dtype=np.uint8)
        self.target_hsv = np.array([25, 255, 183], dtype=np.uint8)
        self.state = "SEARCHING"
        self.prev_gray = None
        self.roi_rect = None
        self.lost_time = 0.0
        self.running = True
        self.reset_kalman()

        # ODCZYT ZE STRUMIENIA SIECIOWEGO
        self.cap = cv2.VideoCapture("udp://@239.255.0.1:5000")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        threading.Thread(target=self.grab_frames, daemon=True).start()
        threading.Thread(target=self.process_vision, daemon=True).start()

        self.initUI()
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(30) # Aktualizacja GUI co 30ms (~30 FPS)

    def reset_kalman(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5.0

    def grab_frames(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret: 
                self.latest_frame = frame
            else: 
                time.sleep(0.01)

    def process_vision(self):
        # Symulacja algorytmów z Malinki na lokalnym laptopie
        morph_kernel = np.ones((9, 9), np.uint8)
        
        while self.running:
            if self.latest_frame is None:
                time.sleep(0.01)
                continue
                
            frame = self.latest_frame.copy()
            h, w = frame.shape[:2]
            roi_w, roi_h = int(w * 0.4), int(h * 0.4)
            self.mask_display = np.zeros((h, w), dtype=np.uint8)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if self.state == "SEARCHING":
                self.laser_firing = False
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
                                self.roi_rect = [x1, y1, min(w, x1 + roi_w), min(h, y1 + roi_h)]
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
                self.mask_display[y1:y2, x1:x2] = mask_roi
                contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

                target_found = False
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(c) > 300:
                        M = cv2.moments(c)
                        if M["m00"] > 0:
                            cx, cy = int(M["m10"] / M["m00"]) + x1, int(M["m01"] / M["m00"]) + y1
                            target_found = True

                if target_found:
                    self.state = "TRACKING"
                    self.laser_firing = True
                    self.kalman.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
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
                            self.laser_firing = False

                        if time.time() - self.lost_time < 2.0:
                            pred = self.kalman.predict()
                            px, py = int(pred[0]), int(pred[1])
                            cv2.circle(frame, (px, py), 15, (0, 0, 255), 2)
                            self.roi_rect = [max(0, px - roi_w//2), max(0, py - roi_h//2),
                                             min(w, px + roi_w//2), min(h, py + roi_h//2)]
                        else:
                            self.state = "SEARCHING"
                            self.prev_gray = None

                self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.prev_gray = cv2.GaussianBlur(self.prev_gray, (21, 21), 0)
                
            self.processed_frame = frame
            time.sleep(0.01)

    def initUI(self):
        self.setWindowTitle('Cyfrowy Bliźniak - Tryb Obserwatora (Tylko Odczyt)')
        self.resize(1100, 900) 

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_layout = QVBoxLayout()
        self.lbl_fps = QLabel("FPS: Symulacja LOKALNA")
        self.lbl_fps.setStyleSheet("color: lime; font-weight: bold; font-size: 14px;")
        left_layout.addWidget(self.lbl_fps)

        self.lbl_rgb = QLabel("Czekam na obraz sieciowy...")
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
        warning_lbl = QLabel("TRYB OBSERWATORA - PRZYCISKI NIEAKTYWNE")
        warning_lbl.setStyleSheet("color: red; font-weight: bold;")
        right_layout.addWidget(warning_lbl)

        # 1. AUDIO (Działa lokalnie na podstawie symulacji stanu lasera)
        audio_group = QGroupBox("Efekty Dźwiękowe")
        audio_layout = QHBoxLayout()
        self.btn_audio = QPushButton("🔊 Dźwięk WŁĄCZONY")
        self.btn_audio.setCheckable(True)
        self.btn_audio.setChecked(True)
        self.btn_audio.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold;")
        self.btn_audio.clicked.connect(self.toggle_audio)
        audio_layout.addWidget(self.btn_audio)
        audio_group.setLayout(audio_layout)
        right_layout.addWidget(audio_group)

        # RESZTA PRZYCISKÓW Z GUI - Zaślepione (Tylko wyglądają)
        mode_group = QGroupBox("Tryb Pracy (Zablokowane)")
        mode_layout = QHBoxLayout()
        self.radio_auto = QRadioButton("AUTOMATYCZNY")
        self.radio_manual = QRadioButton("RĘCZNY")
        self.radio_auto.setChecked(True)
        mode_layout.addWidget(self.radio_auto)
        mode_layout.addWidget(self.radio_manual)
        mode_group.setLayout(mode_layout)
        right_layout.addWidget(mode_group)

        axes_group = QGroupBox("Aktywacja Osi (Zablokowane)")
        axes_layout = QHBoxLayout()
        self.chk_pan = QCheckBox("Aktywuj PAN")
        self.chk_tilt = QCheckBox("Aktywuj TILT")
        self.chk_pan.setChecked(True)
        self.chk_tilt.setChecked(True)
        axes_layout.addWidget(self.chk_pan)
        axes_layout.addWidget(self.chk_tilt)
        axes_group.setLayout(axes_layout)
        right_layout.addWidget(axes_group)

        pid_group = QGroupBox("Nastawy PID (Zablokowane)")
        pid_layout = QVBoxLayout()
        pan_form = QFormLayout()
        self.input_kp_pan = QLineEdit("0.01")
        pan_form.addRow("PAN Kp:", self.input_kp_pan)
        tilt_form = QFormLayout()
        self.input_kp_tilt = QLineEdit("0.01")
        tilt_form.addRow("TILT Kp:", self.input_kp_tilt)
        pid_layout.addLayout(pan_form)
        pid_layout.addLayout(tilt_form)
        btn_apply_pid = QPushButton("Wyślij PID (Zablokowane)")
        pid_layout.addWidget(btn_apply_pid)
        pid_group.setLayout(pid_layout)
        right_layout.addWidget(pid_group)

        manual_group = QGroupBox("Sterowanie Ręczne (Zablokowane)")
        manual_layout = QFormLayout()
        self.input_pan = QLineEdit("0")
        btn_apply_manual = QPushButton("Wyślij Komendę Ręczną (Zablokowane)")
        manual_layout.addRow("Kąt PAN:", self.input_pan)
        manual_layout.addRow(btn_apply_manual)
        manual_group.setLayout(manual_layout)
        right_layout.addWidget(manual_group)

        color_group = QGroupBox("Cel HSV (Lokalna Pipeta)")
        color_layout = QVBoxLayout()
        self.lbl_color_patch = QLabel("Kliknij na RGB")
        self.lbl_color_patch.setFixedSize(150, 40)
        color_layout.addWidget(self.lbl_color_patch, alignment=Qt.AlignCenter)
        color_group.setLayout(color_layout)
        right_layout.addWidget(color_group)

        right_layout.addStretch()
        main_layout.addLayout(right_layout)

    def toggle_audio(self):
        self.sound_enabled = self.btn_audio.isChecked()
        if self.sound_enabled:
            self.btn_audio.setText("🔊 Dźwięk WŁĄCZONY")
            self.btn_audio.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold;")
        else:
            self.btn_audio.setText("🔇 Wyciszony (MUTE)")
            self.btn_audio.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold;")
            pygame.mixer.music.stop()

    def pipette_click(self, event):
        if hasattr(self, 'processed_frame') and self.processed_frame is not None:
            x, y = event.x(), event.y()
            try:
                b, g, r = self.processed_frame[y, x]
                h, s, v = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
                self.lbl_color_patch.setStyleSheet(f"background-color: rgb({r},{g},{b});")
                self.target_hsv = np.array([h, s, v], dtype=np.uint8)
                print(f"Lokalny cel HSV ustawiony na: {h} {s} {v}")
            except IndexError:
                pass

    def update_gui(self):
        # 1. Obsługa dźwięku na bazie lokalnie wyliczonej flagi
        if self.laser_firing and self.sound_enabled:
            if not pygame.mixer.music.get_busy():
                try:
                    pkg_dir = get_package_share_directory('pan_tilt_description')
                    laser_mp3_path = os.path.join(pkg_dir, 'gui', 'laser.mp3')
                    pygame.mixer.music.load(laser_mp3_path)
                    pygame.mixer.music.play(-1)
                except Exception:
                    pass
        elif not self.laser_firing:
            pygame.mixer.music.stop()

        # 2. Aktualizacja grafik
        if hasattr(self, 'processed_frame') and self.processed_frame is not None:
            frame = self.processed_frame
            qimg = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1], QImage.Format_BGR888)
            self.lbl_rgb.setPixmap(QPixmap.fromImage(qimg))
            
        if hasattr(self, 'mask_display'):
            mask_frame = cv2.resize(self.mask_display, (640, 240)) 
            qimg_mask = QImage(mask_frame.data, mask_frame.shape[1], mask_frame.shape[0], mask_frame.shape[1], QImage.Format_Grayscale8)
            self.lbl_mask.setPixmap(QPixmap.fromImage(qimg_mask))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = DigitalTwinGUI()
    gui.show()
    sys.exit(app.exec_())