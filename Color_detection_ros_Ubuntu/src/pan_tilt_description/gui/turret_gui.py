#!/usr/bin/env python3
import sys
import os
import cv2
import numpy as np
import zmq
import pygame

from ament_index_python.packages import get_package_share_directory
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, QFormLayout, QGroupBox, QRadioButton, QCheckBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

RASPBERRY_IP = "192.168.0.43"
# RASPBERRY_IP = "172.20.10.9" 

class CommandCenterGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        pygame.mixer.init()
        self.sound_enabled = True
        self.laser_firing = False

        self.context = zmq.Context()

        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://{RASPBERRY_IP}:5555")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"RGB")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"MASK")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"FPS")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"LASER_STATE")

        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.connect(f"tcp://{RASPBERRY_IP}:5556")

        self.latest_rgb_bgr = None 
        self.initUI()
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(16) 

    def initUI(self):
        self.setWindowTitle('Control Center')
        self.resize(1100, 900)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_layout = QVBoxLayout()

        self.lbl_fps = QLabel("FPS: 0")
        self.lbl_fps.setStyleSheet("color: lime; font-weight: bold; font-size: 14px;")
        left_layout.addWidget(self.lbl_fps)

        self.lbl_rgb = QLabel("Waiting for image...")
        self.lbl_rgb.setFixedSize(640, 480)
        self.lbl_rgb.setAlignment(Qt.AlignCenter)
        self.lbl_rgb.setStyleSheet("background-color: black; border: 2px solid #333;")
        self.lbl_rgb.mousePressEvent = self.pipette_click 
        left_layout.addWidget(self.lbl_rgb)

        self.lbl_mask = QLabel("Waiting for mask...")
        self.lbl_mask.setFixedSize(640, 240)
        self.lbl_mask.setAlignment(Qt.AlignCenter)
        self.lbl_mask.setStyleSheet("background-color: black; border: 2px solid #333;")
        left_layout.addWidget(self.lbl_mask)
        main_layout.addLayout(left_layout)

        right_layout = QVBoxLayout()

        #AUDIO
        audio_group = QGroupBox("Sound Control")
        audio_layout = QHBoxLayout()
        self.btn_audio = QPushButton("On")
        self.btn_audio.setCheckable(True)
        self.btn_audio.setChecked(True)
        self.btn_audio.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold;")
        self.btn_audio.clicked.connect(self.toggle_audio)
        audio_layout.addWidget(self.btn_audio)
        audio_group.setLayout(audio_layout)
        right_layout.addWidget(audio_group)

        #TRYB PRACY
        mode_group = QGroupBox("Mode")
        mode_layout = QHBoxLayout()
        self.radio_auto = QRadioButton("AUTO")
        self.radio_manual = QRadioButton("MANUAL")
        self.radio_auto.setChecked(True)
        self.radio_auto.toggled.connect(self.change_mode)
        mode_layout.addWidget(self.radio_auto)
        mode_layout.addWidget(self.radio_manual)
        mode_group.setLayout(mode_layout)
        right_layout.addWidget(mode_group)

        #AKTYWACJA OSI
        axes_group = QGroupBox("Axis Activation (For Tuning)")
        axes_layout = QHBoxLayout()
        self.chk_pan = QCheckBox("Activate PAN")
        self.chk_tilt = QCheckBox("Activate TILT")
        self.chk_pan.setChecked(True)
        self.chk_tilt.setChecked(True)
        self.chk_pan.stateChanged.connect(self.apply_axes)
        self.chk_tilt.stateChanged.connect(self.apply_axes)
        axes_layout.addWidget(self.chk_pan)
        axes_layout.addWidget(self.chk_tilt)
        axes_group.setLayout(axes_layout)
        right_layout.addWidget(axes_group)

        #NASTAWY PID
        pid_group = QGroupBox("PID Settings")
        pid_layout = QVBoxLayout()
        
        pan_form = QFormLayout()
        self.input_kp_pan = QLineEdit("0.01")
        self.input_ki_pan = QLineEdit("0.0")
        self.input_kd_pan = QLineEdit("0.001")
        pan_form.addRow("PAN Kp:", self.input_kp_pan)
        pan_form.addRow("PAN Ki:", self.input_ki_pan)
        pan_form.addRow("PAN Kd:", self.input_kd_pan)
        
        tilt_form = QFormLayout()
        self.input_kp_tilt = QLineEdit("0.01")
        self.input_ki_tilt = QLineEdit("0.0")
        self.input_kd_tilt = QLineEdit("0.001")
        tilt_form.addRow("TILT Kp:", self.input_kp_tilt)
        tilt_form.addRow("TILT Ki:", self.input_ki_tilt)
        tilt_form.addRow("TILT Kd:", self.input_kd_tilt)

        pid_layout.addLayout(pan_form)
        pid_layout.addWidget(QLabel("---"))
        pid_layout.addLayout(tilt_form)

        btn_apply_pid = QPushButton("Send PID")
        btn_apply_pid.setStyleSheet("background-color: #3498db; color: white;")
        btn_apply_pid.clicked.connect(self.apply_pid)
        pid_layout.addWidget(btn_apply_pid)
        
        pid_group.setLayout(pid_layout)
        right_layout.addWidget(pid_group)

        #STEROWANIE RĘCZNE
        manual_group = QGroupBox("Manual Control")
        manual_layout = QFormLayout()
        self.input_pan = QLineEdit("0")
        self.input_tilt = QLineEdit("0")
        self.input_laser = QLineEdit("0") # 0 lub 1
        btn_apply_manual = QPushButton("Send Manual Command")
        btn_apply_manual.setStyleSheet("background-color: #e67e22; color: white;")
        btn_apply_manual.clicked.connect(self.apply_manual)
        manual_layout.addRow("PAN Angle (-90 to 90):", self.input_pan)
        manual_layout.addRow("TILT Angle (-25 to 25):", self.input_tilt)
        manual_layout.addRow("Laser (0/1):", self.input_laser)
        manual_layout.addRow(btn_apply_manual)
        manual_group.setLayout(manual_layout)
        right_layout.addWidget(manual_group)

        #PIPETA
        color_group = QGroupBox("Target Color")
        color_layout = QVBoxLayout()
        self.lbl_color_patch = QLabel("Click on RGB")
        self.lbl_color_patch.setFixedSize(150, 40)
        color_layout.addWidget(self.lbl_color_patch, alignment=Qt.AlignCenter)
        color_group.setLayout(color_layout)
        right_layout.addWidget(color_group)

        right_layout.addStretch()
        main_layout.addLayout(right_layout)

    def toggle_audio(self):
        self.sound_enabled = self.btn_audio.isChecked()
        if self.sound_enabled:
            self.btn_audio.setText("Sound On")
            self.btn_audio.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold;")
            if self.laser_firing:
                try:
                    pygame.mixer.music.play(-1)
                except Exception as e:
                    print(f"Audio resume error: {e}")
        else:
            self.btn_audio.setText("MUTE")
            self.btn_audio.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold;")
            pygame.mixer.music.stop()

    def change_mode(self):
        mode = "AUTO" if self.radio_auto.isChecked() else "MANUAL"
        self.pub_socket.send_multipart([b"MODE", mode.encode('utf-8')])

    def apply_axes(self):
        pan_en = 1 if self.chk_pan.isChecked() else 0
        tilt_en = 1 if self.chk_tilt.isChecked() else 0
        msg = f"{pan_en} {tilt_en}"
        self.pub_socket.send_multipart([b"AXIS", msg.encode('utf-8')])

    def apply_pid(self):
        kp_p, ki_p, kd_p = self.input_kp_pan.text(), self.input_ki_pan.text(), self.input_kd_pan.text()
        kp_t, ki_t, kd_t = self.input_kp_tilt.text(), self.input_ki_tilt.text(), self.input_kd_tilt.text()
        msg = f"{kp_p} {ki_p} {kd_p} {kp_t} {ki_t} {kd_t}"
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
                
                if topic == b"LASER_STATE":
                    state = int(msg.decode('utf-8'))
                    if state == 1 and not self.laser_firing:
                        self.laser_firing = True
                        if self.sound_enabled:
                            try:
                                pkg_dir = get_package_share_directory('pan_tilt_description')
                                laser_mp3_path = os.path.join(pkg_dir, 'gui', 'laser.mp3')
                                pygame.mixer.music.load(laser_mp3_path)
                                pygame.mixer.music.play(-1)
                            except Exception as e:
                                pass
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