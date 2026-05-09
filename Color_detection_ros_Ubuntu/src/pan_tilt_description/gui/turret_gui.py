#!/usr/bin/env python3
import sys
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Int32MultiArray
from cv_bridge import CvBridge

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, QFormLayout, QGroupBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

class CommandCenterNode(Node):
    def __init__(self):
        super().__init__('command_center_gui')
        self.bridge = CvBridge()
        
        # Subskrypcje obrazów z Malinki
        self.sub_rgb = self.create_subscription(Image, '/camera/image_annotated', self.rgb_callback, 10)
        self.sub_mask = self.create_subscription(Image, '/camera/image_mask', self.mask_callback, 10)
        
        # Publikowanie parametrów (PID i docelowy kolor HSV)
        self.pub_pid = self.create_publisher(Float32MultiArray, '/turret/pid_params', 10)
        self.pub_color = self.create_publisher(Int32MultiArray, '/turret/target_color_hsv', 10)

        # Bufory na najnowsze klatki
        self.latest_rgb = None
        self.latest_mask = None

    def rgb_callback(self, msg):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def mask_callback(self, msg):
        self.latest_mask = self.bridge.imgmsg_to_cv2(msg, "mono8")

    def send_pid(self, kp, ki, kd):
        msg = Float32MultiArray()
        msg.data = [float(kp), float(ki), float(kd)]
        self.pub_pid.publish(msg)
        self.get_logger().info(f"Wyslano nowe PID: Kp={kp}, Ki={ki}, Kd={kd}")

    def send_color(self, h, s, v):
        msg = Int32MultiArray()
        msg.data = [int(h), int(s), int(v)]
        self.pub_color.publish(msg)
        self.get_logger().info(f"Wyslano nowy cel (HSV): {h}, {s}, {v}")


class CommandCenterGUI(QMainWindow):
    def __init__(self, ros_node):
        super().__init__()
        self.ros_node = ros_node
        self.initUI()
        
        # Timer odświeżający GUI (30 FPS)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(30)

    def initUI(self):
        self.setWindowTitle('Wieżyczka - Centrum Dowodzenia')
        self.resize(1000, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # === LEWA STRONA (Obrazy) ===
        left_layout = QVBoxLayout()
        
        # Etykieta na obraz RGB
        self.lbl_rgb = QLabel("Czekam na obraz RGB...")
        self.lbl_rgb.setFixedSize(640, 480)
        self.lbl_rgb.setAlignment(Qt.AlignCenter)
        self.lbl_rgb.setStyleSheet("background-color: black; color: white; border: 2px solid gray;")
        self.lbl_rgb.mousePressEvent = self.pipette_click # Podpięcie narzędzia "Pipeta"
        left_layout.addWidget(self.lbl_rgb)

        # Etykieta na maskę
        self.lbl_mask = QLabel("Czekam na Maskę...")
        self.lbl_mask.setFixedSize(640, 240)
        self.lbl_mask.setAlignment(Qt.AlignCenter)
        self.lbl_mask.setStyleSheet("background-color: black; color: white; border: 2px solid gray;")
        left_layout.addWidget(self.lbl_mask)

        main_layout.addLayout(left_layout)

        # === PRAWA STRONA (Panel sterowania) ===
        right_layout = QVBoxLayout()
        
        # Panel PID
        pid_group = QGroupBox("Nastawy Regulatora PID")
        pid_layout = QFormLayout()
        
        self.input_kp = QLineEdit("0.001")
        self.input_ki = QLineEdit("0.00001")
        self.input_kd = QLineEdit("0.0005")
        
        pid_layout.addRow("Kp:", self.input_kp)
        pid_layout.addRow("Ki:", self.input_ki)
        pid_layout.addRow("Kd:", self.input_kd)
        
        btn_apply_pid = QPushButton("Wyślij nowe PID")
        btn_apply_pid.setStyleSheet("background-color: #2a82da; color: white; font-weight: bold;")
        btn_apply_pid.clicked.connect(self.apply_pid)
        pid_layout.addRow(btn_apply_pid)
        
        pid_group.setLayout(pid_layout)
        right_layout.addWidget(pid_group)

        # Panel Koloru (Pipeta)
        color_group = QGroupBox("Wybrany Cel (Pipeta)")
        color_layout = QVBoxLayout()
        self.lbl_picked_color = QLabel("Kliknij na obraz RGB, aby wybrać cel")
        self.lbl_picked_color.setAlignment(Qt.AlignCenter)
        self.lbl_color_patch = QLabel()
        self.lbl_color_patch.setFixedSize(100, 50)
        self.lbl_color_patch.setStyleSheet("background-color: gray;")
        
        color_layout.addWidget(self.lbl_picked_color)
        color_layout.addWidget(self.lbl_color_patch, alignment=Qt.AlignCenter)
        color_group.setLayout(color_layout)
        right_layout.addWidget(color_group)

        right_layout.addStretch() # Wypycha panele do góry
        main_layout.addLayout(right_layout)

    def pipette_click(self, event):
        # Funkcja wywoływana po kliknięciu myszką w obraz RGB
        if self.ros_node.latest_rgb is not None:
            x, y = event.x(), event.y()
            
            # Pobranie koloru piksela (BGR)
            try:
                b, g, r = self.ros_node.latest_rgb[y, x]
                
                # Konwersja na HSV
                pixel_bgr = np.uint8([[[b, g, r]]])
                h, s, v = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)[0][0]
                
                # Aktualizacja GUI
                self.lbl_picked_color.setText(f"Wybrano HSV: {h}, {s}, {v}")
                self.lbl_color_patch.setStyleSheet(f"background-color: rgb({r},{g},{b});")
                
                # Wysyłka do Malinki
                self.ros_node.send_color(h, s, v)
            except IndexError:
                pass # Kliknięcie poza krawędź tablicy (np. przy dziwnym skalowaniu okna)

    def apply_pid(self):
        kp = self.input_kp.text()
        ki = self.input_ki.text()
        kd = self.input_kd.text()
        self.ros_node.send_pid(kp, ki, kd)

    def update_gui(self):
        # Przetworzenie zdarzeń ROS 2 (odbieranie wiadomości)
        rclpy.spin_once(self.ros_node, timeout_sec=0)

        # Odświeżenie obrazu RGB
        if self.ros_node.latest_rgb is not None:
            rgb_frame = cv2.cvtColor(self.ros_node.latest_rgb, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            qimg = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
            self.lbl_rgb.setPixmap(QPixmap.fromImage(qimg))

        # Odświeżenie obrazu maski
        if self.ros_node.latest_mask is not None:
            mask_frame = self.ros_node.latest_mask
            # Skalowanie maski, żeby zmieściła się w mniejszym oknie na dole
            mask_resized = cv2.resize(mask_frame, (640, 240)) 
            h, w = mask_resized.shape
            qimg = QImage(mask_resized.data, w, h, w, QImage.Format_Grayscale8)
            self.lbl_mask.setPixmap(QPixmap.fromImage(qimg))

def main(args=None):
    rclpy.init(args=args)
    
    # Inicjalizacja aplikacji Qt
    app = QApplication(sys.argv)
    
    # Tworzenie węzła ROS i przekazanie go do okna GUI
    ros_node = CommandCenterNode()
    gui = CommandCenterGUI(ros_node)
    gui.show()
    
    # Uruchomienie głównej pętli okna
    sys.exit(app.exec_())

    # Zamykanie
    ros_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()