#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import math
import time

class TurretWobbler(Node):
    def __init__(self):
        super().__init__('turret_controller')
        # Tworzymy "megafon", przez który będziemy krzyczeć do RViz, pod jakim kątem są silniki
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)
        
        # Pętla wykonująca się 30 razy na sekundę (30 Hz)
        self.timer = self.create_timer(0.033, self.timer_callback)
        self.start_time = time.time()

    def timer_callback(self):
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # UWAGA: Nazwy muszą się zgadzać z tymi z pliku pan_tilt.urdf!
        msg.name = ['pan_joint', 'tilt_joint']

        # Czas od uruchomienia programu
        t = time.time() - self.start_time
        
        # Generujemy płynny ruch za pomocą funkcji sinus
        # Oś Pan (lewo-prawo): obrót o ~1.5 radiana w obie strony
        pan_angle = math.sin(t) * 1.5      
        
        # Oś Tilt (góra-dół): kręci się 2x szybciej (t*2) i wychyla o 0.8 radiana
        tilt_angle = math.sin(t * 2.0) * 0.8 

        # Pakujemy kąty do wiadomości i wysyłamy
        msg.position = [pan_angle, tilt_angle]
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = TurretWobbler()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()