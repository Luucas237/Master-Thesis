import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Dokładna ścieżka do Twojego wycentrowanego modelu
    urdf_file = '/workspace/src/pan_tilt_description/urdf/pan_tilt.urdf'
    
    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()

    return LaunchDescription([
        # Węzeł czytający URDF i budujący fizykę
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_desc}]
        ),
        # Odpalamy czystego RViza (bez ukrytych suwaków)
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2'
        )
    ])