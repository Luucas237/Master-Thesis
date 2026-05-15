import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    pkg_share = get_package_share_directory('pan_tilt_description')
    set_model_path = SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=os.path.join(pkg_share, '..')
    )

    xacro_file = os.path.join(pkg_share, 'urdf', 'pan_tilt.urdf.xacro')
    robot_description_config = xacro.process_file(xacro_file)
    robot_urdf = robot_description_config.toxml()

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_urdf}]
    )

    world_file = os.path.join(pkg_share, 'worlds', 'target_world.sdf')
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': f'-r {world_file}'}.items()
    )

    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'pan_tilt_turret',
            '-z', '0.1',
            '-allow_renaming', 'true'
        ],
        output='screen'
    )

    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/camera/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image',

            '/model/pan_tilt_turret/joint/pan_joint/cmd_pos@std_msgs/msg/Float64]ignition.msgs.Double',

            '/model/pan_tilt_turret/joint/tilt_joint/cmd_pos@std_msgs/msg/Float64]ignition.msgs.Double',
        ],
        output='screen'
    )

    foxglove = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        parameters=[{'port': 8765}]
    )
    roi_tracker_node = Node(
        package='pan_tilt_description',
        executable='roi_detection_simulation.py',
        name='sim_roi_tracker',
        output='screen'
    )

    return LaunchDescription([
        set_model_path,
        robot_state_publisher,
        gazebo_launch,
        spawn_entity,
        gz_bridge,
        foxglove,
        roi_tracker_node
    ])
