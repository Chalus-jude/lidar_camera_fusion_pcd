#!/usr/bin/env python3

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('lidar_camera_fusion')
    default_params = os.path.join(pkg_share, 'config', 'fusion_params.yaml')

    calibration_file_arg = DeclareLaunchArgument(
        'calibration_file',
        description='Path to calibration YAML (camera_matrix, dist_coeffs, '
                    'projection_matrix [optional], T_cam_lidar). Required.',
    )
    lidar_topic_arg = DeclareLaunchArgument(
        'lidar_topic',
        default_value='/ouster/points',
        description='PointCloud2 topic to project into depth images.',
    )
    rgb_topic_arg = DeclareLaunchArgument(
        'rgb_topic',
        default_value='',
        description='sensor_msgs/CompressedImage topic. OPT-IN: leave empty '
                    'for LiDAR-only output (/lidar_depth/sparse, '
                    '/lidar_depth/dense, no image subscription at all). Set '
                    'this to also publish /lidar_depth/dense_rgb_fused via '
                    'ApproximateTimeSynchronizer -- reintroduces the RGB/'
                    'LiDAR sync timing surface, off by default on purpose.',
    )
    qos_reliability_arg = DeclareLaunchArgument(
        'qos_reliability',
        default_value='reliable',
        description="'reliable' or 'best_effort'. Should match what "
                    'qos_override_file declares for the topics being played '
                    'back, or messages may be silently dropped.',
    )
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params,
        description='Path to fusion node parameter YAML.',
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use /clock from bag playback instead of wall time.',
    )

    play_bag_arg = DeclareLaunchArgument(
        'play_bag',
        default_value='false',
        description='If true, also launch `ros2 bag play` for the given bag_path.',
    )
    bag_path_arg = DeclareLaunchArgument(
        'bag_path',
        default_value='',
        description='Path to the rosbag2 directory (e.g. ~/garage_roads1). '
                    'Only used when play_bag:=true.',
    )
    qos_override_file_arg = DeclareLaunchArgument(
        'qos_override_file',
        default_value='',
        description='Path to the QoS override YAML needed to expose '
                    '/ouster/points and the RGB compressed topic during '
                    'playback. Only used when play_bag:=true.',
    )
    read_ahead_queue_size_arg = DeclareLaunchArgument(
        'read_ahead_queue_size',
        default_value='5000',
        description='ros2 bag play --read-ahead-queue-size. Raise this (and/or '
                    'lower --rate) if you see "Message queue starved" warnings, '
                    'which usually mean disk read (especially over USB/external '
                    'drives) cannot keep up with playback of large messages '
                    '(JPEG frames, PointCloud2). Only used when play_bag:=true.',
    )
    playback_rate_arg = DeclareLaunchArgument(
        'playback_rate',
        default_value='1.0',
        description='ros2 bag play --rate. Lower this if increasing '
                    'read_ahead_queue_size alone does not clear starvation '
                    'warnings. Only used when play_bag:=true.',
    )
    bag_start_delay_arg = DeclareLaunchArgument(
        'bag_start_delay',
        default_value='3.0',
        description='Seconds to wait after fusion_node starts before '
                    'starting bag playback. fusion_node needs time to import, '
                    'load calibration, and register its subscription before '
                    'the first messages arrive -- without this delay, '
                    'playback can start (and on short bags, finish) before '
                    'the node is actually subscribed, silently dropping '
                    'every message. Only used when play_bag:=true.',
    )
    fusion_node = Node(
        package='lidar_camera_fusion',
        executable='fusion_node',
        name='lidar_camera_fusion_node',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'calibration_file': LaunchConfiguration('calibration_file'),
                'lidar_topic': LaunchConfiguration('lidar_topic'),
                'rgb_topic': LaunchConfiguration('rgb_topic'),
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'qos_reliability': LaunchConfiguration('qos_reliability'),
            },
        ],
    )

    bag_play_cmd = [
        'ros2', 'bag', 'play',
        LaunchConfiguration('bag_path'),
        '--clock',
        '--rate', LaunchConfiguration('playback_rate'),
        '--read-ahead-queue-size', LaunchConfiguration('read_ahead_queue_size'),
    ]

    bag_play = ExecuteProcess(
        cmd=bag_play_cmd + [
            '--qos-profile-overrides-path', LaunchConfiguration('qos_override_file'),
        ],
        output='screen',
    )

    delayed_bag_play = TimerAction(
        period=LaunchConfiguration('bag_start_delay'),
        actions=[bag_play],
        condition=IfCondition(LaunchConfiguration('play_bag')),
    )

    return LaunchDescription([
        calibration_file_arg,
        lidar_topic_arg,
        rgb_topic_arg,
        qos_reliability_arg,
        params_file_arg,
        use_sim_time_arg,
        play_bag_arg,
        bag_path_arg,
        qos_override_file_arg,
        read_ahead_queue_size_arg,
        playback_rate_arg,
        bag_start_delay_arg,
        fusion_node,
        delayed_bag_play,
    ])
