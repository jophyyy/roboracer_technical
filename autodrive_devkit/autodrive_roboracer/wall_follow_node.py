#!/usr/bin/env python3

"""Deterministic Porto-track waypoint follower for AutoDRIVE RoboRacer.

The path is built from the ordered Porto checkpoint transforms embedded in the
simulator. Vehicle position and orientation come from ground-truth odometry;
LiDAR is deliberately not used.
"""

import math

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Bool, Float32


# Ordered checkpoint centers extracted from the Porto scene. These values are
# in Unity's horizontal (X, Z) plane and are converted to the bridge's ROS
# frame below: ros_x = unity_z, ros_y = -unity_x.
PORTO_UNITY_CHECKPOINTS = np.asarray(
    [
        (-3.610000, 0.881000),
        (-2.316999, 0.778000),
        (-0.845999, 0.700000),
        (0.620001, 0.698000),
        (2.040001, 0.600000),
        (3.630002, 0.536000),
        (5.180002, 0.374000),
        (6.740002, 0.000000),
        (7.367002, -1.413001),
        (6.673004, -2.657001),
        (5.610001, -3.020001),
        (4.381001, -2.556001),
        (3.282001, -1.752001),
        (2.000000, -1.200001),
        (0.540001, -1.290001),
        (-0.879999, -1.830000),
        (-2.270000, -2.390001),
        (-3.810000, -2.620001),
        (-5.100000, -2.340002),
        (-5.940001, -0.960001),
        (-5.240001, 0.420000),
    ],
    dtype=np.float64,
)
PORTO_CHECKPOINTS = np.column_stack(
    (PORTO_UNITY_CHECKPOINTS[:, 1], -PORTO_UNITY_CHECKPOINTS[:, 0])
)


class WallFollowNode(Node):
    """Follow the fixed Porto checkpoint loop using pure pursuit."""

    def __init__(self):
        super().__init__('wall_follow_node')

        self.declare_parameter('lookahead_distance', 0.70)
        self.declare_parameter('brake_lookahead_distance', 4.00)
        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('max_steering_angle', 0.55)
        self.declare_parameter('straight_throttle', 0.11)
        self.declare_parameter('corner_throttle', 0.11)
        self.declare_parameter('path_spacing', 0.08)
        self.declare_parameter('steering_smoothing', 0.85)
        self.declare_parameter('throttle_smoothing', 0.65)

        spacing = self.get_parameter('path_spacing').value
        self.path = self._densify_closed_path(PORTO_CHECKPOINTS, spacing)
        self.prev_steering = 0.0
        self.prev_throttle = 0.0
        self.last_log_time = 0.0

        qos = QoSProfile(
            durability=QoSDurabilityPolicy.VOLATILE,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.throttle_pub = self.create_publisher(
            Float32, '/autodrive/roboracer_1/throttle_command', 1
        )
        self.steering_pub = self.create_publisher(
            Float32, '/autodrive/roboracer_1/steering_command', 1
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            '/autodrive/roboracer_1/odom',
            self.odom_callback,
            qos,
        )
        self.reset_sub = self.create_subscription(
            Bool, '/autodrive/reset_command', self.reset_callback, 1
        )

        self.get_logger().info(
            f'Fixed Porto waypoint controller active ({len(self.path)} path points).'
        )

    @staticmethod
    def _densify_closed_path(points, spacing):
        """Linearly densify the closed checkpoint polygon."""
        spacing = max(0.03, float(spacing))
        dense = []
        for index in range(len(points)):
            start = points[index]
            end = points[(index + 1) % len(points)]
            distance = float(np.linalg.norm(end - start))
            steps = max(1, int(math.ceil(distance / spacing)))
            for step in range(steps):
                fraction = step / steps
                dense.append(start + fraction * (end - start))
        return np.asarray(dense, dtype=np.float64)

    @staticmethod
    def _ros_yaw(quaternion):
        """Return planar ROS yaw from the odometry quaternion."""
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w
        return math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

    def reset_callback(self, msg):
        if msg.data:
            self.prev_steering = 0.0
            self.prev_throttle = 0.0
            self.get_logger().info('Reset received: waypoint controller state cleared.')

    def _lookahead_target(self, position, lookahead):
        """Find the nearest path point, then walk forward by lookahead metres."""
        deltas = self.path - position
        nearest_index = int(np.argmin(np.einsum('ij,ij->i', deltas, deltas)))

        target_index = nearest_index
        distance_walked = 0.0
        while distance_walked < lookahead:
            next_index = (target_index + 1) % len(self.path)
            distance_walked += float(
                np.linalg.norm(self.path[next_index] - self.path[target_index])
            )
            target_index = next_index
        return nearest_index, target_index, self.path[target_index]

    def odom_callback(self, msg):
        position = np.asarray(
            [msg.pose.pose.position.x, msg.pose.pose.position.y],
            dtype=np.float64,
        )
        yaw = self._ros_yaw(msg.pose.pose.orientation)

        lookahead = float(self.get_parameter('lookahead_distance').value)
        brake_lookahead = float(
            self.get_parameter('brake_lookahead_distance').value
        )
        wheelbase = float(self.get_parameter('wheelbase').value)
        max_steering_angle = float(
            self.get_parameter('max_steering_angle').value
        )
        straight_throttle = float(
            self.get_parameter('straight_throttle').value
        )
        corner_throttle = float(self.get_parameter('corner_throttle').value)
        steering_smoothing = float(
            self.get_parameter('steering_smoothing').value
        )
        throttle_smoothing = float(
            self.get_parameter('throttle_smoothing').value
        )

        nearest_index, target_index, target = self._lookahead_target(
            position, lookahead
        )
        delta_x = float(target[0] - position[0])
        delta_z = float(target[1] - position[1])

        # Transform into the ROS vehicle frame: +X forward and +Y left.
        local_forward = math.cos(yaw) * delta_x + math.sin(yaw) * delta_z
        local_left = -math.sin(yaw) * delta_x + math.cos(yaw) * delta_z
        target_distance = max(0.05, math.hypot(local_forward, local_left))

        # If localization briefly puts the chosen point behind the car, advance
        # farther around the fixed path rather than reversing steering.
        attempts = 0
        while local_forward <= 0.05 and attempts < len(self.path):
            target_index = (target_index + 1) % len(self.path)
            target = self.path[target_index]
            delta_x = float(target[0] - position[0])
            delta_z = float(target[1] - position[1])
            local_forward = math.cos(yaw) * delta_x + math.sin(yaw) * delta_z
            local_left = -math.sin(yaw) * delta_x + math.cos(yaw) * delta_z
            target_distance = max(0.05, math.hypot(local_forward, local_left))
            attempts += 1

        curvature = 2.0 * local_left / (target_distance * target_distance)
        steering_angle = math.atan(wheelbase * curvature)

        # AutoDRIVE command convention is positive left and negative right.
        target_steering = float(
            np.clip(steering_angle / max_steering_angle, -1.0, 1.0)
        )
        steering_smoothing = float(np.clip(steering_smoothing, 0.0, 1.0))
        steering_cmd = (
            steering_smoothing * target_steering
            + (1.0 - steering_smoothing) * self.prev_steering
        )
        steering_cmd = float(np.clip(steering_cmd, -1.0, 1.0))
        self.prev_steering = steering_cmd

        # Preview farther down the fixed path for braking. This intentionally
        # uses the unsmoothed future steering demand so deceleration begins
        # before the steering actuator reaches the corner.
        _, _, brake_target = self._lookahead_target(position, brake_lookahead)
        brake_delta_x = float(brake_target[0] - position[0])
        brake_delta_y = float(brake_target[1] - position[1])
        brake_forward = (
            math.cos(yaw) * brake_delta_x + math.sin(yaw) * brake_delta_y
        )
        brake_left = (
            -math.sin(yaw) * brake_delta_x + math.cos(yaw) * brake_delta_y
        )
        brake_distance = max(0.05, math.hypot(brake_forward, brake_left))
        brake_curvature = 2.0 * brake_left / (brake_distance * brake_distance)
        brake_angle = math.atan(wheelbase * brake_curvature)
        preview_steering = float(
            np.clip(brake_angle / max_steering_angle, -1.0, 1.0)
        )

        turn_amount = min(
            1.0,
            max(abs(target_steering), 1.25 * abs(preview_steering)),
        )
        target_throttle = (
            straight_throttle
            - turn_amount * (straight_throttle - corner_throttle)
        )
        throttle_smoothing = float(np.clip(throttle_smoothing, 0.0, 1.0))
        throttle_cmd = (
            throttle_smoothing * target_throttle
            + (1.0 - throttle_smoothing) * self.prev_throttle
        )
        throttle_cmd = float(
            np.clip(throttle_cmd, corner_throttle, straight_throttle)
        )
        self.prev_throttle = throttle_cmd

        steering_msg = Float32()
        steering_msg.data = steering_cmd
        self.steering_pub.publish(steering_msg)

        throttle_msg = Float32()
        throttle_msg.data = throttle_cmd
        self.throttle_pub.publish(throttle_msg)

        now_time = self.get_clock().now().nanoseconds * 1e-9
        if now_time - self.last_log_time >= 0.5:
            self.last_log_time = now_time
            self.get_logger().info(
                f'WP nearest={nearest_index:03d} target={target_index:03d} | '
                f'pos=({position[0]:+.2f},{position[1]:+.2f}) | '
                f'local=(left={local_left:+.2f},fwd={local_forward:+.2f}) | '
                f'steer={steering_cmd:+.2f} preview={preview_steering:+.2f} '
                f'thr={throttle_cmd:.2f}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = WallFollowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
