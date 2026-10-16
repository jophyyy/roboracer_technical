#!/usr/bin/env python3

"""
Penn RoboRacer Autonomous Competition Controller v2.3
Proactive Pre-Turn Steering, Even High-Traction Throttle, and Centered Corridor Tracking.

Subscribes:
  /autodrive/roboracer_1/lidar (sensor_msgs/LaserScan)

Publishes:
  /autodrive/roboracer_1/throttle_command (std_msgs/Float32)
  /autodrive/roboracer_1/steering_command (std_msgs/Float32)
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32


class WallFollowNode(Node):
    def __init__(self):
        super().__init__('wall_follow_node')

        # Speed Settings
        # Even cornering speed (0.24) delivers steady torque to overcome steering scrub without bogging down
        self.straight_speed = 0.35      # Swift, stable straightaway pace
        self.corner_speed = 0.24        # Even, steady cornering speed with strong forward drive
        self.min_drive_speed = 0.22     # Positive floor (never stalls, never reverses)

        # Lookahead & Turn Initiation
        self.turn_anticipation_dist = 4.8 # Begin turning 4.8m before reaching the front wall
        self.full_turn_dist = 2.2         # Reach full steering lock by 2.2m
        self.max_lidar_range = 8.0        # Cap distant beams

        # Active turn direction memory (+1.0 = Left, -1.0 = Right)
        self.active_turn_dir = 0.0

        # Steering smoothing (fast response to command early turn angle)
        self.prev_steering = 0.0
        self.steering_smooth_alpha = 0.88

        # Telemetry counter
        self.call_count = 0

        # Reliable QoS matching bridge
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/autodrive/roboracer_1/lidar',
            self.lidar_callback,
            qos
        )

        self.throttle_pub = self.create_publisher(
            Float32,
            '/autodrive/roboracer_1/throttle_command',
            1
        )

        self.steering_pub = self.create_publisher(
            Float32,
            '/autodrive/roboracer_1/steering_command',
            1
        )

        self.get_logger().info('RoboRacer Proactive Cornering Controller v2.3 active.')

    def get_index_for_angle(self, angle_rad, angle_min, angle_inc):
        return int(round((angle_rad - angle_min) / angle_inc))

    def preprocess_lidar(self, raw_ranges):
        """Clean sensor data: eliminate self-chassis reflections (<0.22m), NaNs, and infs."""
        proc = np.array(raw_ranges, dtype=np.float32)
        proc[np.isnan(proc)] = self.max_lidar_range
        proc[np.isinf(proc)] = self.max_lidar_range
        proc[proc < 0.22] = self.max_lidar_range  # Ignore self chassis
        proc = np.clip(proc, 0.0, self.max_lidar_range)

        # 5-beam moving average filter to remove noise
        kernel = np.ones(5) / 5.0
        proc = np.convolve(proc, kernel, mode='same')
        return proc

    def lidar_callback(self, scan_msg: LaserScan):
        self.call_count += 1
        angle_min = scan_msg.angle_min
        angle_inc = scan_msg.angle_increment
        total_beams = len(scan_msg.ranges)

        if total_beams == 0:
            return

        # 1. Clean raw LiDAR
        clean_ranges = self.preprocess_lidar(scan_msg.ranges)

        # 2. Key directional sectors:
        # A) Forward Cone (+-15 degrees)
        idx_f_l = self.get_index_for_angle(math.radians(15.0), angle_min, angle_inc)
        idx_f_r = self.get_index_for_angle(math.radians(-15.0), angle_min, angle_inc)
        forward_dist = float(np.min(clean_ranges[min(idx_f_r, idx_f_l):max(idx_f_r, idx_f_l) + 1]))

        # B) Forward-Side Corner Opening Sectors (+15 to +65 deg vs -65 to -15 deg)
        idx_l1 = self.get_index_for_angle(math.radians(15.0), angle_min, angle_inc)
        idx_l2 = self.get_index_for_angle(math.radians(65.0), angle_min, angle_inc)
        left_depth = float(np.mean(clean_ranges[min(idx_l1, idx_l2):max(idx_l1, idx_l2) + 1]))

        idx_r1 = self.get_index_for_angle(math.radians(-65.0), angle_min, angle_inc)
        idx_r2 = self.get_index_for_angle(math.radians(-15.0), angle_min, angle_inc)
        right_depth = float(np.mean(clean_ranges[min(idx_r1, idx_r2):max(idx_r1, idx_r2) + 1]))

        # C) Side Distances at +-90 deg for corridor centering
        idx_90_l = min(total_beams - 1, self.get_index_for_angle(math.radians(90.0), angle_min, angle_inc))
        idx_90_r = max(0, self.get_index_for_angle(math.radians(-90.0), angle_min, angle_inc))
        side_l = float(np.mean(clean_ranges[max(0, idx_90_l - 10):min(total_beams, idx_90_l + 11)]))
        side_r = float(np.mean(clean_ranges[max(0, idx_90_r - 10):min(total_beams, idx_90_r + 11)]))

        # 3. Decision Logic: Proactive Cornering vs. Straightaway Centering
        if forward_dist < self.turn_anticipation_dist:
            # We are approaching, entering, or actively in a turn!
            # Determine/maintain turn direction:
            if left_depth > right_depth:
                self.active_turn_dir = 1.0   # Left turn
            else:
                self.active_turn_dir = -1.0  # Right turn

            # Turn intensity: immediately apply 0.50 steering at 4.8m, scaling smoothly up to 1.0 full lock at 2.2m
            progress = float(np.clip((self.turn_anticipation_dist - forward_dist) / (self.turn_anticipation_dist - self.full_turn_dist), 0.0, 1.0))
            raw_steering = self.active_turn_dir * (0.50 + 0.50 * progress)

            # Even, constant forward speed throughout cornering to overcome tire scrub
            throttle_cmd = self.corner_speed

        else:
            # Clear Straightaway: Maintain perfect track centerline
            self.active_turn_dir = 0.0
            wall_error = side_l - side_r
            raw_steering = float(np.clip(wall_error * 0.14, -0.25, 0.25))

            # Accelerate down straight
            throttle_cmd = self.straight_speed

        # 4. Smooth steering to protect steering servo while responding rapidly
        steering_cmd = self.steering_smooth_alpha * raw_steering + (1.0 - self.steering_smooth_alpha) * self.prev_steering
        self.prev_steering = steering_cmd

        # 5. Strictly clamp throttle: never stall, never reverse!
        throttle_cmd = float(np.clip(throttle_cmd, self.min_drive_speed, self.straight_speed))

        # Heartbeat log every 40 frames (~1 sec)
        if self.call_count % 40 == 0:
            self.get_logger().info(
                f'[LIVE v2.3] Steer: {steering_cmd:+.2f} | Throttle: {throttle_cmd:.2f} | Front: {forward_dist:.2f}m'
            )

        # 6. Publish actuator commands
        str_msg = Float32()
        str_msg.data = float(steering_cmd)
        self.steering_pub.publish(str_msg)

        thr_msg = Float32()
        thr_msg.data = float(throttle_cmd)
        self.throttle_pub.publish(thr_msg)


def main(args=None):
    rclpy.init(args=args)
    node = WallFollowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
