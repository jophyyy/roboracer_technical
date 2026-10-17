#!/usr/bin/env python3

"""
Penn RoboRacer Autonomous Competition Controller v2.5
Early Proactive Turn Commitment, Symmetric Straightaway Stability, and Smooth Forward Drive.

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

        # Speed settings:
        # Calibrated straight speed (0.28) prevents excessive kinetic energy build-up
        # Corner speed (0.22) delivers continuous motor torque through steering scrub
        self.straight_speed = 0.28      # Controlled straightaway speed
        self.corner_speed = 0.22        # Steady forward cornering speed (full tire grip)
        self.min_drive_speed = 0.18     # Strict positive floor (never stalls, never reverses)

        # Lookahead & Corner Decision Settings
        self.max_lidar_range = 8.0      # Cap distant beams
        self.turn_initiate_dist = 3.8   # Initiate turn when front wall is 3.8m away
        self.turn_lock_dist = 2.0       # Full steering lock reached by 2.0m
        self.opening_diff_thresh = 0.40 # Threshold (meters) to confirm an asymmetric corridor opening

        # Turn state tracking
        self.in_turn = False
        self.latched_turn_dir = 0.0     # +1.0 = Left, -1.0 = Right

        # Steering smoothing filter
        self.prev_steering = 0.0
        self.steering_smooth_alpha = 0.85

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

        self.get_logger().info('RoboRacer Competition Controller v2.5 active.')

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
        # A) Front cone (+-12 degrees)
        idx_f_l = self.get_index_for_angle(math.radians(12.0), angle_min, angle_inc)
        idx_f_r = self.get_index_for_angle(math.radians(-12.0), angle_min, angle_inc)
        front_dist = float(np.min(clean_ranges[min(idx_f_r, idx_f_l):max(idx_f_r, idx_f_l) + 1]))

        # B) Forward-Diagonal Sectors (+10 to +50 deg vs -50 to -10 deg)
        idx_l1 = self.get_index_for_angle(math.radians(10.0), angle_min, angle_inc)
        idx_l2 = self.get_index_for_angle(math.radians(50.0), angle_min, angle_inc)
        left_diag = float(np.mean(clean_ranges[min(idx_l1, idx_l2):max(idx_l1, idx_l2) + 1]))

        idx_r1 = self.get_index_for_angle(math.radians(-50.0), angle_min, angle_inc)
        idx_r2 = self.get_index_for_angle(math.radians(-10.0), angle_min, angle_inc)
        right_diag = float(np.mean(clean_ranges[min(idx_r1, idx_r2):max(idx_r1, idx_r2) + 1]))

        opening_diff = left_diag - right_diag

        # C) Direct side distances at +-90 deg
        idx_90_l = min(total_beams - 1, self.get_index_for_angle(math.radians(90.0), angle_min, angle_inc))
        idx_90_r = max(0, self.get_index_for_angle(math.radians(-90.0), angle_min, angle_inc))
        side_l = float(np.mean(clean_ranges[max(0, idx_90_l - 10):min(total_beams, idx_90_l + 11)]))
        side_r = float(np.mean(clean_ranges[max(0, idx_90_r - 10):min(total_beams, idx_90_r + 11)]))

        # 3. Decision Logic:
        if self.in_turn:
            # Active in a turn: hold turn authority until front clearance opens up into the new straightaway
            if front_dist > 4.2:
                self.in_turn = False
                self.latched_turn_dir = 0.0
                raw_steering = 0.0
                throttle_cmd = self.straight_speed
            else:
                # Hold full lock in the latched corner direction
                raw_steering = self.latched_turn_dir * 1.0
                throttle_cmd = self.corner_speed

        else:
            # Check for approaching turn:
            # Front wall approaching (<3.8m) AND clear asymmetric opening to one side
            if front_dist < self.turn_initiate_dist and abs(opening_diff) > self.opening_diff_thresh:
                self.in_turn = True
                self.latched_turn_dir = 1.0 if opening_diff > 0 else -1.0
                # Early turn authority: starts decisively at 0.75, scaling to 1.0
                progress = float(np.clip((self.turn_initiate_dist - front_dist) / (self.turn_initiate_dist - self.turn_lock_dist), 0.0, 1.0))
                raw_steering = self.latched_turn_dir * (0.75 + 0.25 * progress)
                throttle_cmd = self.corner_speed

            else:
                # Straightaway Drive: DRIVE FORWARD (Zero oscillation!)
                # Only gentle buffer if dangerously close to a wall (<0.32m)
                if side_l < 0.32:
                    raw_steering = -0.10
                elif side_r < 0.32:
                    raw_steering = 0.10
                else:
                    raw_steering = 0.0

                # Decelerate smoothly approaching turn
                if front_dist < 5.0:
                    alpha = (front_dist - self.turn_initiate_dist) / (5.0 - self.turn_initiate_dist)
                    throttle_cmd = self.corner_speed + alpha * (self.straight_speed - self.corner_speed)
                else:
                    throttle_cmd = self.straight_speed

        # 4. Smooth steering command
        steering_cmd = self.steering_smooth_alpha * raw_steering + (1.0 - self.steering_smooth_alpha) * self.prev_steering
        self.prev_steering = steering_cmd

        # 5. Strict forward throttle clamp
        throttle_cmd = float(np.clip(throttle_cmd, self.min_drive_speed, self.straight_speed))

        # Heartbeat log every 40 frames (~1 sec)
        if self.call_count % 40 == 0:
            mode = f'TURN[{self.latched_turn_dir:+.0f}]' if self.in_turn else 'STRAIGHT'
            self.get_logger().info(
                f'[LIVE v2.5] {mode} | Steer: {steering_cmd:+.2f} | Thr: {throttle_cmd:.2f} | Front: {front_dist:.2f}m | Diff: {opening_diff:+.2f}m'
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
