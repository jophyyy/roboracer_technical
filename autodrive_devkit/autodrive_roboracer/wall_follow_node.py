#!/usr/bin/env python3

"""
Penn RoboRacer Autonomous Competition Controller v2.4
Zero-Jitter Straightaway Drive, Asymmetric Corridor Corner Locking, and Even Throttle.

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
        # Constant, steady 0.24 cornering speed provides continuous torque through steering scrub
        self.straight_speed = 0.32      # Controlled straightaway speed
        self.corner_speed = 0.24        # Even, steady cornering speed with strong forward drive
        self.min_drive_speed = 0.20     # Strict positive floor (never stalls, never reverses)

        # Perception & Corner Decision Thresholds
        self.max_lidar_range = 8.0      # Cap distant beams
        self.corner_entry_dist = 4.2    # Distance to front wall when cornering engages
        self.asymmetry_threshold = 0.60 # Minimum opening difference (meters) to confirm an actual turn

        # Corner Lock State
        self.in_turn = False
        self.latched_turn_dir = 0.0     # +1.0 = Left, -1.0 = Right

        # Steering smoothing (heavily damped on straights, responsive in corners)
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

        self.get_logger().info('RoboRacer Zero-Jitter Competition Controller v2.4 active.')

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

        # B) Forward-Side Corner Opening Sectors (+20 to +70 deg vs -70 to -20 deg)
        idx_l1 = self.get_index_for_angle(math.radians(20.0), angle_min, angle_inc)
        idx_l2 = self.get_index_for_angle(math.radians(70.0), angle_min, angle_inc)
        left_depth = float(np.mean(clean_ranges[min(idx_l1, idx_l2):max(idx_l1, idx_l2) + 1]))

        idx_r1 = self.get_index_for_angle(math.radians(-70.0), angle_min, angle_inc)
        idx_r2 = self.get_index_for_angle(math.radians(-20.0), angle_min, angle_inc)
        right_depth = float(np.mean(clean_ranges[min(idx_r1, idx_r2):max(idx_r1, idx_r2) + 1]))

        depth_diff = left_depth - right_depth

        # C) Direct side distances at +-90 deg
        idx_90_l = min(total_beams - 1, self.get_index_for_angle(math.radians(90.0), angle_min, angle_inc))
        idx_90_r = max(0, self.get_index_for_angle(math.radians(-90.0), angle_min, angle_inc))
        side_l = float(np.mean(clean_ranges[max(0, idx_90_l - 10):min(total_beams, idx_90_l + 11)]))
        side_r = float(np.mean(clean_ranges[max(0, idx_90_r - 10):min(total_beams, idx_90_r + 11)]))

        # 3. State Machine: Locked Turn Execution vs. Smooth Forward Drive
        if self.in_turn:
            # We are actively rotating through a corner
            if front_dist > 4.2:
                # The car has completed rotation and is looking straight down the new corridor!
                self.in_turn = False
                self.latched_turn_dir = 0.0
                raw_steering = 0.0
                throttle_cmd = self.straight_speed
            else:
                # Hold decisive full lock in the latched corner direction
                raw_steering = self.latched_turn_dir * 1.0
                # Maintain constant, even torque to pull through the turn
                throttle_cmd = self.corner_speed

        else:
            # Check if a real turn is approaching:
            # Front wall is closer than 4.2m AND there is a clear opening on one side
            if front_dist < self.corner_entry_dist and abs(depth_diff) > self.asymmetry_threshold:
                self.in_turn = True
                self.latched_turn_dir = 1.0 if depth_diff > 0 else -1.0
                raw_steering = self.latched_turn_dir * 0.90
                throttle_cmd = self.corner_speed

            else:
                # Straightaway Drive: DRIVE FORWARD (Zero steering jitter!)
                # Apply a soft boundary nudge ONLY if dangerously close to a wall (<0.32m)
                if side_l < 0.32:
                    raw_steering = -0.10  # Gentle nudge right away from left wall
                elif side_r < 0.32:
                    raw_steering = 0.10   # Gentle nudge left away from right wall
                else:
                    raw_steering = 0.0    # DEAD-STRAIGHT FORWARD DRIVE

                throttle_cmd = self.straight_speed

        # 4. Smooth steering
        steering_cmd = self.steering_smooth_alpha * raw_steering + (1.0 - self.steering_smooth_alpha) * self.prev_steering
        self.prev_steering = steering_cmd

        # 5. Strict forward throttle clamp
        throttle_cmd = float(np.clip(throttle_cmd, self.min_drive_speed, self.straight_speed))

        # Heartbeat log every 40 frames (~1 sec)
        if self.call_count % 40 == 0:
            mode = f'TURN[{self.latched_turn_dir:+.0f}]' if self.in_turn else 'STRAIGHT'
            self.get_logger().info(
                f'[LIVE v2.4] {mode} | Steer: {steering_cmd:+.2f} | Thr: {throttle_cmd:.2f} | Front: {front_dist:.2f}m | Diff: {depth_diff:+.2f}m'
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
