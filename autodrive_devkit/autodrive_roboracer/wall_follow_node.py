#!/usr/bin/env python3

"""
Penn RoboRacer Autonomous Competition Controller
Multi-Phase Corner Anticipation, Turn Latch, and Safe Drivable Gap Following.

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

        # Speed Profiles (calibrated for high stability and fast lap times)
        self.straight_speed = 0.35      # Swift straightaway speed
        self.corner_speed = 0.20        # Safe, high-traction cornering speed
        self.min_drive_speed = 0.18     # Minimum forward throttle (guaranteed no stall)

        # Perception Geometry
        self.max_lidar_range = 8.0      # Cap distant beams
        self.forward_fov_deg = 200.0    # Look wide (+-100 deg) to detect 90-deg corridors
        self.max_steer_rad = 0.52       # Max steering limit (~30 degrees)
        self.safety_bubble_radius = 0.40 # Wall safety clearance margin (meters)

        # Lookahead & Early Turn Initiation
        self.early_lookahead_dist = 5.2 # Begin decelerating 5.2m before front wall
        self.corner_turn_dist = 4.0     # Begin steering 4.0m in advance of the corner

        # Corner Latch State (prevents premature wheel straightening mid-turn)
        self.in_corner = False
        self.latched_turn_dir = 0.0     # +1.0 = Left, -1.0 = Right

        # Steering smoothing filter
        self.prev_steering = 0.0
        self.steering_smooth_alpha = 0.80

        # Reliable QoS matching bridge
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers & Publishers
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

        self.get_logger().info('RoboRacer High-Precision Cornering Controller active.')

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

    def apply_safety_bubble(self, ranges, angle_min, angle_inc):
        """Inflates closest obstacle by safety_bubble_radius to prevent clipping corner apexes."""
        out = np.copy(ranges)
        min_idx = int(np.argmin(out))
        min_dist = float(out[min_idx])

        if min_dist < 0.80 and min_dist > 0.05:
            half_bubble_rad = math.atan2(self.safety_bubble_radius, min_dist)
            beams_to_mask = int(math.ceil(half_bubble_rad / angle_inc))
            start_idx = max(0, min_idx - beams_to_mask)
            end_idx = min(len(out) - 1, min_idx + beams_to_mask)
            out[start_idx:end_idx + 1] = 0.0

        return out

    def find_best_gap_heading(self, ranges, angle_min, angle_inc, threshold=1.4):
        """Finds the deepest contiguous drivable corridor and returns its target angle in radians."""
        total_beams = len(ranges)
        fov_rad = math.radians(self.forward_fov_deg / 2.0)
        idx_right = max(0, self.get_index_for_angle(-fov_rad, angle_min, angle_inc))
        idx_left = min(total_beams - 1, self.get_index_for_angle(fov_rad, angle_min, angle_inc))
        
        idx_start = min(idx_right, idx_left)
        idx_end = max(idx_right, idx_left)
        sector = np.copy(ranges[idx_start:idx_end + 1])

        mask = sector > threshold
        if not np.any(mask):
            deepest_sector_idx = int(np.argmax(sector))
            return angle_min + (idx_start + deepest_sector_idx) * angle_inc

        padded = np.concatenate(([0], mask.astype(int), [0]))
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        best_score = -1.0
        best_target_idx = (starts[0] + ends[0] - 1) // 2

        for s, e in zip(starts, ends):
            gap_slice = sector[s:e]
            width = e - s
            mean_depth = float(np.mean(gap_slice))
            max_depth = float(np.max(gap_slice))

            # Reward long, clear corridors
            score = width * (mean_depth ** 1.5) * max_depth
            if score > best_score:
                best_score = score
                # Steer towards center of the gap to stay clear of both walls
                best_target_idx = s + int(width / 2.0)

        target_global_idx = idx_start + best_target_idx
        return angle_min + target_global_idx * angle_inc

    def lidar_callback(self, scan_msg: LaserScan):
        angle_min = scan_msg.angle_min
        angle_inc = scan_msg.angle_increment
        total_beams = len(scan_msg.ranges)

        if total_beams == 0:
            return

        # 1. Clean raw LiDAR
        clean_ranges = self.preprocess_lidar(scan_msg.ranges)

        # 2. Key directional sectors:
        # Front cone: -18 deg to +18 deg
        idx_f_left = self.get_index_for_angle(math.radians(18.0), angle_min, angle_inc)
        idx_f_right = self.get_index_for_angle(math.radians(-18.0), angle_min, angle_inc)
        front_dist = float(np.min(clean_ranges[min(idx_f_right, idx_f_left):max(idx_f_right, idx_f_left) + 1]))

        # Left opening sector: +25 deg to +95 deg
        idx_l_start = self.get_index_for_angle(math.radians(25.0), angle_min, angle_inc)
        idx_l_end = self.get_index_for_angle(math.radians(95.0), angle_min, angle_inc)
        left_ranges = clean_ranges[min(idx_l_start, idx_l_end):max(idx_l_start, idx_l_end) + 1]
        left_openness = float(np.mean(left_ranges))

        # Right opening sector: -95 deg to -25 deg
        idx_r_start = self.get_index_for_angle(math.radians(-95.0), angle_min, angle_inc)
        idx_r_end = self.get_index_for_angle(math.radians(-25.0), angle_min, angle_inc)
        right_ranges = clean_ranges[min(idx_r_start, idx_r_end):max(idx_r_start, idx_r_end) + 1]
        right_openness = float(np.mean(right_ranges))

        # Side wall distances at 90 deg for straight line centering
        idx_90_l = min(total_beams - 1, self.get_index_for_angle(math.radians(90.0), angle_min, angle_inc))
        idx_90_r = max(0, self.get_index_for_angle(math.radians(-90.0), angle_min, angle_inc))
        side_left = float(np.mean(clean_ranges[max(0, idx_90_l - 12):min(total_beams, idx_90_l + 13)]))
        side_right = float(np.mean(clean_ranges[max(0, idx_90_r - 12):min(total_beams, idx_90_r + 13)]))

        # 3. Corner Detection & Early Turn Decision
        # If the front wall is within 4.0m, a corner is imminent
        corner_approaching = (front_dist < self.corner_turn_dist)

        if corner_approaching or self.in_corner:
            # We are preparing for, entering, or actively inside a corner
            if not self.in_corner:
                # Latched turn direction at entry
                self.in_corner = True
                self.latched_turn_dir = 1.0 if (left_openness > right_openness) else -1.0

            # Turn intensity scales decisively from 0.60 at 4.0m to 1.0 at 2.2m
            turn_progress = float(np.clip((self.corner_turn_dist - front_dist) / (self.corner_turn_dist - 2.2), 0.0, 1.0))
            corner_steering = self.latched_turn_dir * (0.60 + 0.40 * turn_progress)

            target_steering = float(np.clip(corner_steering, -1.0, 1.0))
            throttle_cmd = self.corner_speed

            # Release corner latch only when car is cleanly aligned in new corridor:
            # - Front distance must be clear (> 4.2m)
            # - Both side walls must be visible (< 2.2m), indicating the car has re-entered a corridor
            if front_dist > 4.2 and side_left < 2.2 and side_right < 2.2:
                self.in_corner = False

        else:
            # On a clear corridor: Follow the safest gap and maintain centerline balance
            bubbled_ranges = self.apply_safety_bubble(clean_ranges, angle_min, angle_inc)
            gap_heading_rad = self.find_best_gap_heading(bubbled_ranges, angle_min, angle_inc)

            # Lateral centering trim
            wall_error = side_left - side_right
            centering_trim = float(np.clip(wall_error * 0.15, -0.25, 0.25))

            # Combine gap heading and lateral trim
            gap_steering = float(np.clip(gap_heading_rad / self.max_steer_rad, -0.80, 0.80))
            target_steering = float(np.clip(gap_steering + centering_trim, -1.0, 1.0))

            # Speed Profiling:
            if front_dist > self.early_lookahead_dist:
                throttle_cmd = self.straight_speed
            else:
                # Approaching corner (between 5.2m and 4.0m): smoothly decelerate to corner speed
                alpha = (front_dist - self.corner_turn_dist) / (self.early_lookahead_dist - self.corner_turn_dist)
                throttle_cmd = self.corner_speed + alpha * (self.straight_speed - self.corner_speed)

        # 4. Smooth steering to eliminate oscillation while keeping rapid turn-in
        steering_cmd = self.steering_smooth_alpha * target_steering + (1.0 - self.steering_smooth_alpha) * self.prev_steering
        self.prev_steering = steering_cmd

        # 5. Strict forward throttle clamp (NEVER REVERSE, NEVER STALL)
        throttle_cmd = float(np.clip(throttle_cmd, self.min_drive_speed, self.straight_speed))

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
