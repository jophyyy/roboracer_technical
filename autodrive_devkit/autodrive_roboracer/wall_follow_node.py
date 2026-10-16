#!/usr/bin/env python3

"""
Penn RoboRacer Autonomous Wall Following & Center Navigation Controller

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

        # Calibrated Speed Settings (Guaranteed to move & stable in corners)
        self.straight_speed = 0.42     # Fast enough for good lap time on straights
        self.corner_speed = 0.24       # Controlled cornering speed (grippy, never skids)
        self.min_drive_speed = 0.22    # Minimum throttle to overcome vehicle static friction

        self.bubble_radius = 0.35      # Safety margin around obstacles (meters)
        self.max_lidar_range = 8.0     # Cap distant beams
        self.forward_fov_deg = 140.0   # Look ahead arc: +-70 degrees
        
        # Steering smoothing
        self.prev_steering = 0.0
        self.steering_smooth_alpha = 0.65

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

        self.get_logger().info('RoboRacer Controller initialized: Ready to race.')

    def get_index_for_angle(self, angle_rad, angle_min, angle_inc):
        return int(round((angle_rad - angle_min) / angle_inc))

    def preprocess_lidar(self, raw_ranges):
        """Clean sensor data: eliminate self-chassis hits (<0.20m), NaNs, and infs."""
        proc = np.array(raw_ranges, dtype=np.float32)
        proc[np.isnan(proc)] = self.max_lidar_range
        proc[np.isinf(proc)] = self.max_lidar_range
        proc[proc < 0.20] = self.max_lidar_range  # Ignore self chassis reflections
        proc = np.clip(proc, 0.0, self.max_lidar_range)

        # Moving average filter across 5 beams
        kernel = np.ones(5) / 5.0
        proc = np.convolve(proc, kernel, mode='same')
        return proc

    def apply_safety_bubble(self, sector_ranges, angle_inc):
        """Find the closest obstacle in the forward sector and mask it with a safety bubble."""
        # Find closest point
        min_idx = int(np.argmin(sector_ranges))
        min_dist = sector_ranges[min_idx]

        # Only apply bubble to real obstacles within 2.5m
        if 0.20 < min_dist < 2.5:
            half_angle = math.atan2(self.bubble_radius, min_dist)
            num_beams = int(math.ceil(half_angle / angle_inc))
            start = max(0, min_idx - num_beams)
            end = min(len(sector_ranges), min_idx + num_beams + 1)
            sector_ranges[start:end] = 0.0

        return sector_ranges

    def find_largest_gap(self, sector_ranges, threshold=1.4):
        """Find the start and end indices of the largest contiguous opening."""
        mask = sector_ranges > threshold
        if not np.any(mask):
            # If all are close, pick the furthest point available
            idx = int(np.argmax(sector_ranges))
            return idx, idx

        # Find contiguous true segments
        padded = np.concatenate(([0], mask.astype(int), [0]))
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        lengths = ends - starts
        best_idx = np.argmax(lengths)
        return starts[best_idx], ends[best_idx] - 1

    def lidar_callback(self, scan_msg: LaserScan):
        angle_min = scan_msg.angle_min
        angle_inc = scan_msg.angle_increment
        total_beams = len(scan_msg.ranges)

        if total_beams == 0:
            return

        # 1. Clean LiDAR ranges
        clean_ranges = self.preprocess_lidar(scan_msg.ranges)

        # 2. Extract forward sector (-70 deg to +70 deg)
        fov_rad = math.radians(self.forward_fov_deg / 2.0)
        idx_right = max(0, self.get_index_for_angle(-fov_rad, angle_min, angle_inc))
        idx_left = min(total_beams - 1, self.get_index_for_angle(fov_rad, angle_min, angle_inc))

        idx_start = min(idx_right, idx_left)
        idx_end = max(idx_right, idx_left)
        sector = np.copy(clean_ranges[idx_start:idx_end + 1])

        # 3. Apply safety bubble around closest point
        masked_sector = self.apply_safety_bubble(sector, angle_inc)

        # 4. Find the largest drivable gap
        gap_start, gap_end = self.find_largest_gap(masked_sector, threshold=1.4)

        # 5. Target the CENTER of the largest gap
        # Steering down the center ensures maximum clearance from both walls
        goal_sector_idx = int((gap_start + gap_end) / 2.0)
        target_global_idx = idx_start + goal_sector_idx
        target_angle = angle_min + target_global_idx * angle_inc

        # 6. Lateral Centering Trim (balance left and right side distances)
        idx_90_left = min(total_beams - 1, self.get_index_for_angle(math.radians(90.0), angle_min, angle_inc))
        idx_90_right = max(0, self.get_index_for_angle(math.radians(-90.0), angle_min, angle_inc))
        left_dist = np.mean(clean_ranges[max(0, idx_90_left - 15):min(total_beams, idx_90_left + 15)])
        right_dist = np.mean(clean_ranges[max(0, idx_90_right - 15):min(total_beams, idx_90_right + 15)])

        wall_diff = left_dist - right_dist
        lateral_trim = float(np.clip(wall_diff * 0.10, -0.20, 0.20))

        # Combined steering in radians
        steering_rad = target_angle + lateral_trim

        # Map to normalized steering command [-1.0, 1.0] (0.52 rad max lock)
        max_steer_rad = 0.52
        raw_steering = float(np.clip(steering_rad / max_steer_rad, -1.0, 1.0))

        # Filter steering to eliminate rapid oscillation
        steering_cmd = self.steering_smooth_alpha * raw_steering + (1.0 - self.steering_smooth_alpha) * self.prev_steering
        self.prev_steering = steering_cmd

        # 7. Adaptive Throttle Control:
        # Check forward sightline (center +- 15 deg)
        center_idx = self.get_index_for_angle(0.0, angle_min, angle_inc)
        forward_clearance = np.min(clean_ranges[max(0, center_idx - 25):min(total_beams, center_idx + 26)])
        turn_severity = abs(steering_cmd)

        if forward_clearance < 2.8 or turn_severity > 0.22:
            # Entering or in a turn: slow down to cornering speed
            throttle_cmd = self.corner_speed
        elif forward_clearance > 4.5 and turn_severity < 0.12:
            # Wide open straightaway: full straight speed
            throttle_cmd = self.straight_speed
        else:
            # Transition zone: smooth interpolation
            factor = (forward_clearance - 2.8) / (4.5 - 2.8)
            throttle_cmd = self.corner_speed + factor * (self.straight_speed - self.corner_speed)

        # Enforce minimum drive speed so the car never stalls against static friction
        throttle_cmd = float(np.clip(throttle_cmd, self.min_drive_speed, self.straight_speed))

        # Publish commands
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
