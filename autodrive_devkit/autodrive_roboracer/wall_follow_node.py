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

        # Tunable Speed & Safety Parameters
        # Slowed down for safe, stable cornering without understeer or skidding
        self.max_speed = 0.32          # Straightaway speed (calm and controlled)
        self.corner_speed = 0.13       # Very slow cornering speed so turns are taken gently
        self.crawl_speed = 0.08        # Emergency crawl when close to any wall/obstacle
        
        self.bubble_radius = 0.50      # Generous safety bubble around corners and obstacles (meters)
        self.max_lidar_range = 8.0     # Cap far readings
        self.forward_fov_deg = 140.0   # Field of view for gap finding (+- 70 deg)
        self.lookahead_dist = 3.0      # Distance to activate safety bubble
        
        # Steering smoothing
        self.prev_steering = 0.0
        self.steering_smooth_alpha = 0.70

        # Reliable QoS matching autodrive_bridge
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

        self.get_logger().info('WallFollowNode initialized: Slow & Smooth Cornering Mode active.')

    def get_index_for_angle(self, angle_rad, angle_min, angle_inc):
        idx = int(round((angle_rad - angle_min) / angle_inc))
        return idx

    def preprocess_lidar(self, ranges, max_range):
        proc_ranges = np.array(ranges, dtype=np.float32)
        proc_ranges[np.isnan(proc_ranges)] = max_range
        proc_ranges[np.isinf(proc_ranges)] = max_range
        proc_ranges[proc_ranges < 0.05] = max_range
        proc_ranges = np.clip(proc_ranges, 0.0, max_range)

        # 5-beam moving average filter to remove sensor noise
        kernel = np.ones(5) / 5.0
        proc_ranges = np.convolve(proc_ranges, kernel, mode='same')
        return proc_ranges

    def apply_safety_bubble(self, ranges, angle_inc):
        """
        Finds the closest point in the planning sector and zeroes out a safety bubble
        around it to prevent clipping corners or obstacles.
        """
        min_idx = np.argmin(ranges)
        min_dist = ranges[min_idx]

        if min_dist < self.lookahead_dist and min_dist > 0.05:
            half_bubble_angle = math.atan2(self.bubble_radius, min_dist)
            bubble_beams = int(math.ceil(half_bubble_angle / angle_inc))

            start_b = max(0, min_idx - bubble_beams)
            end_b = min(len(ranges), min_idx + bubble_beams + 1)
            ranges[start_b:end_b] = 0.0

        return ranges

    def find_best_gap(self, ranges, threshold=1.2):
        """
        Finds the largest contiguous opening of drivable space.
        """
        above_thresh = ranges > threshold
        if not np.any(above_thresh):
            furthest_idx = np.argmax(ranges)
            return furthest_idx, furthest_idx

        padded = np.concatenate(([0], above_thresh.astype(int), [0]))
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        lengths = ends - starts
        max_idx = np.argmax(lengths)
        return starts[max_idx], ends[max_idx] - 1

    def lidar_callback(self, scan_msg: LaserScan):
        angle_min = scan_msg.angle_min
        angle_inc = scan_msg.angle_increment
        total_beams = len(scan_msg.ranges)

        if total_beams == 0:
            return

        # 1. Preprocess LiDAR
        ranges = self.preprocess_lidar(scan_msg.ranges, self.max_lidar_range)

        # 2. Extract forward planning sector (+- 70 degrees)
        fov_rad = math.radians(self.forward_fov_deg / 2.0)
        idx_right = max(0, self.get_index_for_angle(-fov_rad, angle_min, angle_inc))
        idx_left = min(total_beams - 1, self.get_index_for_angle(fov_rad, angle_min, angle_inc))

        idx_start = min(idx_right, idx_left)
        idx_end = max(idx_right, idx_left)
        sector_ranges = np.copy(ranges[idx_start:idx_end + 1])

        # 3. Apply safety bubble around closest obstacle
        sector_ranges = self.apply_safety_bubble(sector_ranges, angle_inc)

        # 4. Find the largest drivable gap
        gap_start, gap_end = self.find_best_gap(sector_ranges, threshold=1.2)

        # 5. Steer toward the CENTER of the largest gap
        # (Targeting the center guarantees the car stays away from both left and right walls)
        goal_sector_idx = int((gap_start + gap_end) / 2.0)
        target_global_idx = idx_start + goal_sector_idx
        target_angle = angle_min + target_global_idx * angle_inc

        # 6. Lateral Centering Trim (balance distances to left and right walls)
        idx_90_left = min(total_beams - 1, self.get_index_for_angle(math.radians(90.0), angle_min, angle_inc))
        idx_90_right = max(0, self.get_index_for_angle(math.radians(-90.0), angle_min, angle_inc))
        left_dist = np.mean(ranges[max(0, idx_90_left - 15):min(total_beams, idx_90_left + 15)])
        right_dist = np.mean(ranges[max(0, idx_90_right - 15):min(total_beams, idx_90_right + 15)])

        # Steer toward center of the hallway
        wall_diff = left_dist - right_dist
        lateral_trim = float(np.clip(wall_diff * 0.12, -0.25, 0.25))

        # Combined steering angle in radians
        steering_rad = target_angle + lateral_trim

        # Map to [-1, 1] (0.52 rad max steering limit)
        max_steer_rad = 0.52
        raw_steering = float(np.clip(steering_rad / max_steer_rad, -1.0, 1.0))

        # Smooth steering to avoid oscillation
        steering_cmd = self.steering_smooth_alpha * raw_steering + (1.0 - self.steering_smooth_alpha) * self.prev_steering
        self.prev_steering = steering_cmd

        # 7. Adaptive Slow Speed Profiling:
        # Check forward distance right in front of the car (+- 15 degrees)
        center_idx = self.get_index_for_angle(0.0, angle_min, angle_inc)
        forward_clearance = np.min(ranges[max(0, center_idx - 30):min(total_beams, center_idx + 31)])

        turn_severity = abs(steering_cmd)  # 0.0 is straight, 1.0 is full turn

        # Continuous throttle calculation:
        # Straightaway speed: drops immediately when steering or approaching turns
        if forward_clearance < 1.6:
            # Wall directly ahead: slow down to crawl so front tires turn
            throttle_cmd = self.crawl_speed
        elif turn_severity > 0.25:
            # Taking a turn: slow down to corner_speed
            throttle_cmd = self.corner_speed
        elif forward_clearance < 4.0:
            # Approaching a turn: decelerate smoothly
            factor = (forward_clearance - 1.6) / (4.0 - 1.6)
            throttle_cmd = self.corner_speed + factor * (self.max_speed - self.corner_speed)
        else:
            # Clear straightaway
            throttle_cmd = self.max_speed

        throttle_cmd = float(np.clip(throttle_cmd, self.crawl_speed, self.max_speed))

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
