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

        # Tuned Speeds: Prevents over-speeding down the straightaway into Turn 1
        self.straight_speed = 0.29     # Controlled, smooth straightaway speed (prevents massive momentum)
        self.corner_speed = 0.17       # Slow, planted cornering speed (matches respawn speed)
        self.min_drive_speed = 0.15    # Minimum forward throttle
        self.active_brake = -0.15      # Brief active braking pulse when approaching a wall at speed

        self.max_lidar_range = 8.0     # Cap distant beams
        self.forward_fov_deg = 140.0   # Look ahead arc: +-70 degrees
        self.max_steer_rad = 0.52      # Max steering angle (~30 degrees)

        # Steering smoothing filter
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

        self.get_logger().info('RoboRacer Wall Follower active: Early-braking Turn 1 fix active.')

    def get_index_for_angle(self, angle_rad, angle_min, angle_inc):
        return int(round((angle_rad - angle_min) / angle_inc))

    def preprocess_lidar(self, raw_ranges):
        """Clean sensor data: eliminate self-chassis hits (<0.20m), NaNs, and infs."""
        proc = np.array(raw_ranges, dtype=np.float32)
        proc[np.isnan(proc)] = self.max_lidar_range
        proc[np.isinf(proc)] = self.max_lidar_range
        proc[proc < 0.20] = self.max_lidar_range  # Ignore self chassis reflections
        proc = np.clip(proc, 0.0, self.max_lidar_range)

        # 5-beam moving average filter to remove speckle noise
        kernel = np.ones(5) / 5.0
        proc = np.convolve(proc, kernel, mode='same')
        return proc

    def find_deepest_navigable_gap(self, sector_ranges, threshold=1.4):
        """
        Finds contiguous drivable corridors and scores them by DEPTH * WIDTH.
        This guarantees the car chooses the real track opening rather than a shallow wall pocket.
        """
        mask = sector_ranges > threshold
        if not np.any(mask):
            furthest_idx = int(np.argmax(sector_ranges))
            return furthest_idx, furthest_idx

        padded = np.concatenate(([0], mask.astype(int), [0]))
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        best_score = -1.0
        best_gap = (starts[0], ends[0] - 1)

        for s, e in zip(starts, ends):
            gap_slice = sector_ranges[s:e]
            width = e - s
            mean_depth = float(np.mean(gap_slice))
            max_depth = float(np.max(gap_slice))

            # Score heavily favors corridors with deep line of sight
            score = width * (mean_depth ** 1.5) * max_depth
            if score > best_score:
                best_score = score
                best_gap = (s, e - 1)

        return best_gap

    def lidar_callback(self, scan_msg: LaserScan):
        angle_min = scan_msg.angle_min
        angle_inc = scan_msg.angle_increment
        total_beams = len(scan_msg.ranges)

        if total_beams == 0:
            return

        # 1. Clean raw LiDAR
        clean_ranges = self.preprocess_lidar(scan_msg.ranges)

        # 2. Extract forward planning sector (+- 70 degrees)
        fov_rad = math.radians(self.forward_fov_deg / 2.0)
        idx_right = max(0, self.get_index_for_angle(-fov_rad, angle_min, angle_inc))
        idx_left = min(total_beams - 1, self.get_index_for_angle(fov_rad, angle_min, angle_inc))

        idx_start = min(idx_right, idx_left)
        idx_end = max(idx_right, idx_left)
        sector = np.copy(clean_ranges[idx_start:idx_end + 1])

        # 3. Find the best drivable corridor scored by depth and width
        gap_start, gap_end = self.find_deepest_navigable_gap(sector, threshold=1.4)

        # 4. Target the SAFE CENTER of the corridor
        goal_sector_idx = int((gap_start + gap_end) / 2.0)
        target_global_idx = idx_start + goal_sector_idx
        target_angle = angle_min + target_global_idx * angle_inc

        # 5. Lateral Centering Trim (balance left and right side distances)
        idx_90_left = min(total_beams - 1, self.get_index_for_angle(math.radians(90.0), angle_min, angle_inc))
        idx_90_right = max(0, self.get_index_for_angle(math.radians(-90.0), angle_min, angle_inc))
        left_dist = float(np.mean(clean_ranges[max(0, idx_90_left - 15):min(total_beams, idx_90_left + 15)]))
        right_dist = float(np.mean(clean_ranges[max(0, idx_90_right - 15):min(total_beams, idx_90_right + 15)]))

        wall_diff = left_dist - right_dist
        lateral_trim = float(np.clip(wall_diff * 0.12, -0.20, 0.20))

        # Combined steering goal
        steering_rad = target_angle + lateral_trim

        # Normalize to [-1.0, 1.0] (+1.0 = left, -1.0 = right)
        raw_steering = float(np.clip(steering_rad / self.max_steer_rad, -1.0, 1.0))

        # Smooth steering to eliminate oscillation
        steering_cmd = self.steering_smooth_alpha * raw_steering + (1.0 - self.steering_smooth_alpha) * self.prev_steering
        self.prev_steering = steering_cmd

        # 6. Early Lookahead Braking:
        # Check forward distance directly ahead in front of the car (+- 15 degrees)
        center_idx = self.get_index_for_angle(0.0, angle_min, angle_inc)
        forward_clearance = float(np.min(clean_ranges[max(0, center_idx - 25):min(total_beams, center_idx + 26)]))
        turn_severity = abs(steering_cmd)

        # Proactive multi-stage braking to prevent carrying straightaway momentum into Turn 1:
        if forward_clearance < 2.0 and turn_severity < 0.35:
            # Approaching the end-of-straight wall at speed: apply active braking pulse!
            throttle_cmd = self.active_brake
        elif forward_clearance < 3.2 or turn_severity > 0.18:
            # In or entering the turn: drop straight to slow cornering speed
            throttle_cmd = self.corner_speed
        elif forward_clearance < 5.5:
            # Early deceleration zone (5.5m to 3.2m): smooth ramp down from straight speed
            factor = (forward_clearance - 3.2) / (5.5 - 3.2)
            throttle_cmd = self.corner_speed + factor * (self.straight_speed - self.corner_speed)
        else:
            # Wide open straightaway
            throttle_cmd = self.straight_speed

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
