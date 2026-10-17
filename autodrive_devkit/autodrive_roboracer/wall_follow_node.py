#!/usr/bin/env python3

"""
Penn RoboRacer Autonomous Wall Following & Intersection Navigation Controller

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

        # Tuned Speed & Turning Dynamics
        # ALWAYS positive throttle - never reverses or rocks back and forth!
        self.straight_speed = 0.28     # Controlled straightaway speed (prevents overspeeding Turn 1)
        self.corner_speed = 0.20       # Steady forward cornering speed (maximum front tire grip)
        self.min_drive_speed = 0.18    # Minimum forward throttle (guaranteed roll)

        self.max_lidar_range = 8.0     # Cap distant beams
        # Wide 200-degree FOV (+-100 deg) so 90-degree corners and intersections are completely visible!
        self.forward_fov_deg = 200.0   
        self.max_steer_rad = 0.44      # Dynamic steering authority (~25 degrees gives full lock)

        # Responsive steering smoothing
        self.prev_steering = 0.0
        self.steering_smooth_alpha = 0.85  # Snappy, decisive turn-in at corners

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

        self.get_logger().info('RoboRacer Refined Corridor Controller active.')

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

    def find_best_corridor(self, sector_ranges, threshold=2.8):
        """
        Finds the true track corridor by scoring openings using width * depth.
        Threshold = 2.8m ensures the corner corridor is isolated early (3m before the wall),
        eliminating the delayed steering trap without affecting straightaway driving.
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

            # Depth-weighted score: heavily rewards the long open corridor
            score = width * (mean_depth ** 1.8) * max_depth
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

        # 2. Extract wide planning sector (+- 100 degrees to see 90-deg turns)
        fov_rad = math.radians(self.forward_fov_deg / 2.0)
        idx_right = max(0, self.get_index_for_angle(-fov_rad, angle_min, angle_inc))
        idx_left = min(total_beams - 1, self.get_index_for_angle(fov_rad, angle_min, angle_inc))

        idx_start = min(idx_right, idx_left)
        idx_end = max(idx_right, idx_left)
        sector = np.copy(clean_ranges[idx_start:idx_end + 1])

        # 3. Find the best corridor leading into the turn/intersection
        # Threshold = 2.8m isolates the open turn corridor well before the car reaches the wall
        gap_start, gap_end = self.find_best_corridor(sector, threshold=2.8)

        # 4. Target the SAFE CENTER of the corridor
        goal_sector_idx = int((gap_start + gap_end) / 2.0)
        target_global_idx = idx_start + goal_sector_idx
        target_angle = angle_min + target_global_idx * angle_inc

        # 5. Direct Decisive Steering:
        # Normalize target angle to steering range [-1.0, 1.0]
        raw_steering = float(np.clip(target_angle / self.max_steer_rad, -1.0, 1.0))

        # Smooth steering to prevent twitching while maintaining rapid turn response
        steering_cmd = self.steering_smooth_alpha * raw_steering + (1.0 - self.steering_smooth_alpha) * self.prev_steering
        self.prev_steering = steering_cmd

        # 6. Smooth Forward Throttle Control (ALWAYS POSITIVE - NO REVERSING!):
        # Forward clearance directly ahead (+- 15 degrees)
        center_idx = self.get_index_for_angle(0.0, angle_min, angle_inc)
        forward_clearance = float(np.min(clean_ranges[max(0, center_idx - 25):min(total_beams, center_idx + 26)]))
        turn_severity = abs(steering_cmd)

        # If a turn or wall is approaching, smoothly settle into cornering speed
        if forward_clearance < 3.8 or turn_severity > 0.20:
            # Entering or in the turn: steady, planted corner speed
            throttle_cmd = self.corner_speed
        elif forward_clearance < 5.5:
            # Smoothly transition from straight speed down to corner speed
            alpha = (forward_clearance - 3.8) / (5.5 - 3.8)
            throttle_cmd = self.corner_speed + alpha * (self.straight_speed - self.corner_speed)
        else:
            # Wide open straight
            throttle_cmd = self.straight_speed

        # Guarantee forward drive: clamp between min_drive_speed and straight_speed
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
