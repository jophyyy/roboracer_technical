#!/usr/bin/env python3

"""
Penn RoboRacer Autonomous Competition Controller
Adaptive Safety-Bubble Gap Following with Wide Field of View & Dynamic Corner Braking.

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
        self.straight_speed = 0.35      # Swift straightaway speed
        self.corner_speed = 0.20        # Safe, high-traction cornering speed
        self.min_drive_speed = 0.18     # Minimum throttle to guarantee no stalling

        # Perception Geometry
        self.max_lidar_range = 8.0      # Cap distant beams
        self.forward_fov_deg = 190.0    # +-95 degrees FOV to see perpendicular turns
        self.max_steer_rad = 0.52       # Max steering limit (~30 deg)
        self.safety_bubble_radius = 0.45 # Safety clearance around inner apexes & walls (meters)

        # Steering smoothing
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

        self.get_logger().info('RoboRacer Adaptive Gap Follower active.')

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

    def apply_safety_bubble(self, ranges, angles, fov_mask, angle_inc):
        """Inflates any nearby obstacle (<0.85m) by safety_bubble_radius so the car never clips corners."""
        bubbled = np.copy(ranges)
        close_indices = np.where((bubbled < 0.85) & (bubbled > 0.15) & fov_mask)[0]
        
        for idx in close_indices:
            dist = float(bubbled[idx])
            half_ang = math.atan2(self.safety_bubble_radius, dist)
            beams = int(math.ceil(half_ang / angle_inc))
            start_idx = max(0, idx - beams)
            end_idx = min(len(bubbled) - 1, idx + beams)
            bubbled[start_idx:end_idx + 1] = 0.0

        return bubbled

    def find_best_gap_angle(self, sector_ranges, sector_angles, threshold=1.8):
        """Finds the largest, deepest contiguous free opening and returns its target angle."""
        mask = sector_ranges > threshold
        if not np.any(mask):
            # Fallback if in tight space: steer toward furthest beam
            best_idx = int(np.argmax(sector_ranges))
            return sector_angles[best_idx]

        padded = np.concatenate(([0], mask.astype(int), [0]))
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        best_score = -1.0
        best_target_idx = (starts[0] + ends[0] - 1) // 2

        for s, e in zip(starts, ends):
            gap_slice = sector_ranges[s:e]
            width = e - s
            mean_d = float(np.mean(gap_slice))
            max_d = float(np.max(gap_slice))

            # Score heavily favors deep open track corridors over shallow dead-ends
            score = width * (mean_d ** 1.8) * max_d
            if score > best_score:
                best_score = score
                # Steer towards center of the open gap to stay clear of both walls
                best_target_idx = s + int(width / 2.0)

        return sector_angles[best_target_idx]

    def lidar_callback(self, scan_msg: LaserScan):
        angle_min = scan_msg.angle_min
        angle_inc = scan_msg.angle_increment
        total_beams = len(scan_msg.ranges)

        if total_beams == 0:
            return

        # 1. Clean raw LiDAR
        clean_ranges = self.preprocess_lidar(scan_msg.ranges)

        # 2. Forward arc (+-95 degrees)
        angles = np.array([angle_min + i * angle_inc for i in range(total_beams)])
        fov_rad = math.radians(self.forward_fov_deg / 2.0)
        fov_mask = np.abs(angles) <= fov_rad

        # 3. Apply safety bubble around close walls / apexes
        bubbled_ranges = self.apply_safety_bubble(clean_ranges, angles, fov_mask, angle_inc)

        # 4. Extract sector within FOV
        sector_indices = np.where(fov_mask)[0]
        sector_ranges = bubbled_ranges[sector_indices]
        sector_angles = angles[sector_indices]

        # 5. Find optimal heading into the deepest gap
        target_angle = self.find_best_gap_angle(sector_ranges, sector_angles, threshold=1.8)

        # 6. Normalize steering command [-1.0, 1.0] (+1.0 = Left, -1.0 = Right)
        raw_steering = float(np.clip(target_angle / self.max_steer_rad, -1.0, 1.0))

        # 7. Lateral Centering Trim on straights
        idx_90_l = min(total_beams - 1, self.get_index_for_angle(math.radians(90.0), angle_min, angle_inc))
        idx_90_r = max(0, self.get_index_for_angle(math.radians(-90.0), angle_min, angle_inc))
        side_l = float(np.mean(clean_ranges[max(0, idx_90_l - 12):min(total_beams, idx_90_l + 13)]))
        side_r = float(np.mean(clean_ranges[max(0, idx_90_r - 12):min(total_beams, idx_90_r + 13)]))

        if abs(raw_steering) < 0.20:
            wall_diff = side_l - side_r
            trim = float(np.clip(wall_diff * 0.10, -0.15, 0.15))
            raw_steering = float(np.clip(raw_steering + trim, -1.0, 1.0))

        # 8. Smooth steering
        steering_cmd = self.steering_smooth_alpha * raw_steering + (1.0 - self.steering_smooth_alpha) * self.prev_steering
        self.prev_steering = steering_cmd

        # 9. Adaptive Speed Profiling (Slow in, fast out):
        # Measure forward clearance ahead (+-15 degrees)
        idx_f_l = self.get_index_for_angle(math.radians(15.0), angle_min, angle_inc)
        idx_f_r = self.get_index_for_angle(math.radians(-15.0), angle_min, angle_inc)
        forward_dist = float(np.min(clean_ranges[min(idx_f_r, idx_f_l):max(idx_f_r, idx_f_l) + 1]))

        turn_severity = abs(steering_cmd)

        if forward_dist < 4.0 or turn_severity > 0.25:
            # Entering or in a turn: slow, stable cornering speed (full tire grip)
            throttle_cmd = self.corner_speed
        elif forward_dist > 5.5 and turn_severity < 0.12:
            # Clear straightaway: swift acceleration
            throttle_cmd = self.straight_speed
        else:
            # Smooth transition between corner and straight speed
            factor = (forward_dist - 4.0) / (5.5 - 4.0)
            throttle_cmd = self.corner_speed + factor * (self.straight_speed - self.corner_speed)

        # Strictly clamp throttle: never stall, never reverse!
        throttle_cmd = float(np.clip(throttle_cmd, self.min_drive_speed, self.straight_speed))

        # 10. Publish control commands
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
