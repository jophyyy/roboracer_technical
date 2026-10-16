#!/usr/bin/env python3

"""
Penn RoboRacer Autonomous Competition Controller v2.1
Proven Gap Follower with Proactive Corner Braking and Real-Time Telemetry.

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

        # Calibrated Speeds
        self.straight_speed = 0.32      # Controlled straightaway speed
        self.corner_speed = 0.18        # Safe, high-traction cornering speed (full tire grip)
        self.min_drive_speed = 0.16     # Strictly positive forward throttle (no reverse!)

        # Geometry & Safety
        self.max_lidar_range = 8.0      # Cap distant beams
        self.forward_fov_deg = 150.0    # Look ahead +-75 degrees
        self.max_steer_rad = 0.52       # Max steering limit (~30 deg)
        self.safety_bubble_radius = 0.35 # Radius to bubble closest obstacle point (meters)

        # Steering smoothing
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

        self.get_logger().info('RoboRacer Competition Controller v2.1 active.')

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

    def apply_safety_bubble(self, sector_ranges, angle_inc):
        """Inflates ONLY the single closest forward obstacle to prevent clipping corner vertices."""
        out = np.copy(sector_ranges)
        min_idx = int(np.argmin(out))
        min_dist = float(out[min_idx])

        # If an obstacle is close ahead, mask it with a bubble
        if min_dist < 2.0 and min_dist > 0.15:
            half_bubble_rad = math.atan2(self.safety_bubble_radius, min_dist)
            beams_to_mask = int(math.ceil(half_bubble_rad / angle_inc))
            start_idx = max(0, min_idx - beams_to_mask)
            end_idx = min(len(out) - 1, min_idx + beams_to_mask)
            out[start_idx:end_idx + 1] = 0.0

        return out

    def find_best_gap(self, sector_ranges, threshold=1.2):
        """Finds the largest, deepest contiguous free opening and returns the target index."""
        mask = sector_ranges > threshold
        if not np.any(mask):
            return int(np.argmax(sector_ranges))

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

            # Score favors wide, deep drivable openings
            score = width * (mean_d ** 1.5) * max_d
            if score > best_score:
                best_score = score
                # Steer towards center of the open gap to stay clear of both walls
                best_target_idx = s + int(width / 2.0)

        return best_target_idx

    def lidar_callback(self, scan_msg: LaserScan):
        self.call_count += 1
        angle_min = scan_msg.angle_min
        angle_inc = scan_msg.angle_increment
        total_beams = len(scan_msg.ranges)

        if total_beams == 0:
            return

        # 1. Clean raw LiDAR
        clean_ranges = self.preprocess_lidar(scan_msg.ranges)

        # 2. Extract forward planning sector (+-75 degrees)
        fov_rad = math.radians(self.forward_fov_deg / 2.0)
        idx_right = max(0, self.get_index_for_angle(-fov_rad, angle_min, angle_inc))
        idx_left = min(total_beams - 1, self.get_index_for_angle(fov_rad, angle_min, angle_inc))

        idx_start = min(idx_right, idx_left)
        idx_end = max(idx_right, idx_left)
        sector_ranges = np.copy(clean_ranges[idx_start:idx_end + 1])

        # 3. Apply safety bubble around closest obstacle
        bubbled_sector = self.apply_safety_bubble(sector_ranges, angle_inc)

        # 4. Find the best drivable gap
        target_sector_idx = self.find_best_gap(bubbled_sector, threshold=1.2)
        target_global_idx = idx_start + target_sector_idx
        target_angle = angle_min + target_global_idx * angle_inc

        # 5. Base steering command from gap heading
        raw_steering = float(np.clip(target_angle / self.max_steer_rad, -1.0, 1.0))

        # 6. Lateral Centering Trim (balance side wall distances on straights)
        idx_90_l = min(total_beams - 1, self.get_index_for_angle(math.radians(90.0), angle_min, angle_inc))
        idx_90_r = max(0, self.get_index_for_angle(math.radians(-90.0), angle_min, angle_inc))
        side_l = float(np.mean(clean_ranges[max(0, idx_90_l - 10):min(total_beams, idx_90_l + 11)]))
        side_r = float(np.mean(clean_ranges[max(0, idx_90_r - 10):min(total_beams, idx_90_r + 11)]))

        if abs(raw_steering) < 0.20:
            wall_diff = side_l - side_r
            centering_trim = float(np.clip(wall_diff * 0.10, -0.15, 0.15))
            raw_steering = float(np.clip(raw_steering + centering_trim, -1.0, 1.0))

        # 7. Smooth steering
        steering_cmd = self.steering_smooth_alpha * raw_steering + (1.0 - self.steering_smooth_alpha) * self.prev_steering
        self.prev_steering = steering_cmd

        # 8. Adaptive Speed Profiling (Slow In, Fast Out):
        # Measure forward clearance directly ahead (+-15 degrees)
        idx_f_l = self.get_index_for_angle(math.radians(15.0), angle_min, angle_inc)
        idx_f_r = self.get_index_for_angle(math.radians(-15.0), angle_min, angle_inc)
        forward_dist = float(np.min(clean_ranges[min(idx_f_r, idx_f_l):max(idx_f_r, idx_f_l) + 1]))

        turn_severity = abs(steering_cmd)

        if forward_dist < 4.0 or turn_severity > 0.22:
            # Entering or in a turn: slow down to stable cornering speed
            throttle_cmd = self.corner_speed
        elif forward_dist > 5.2 and turn_severity < 0.12:
            # Clear straight: swift speed
            throttle_cmd = self.straight_speed
        else:
            # Smooth deceleration approaching corner
            factor = (forward_dist - 4.0) / (5.2 - 4.0)
            throttle_cmd = self.corner_speed + factor * (self.straight_speed - self.corner_speed)

        # Strictly clamp throttle: never stall, never reverse!
        throttle_cmd = float(np.clip(throttle_cmd, self.min_drive_speed, self.straight_speed))

        # Log heartbeat every 40 frames (~1 sec)
        if self.call_count % 40 == 0:
            self.get_logger().info(
                f'[LIVE v2.1] Steer: {steering_cmd:+.2f} | Throttle: {throttle_cmd:.2f} | Front: {forward_dist:.2f}m'
            )

        # 9. Publish control commands
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
