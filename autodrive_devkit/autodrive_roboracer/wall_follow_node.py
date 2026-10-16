#!/usr/bin/env python3

"""
Penn RoboRacer Autonomous Competition Controller v2.2
Deepest Corridor Cluster Targeting, High Steering Authority, and Lookahead Braking.

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
        self.corner_speed = 0.17        # Slow, high-traction cornering speed (full tire grip)
        self.min_drive_speed = 0.15     # Strictly positive forward throttle (no reverse!)

        # Geometry & Perception
        self.max_lidar_range = 8.0      # Cap distant beams
        self.forward_fov_deg = 170.0    # Look ahead +-85 degrees
        self.steering_gain_rad = 0.35   # 0.35 rad (~20 deg) target commands full 1.0 steering lock
        self.safety_bubble_radius = 0.35 # Radius to bubble closest obstacle point (meters)

        # Steering smoothing (fast response to turn early and decisively)
        self.prev_steering = 0.0
        self.steering_smooth_alpha = 0.90

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

        self.get_logger().info('RoboRacer High-Authority Cornering Controller v2.2 active.')

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
        """Inflates only dangerously close obstacles in front to prevent clipping inner apexes."""
        out = np.copy(sector_ranges)
        min_idx = int(np.argmin(out))
        min_dist = float(out[min_idx])

        # Only bubble if an obstacle is dangerously close in front (<1.0m)
        if min_dist < 1.0 and min_dist > 0.15:
            half_bubble_rad = math.atan2(self.safety_bubble_radius, min_dist)
            beams_to_mask = int(math.ceil(half_bubble_rad / angle_inc))
            start_idx = max(0, min_idx - beams_to_mask)
            end_idx = min(len(out) - 1, min_idx + beams_to_mask)
            out[start_idx:end_idx + 1] = 0.0

        return out

    def find_deepest_corridor_heading(self, sector_ranges, sector_angles):
        """
        Finds the deepest cluster of beams (the open corridor).
        Prioritizes the true open corridor over the dead-end front wall,
        initiating turning early and with full steering authority.
        """
        max_d = float(np.max(sector_ranges))
        # Deep threshold isolates the long open corridor from the approaching wall ahead
        threshold = max(2.2, 0.70 * max_d)
        deep_mask = sector_ranges >= threshold

        if not np.any(deep_mask):
            best_idx = int(np.argmax(sector_ranges))
            return sector_angles[best_idx]

        padded = np.concatenate(([0], deep_mask.astype(int), [0]))
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        best_len = -1
        best_target_idx = len(sector_ranges) // 2

        for s, e in zip(starts, ends):
            length = e - s
            if length > best_len:
                best_len = length
                # Steer towards center of the deep opening
                best_target_idx = s + length // 2

        return sector_angles[best_target_idx]

    def lidar_callback(self, scan_msg: LaserScan):
        self.call_count += 1
        angle_min = scan_msg.angle_min
        angle_inc = scan_msg.angle_increment
        total_beams = len(scan_msg.ranges)

        if total_beams == 0:
            return

        # 1. Clean raw LiDAR
        clean_ranges = self.preprocess_lidar(scan_msg.ranges)

        # 2. Extract forward planning sector (+-85 degrees)
        fov_rad = math.radians(self.forward_fov_deg / 2.0)
        idx_right = max(0, self.get_index_for_angle(-fov_rad, angle_min, angle_inc))
        idx_left = min(total_beams - 1, self.get_index_for_angle(fov_rad, angle_min, angle_inc))

        idx_start = min(idx_right, idx_left)
        idx_end = max(idx_right, idx_left)
        sector_ranges = np.copy(clean_ranges[idx_start:idx_end + 1])
        angles = np.array([angle_min + i * angle_inc for i in range(total_beams)])
        sector_angles = angles[idx_start:idx_end + 1]

        # 3. Apply safety bubble around closest obstacle
        bubbled_sector = self.apply_safety_bubble(sector_ranges, angle_inc)

        # 4. Find optimal heading into the deepest corridor
        target_angle = self.find_deepest_corridor_heading(bubbled_sector, sector_angles)

        # 5. High-Authority Steering Scaling:
        # A ~20 deg target angle commands 100% full steering lock (1.0 or -1.0)
        raw_steering = float(np.clip(target_angle / self.steering_gain_rad, -1.0, 1.0))

        # 6. Lateral Centering Trim on straights
        idx_90_l = min(total_beams - 1, self.get_index_for_angle(math.radians(90.0), angle_min, angle_inc))
        idx_90_r = max(0, self.get_index_for_angle(math.radians(-90.0), angle_min, angle_inc))
        side_l = float(np.mean(clean_ranges[max(0, idx_90_l - 10):min(total_beams, idx_90_l + 11)]))
        side_r = float(np.mean(clean_ranges[max(0, idx_90_r - 10):min(total_beams, idx_90_r + 11)]))

        if abs(raw_steering) < 0.20:
            wall_diff = side_l - side_r
            centering_trim = float(np.clip(wall_diff * 0.10, -0.15, 0.15))
            raw_steering = float(np.clip(raw_steering + centering_trim, -1.0, 1.0))

        # 7. Smooth steering (fast alpha = 0.90 for instant corner entry)
        steering_cmd = self.steering_smooth_alpha * raw_steering + (1.0 - self.steering_smooth_alpha) * self.prev_steering
        self.prev_steering = steering_cmd

        # 8. Adaptive Speed Profiling (Slow In, Fast Out):
        # Measure forward clearance directly ahead (+-15 degrees)
        idx_f_l = self.get_index_for_angle(math.radians(15.0), angle_min, angle_inc)
        idx_f_r = self.get_index_for_angle(math.radians(-15.0), angle_min, angle_inc)
        forward_dist = float(np.min(clean_ranges[min(idx_f_r, idx_f_l):max(idx_f_r, idx_f_l) + 1]))

        turn_severity = abs(steering_cmd)

        if forward_dist < 4.2 or turn_severity > 0.22:
            # Approaching corner or turning: slow down to high-traction corner speed
            throttle_cmd = self.corner_speed
        elif forward_dist > 5.5 and turn_severity < 0.12:
            # Clear straightaway: swift speed
            throttle_cmd = self.straight_speed
        else:
            # Smooth proactive braking
            factor = (forward_dist - 4.2) / (5.5 - 4.2)
            throttle_cmd = self.corner_speed + factor * (self.straight_speed - self.corner_speed)

        # Strictly clamp throttle: never stall, never reverse!
        throttle_cmd = float(np.clip(throttle_cmd, self.min_drive_speed, self.straight_speed))

        # Log heartbeat every 40 frames (~1 sec)
        if self.call_count % 40 == 0:
            self.get_logger().info(
                f'[LIVE v2.2] Steer: {steering_cmd:+.2f} | Throttle: {throttle_cmd:.2f} | Front: {forward_dist:.2f}m'
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
