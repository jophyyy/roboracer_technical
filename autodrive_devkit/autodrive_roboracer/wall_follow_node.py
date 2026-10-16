#!/usr/bin/env python3

"""
Penn RoboRacer Autonomous Wall Following & Follow-the-Gap Controller

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

        # Tunable Controller Parameters
        self.max_speed = 0.85          # Top speed on straights (0.0 to 1.0)
        self.min_speed = 0.40          # Cornering speed (0.0 to 1.0)
        self.bubble_radius = 0.38      # Safety bubble around close obstacles in meters
        self.max_lidar_range = 8.0     # Cap very far readings to avoid false gaps
        self.forward_fov_deg = 150.0   # Field of view considered for path planning (+- 75 deg)
        self.lookahead_dist = 2.5      # Distance threshold to consider obstacle close

        # Previous steering for derivative smoothing
        self.prev_steering = 0.0
        self.steering_smooth_alpha = 0.85

        # Best QoS for real-time sensor streams
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
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

        self.get_logger().info('WallFollowNode initialized. Ready for Autonomous Racing!')

    def get_index_for_angle(self, angle_rad, angle_min, angle_inc):
        """Converts an angle in radians to the closest LiDAR scan index."""
        idx = int(round((angle_rad - angle_min) / angle_inc))
        return idx

    def preprocess_lidar(self, ranges, angle_min, angle_inc):
        """
        Filters and cleans raw scan data.
        Replaces infs, NaNs, and caps ranges to max_lidar_range.
        """
        proc_ranges = np.array(ranges, dtype=np.float32)

        # Replace invalid readings
        proc_ranges[np.isnan(proc_ranges)] = self.max_lidar_range
        proc_ranges[np.isinf(proc_ranges)] = self.max_lidar_range
        proc_ranges[proc_ranges < 0.05] = self.max_lidar_range
        proc_ranges = np.clip(proc_ranges, 0.0, self.max_lidar_range)

        # Smooth using a moving average window of 5 beams to eliminate speckle noise
        kernel = np.ones(5) / 5.0
        proc_ranges = np.convolve(proc_ranges, kernel, mode='same')

        return proc_ranges

    def apply_safety_bubble(self, ranges, angle_min, angle_inc):
        """
        Finds the closest point within the forward sector and zeroes out
        a bubble of radius `bubble_radius` around it.
        """
        fov_rad = math.radians(self.forward_fov_deg / 2.0)
        idx_min = max(0, self.get_index_for_angle(-fov_rad, angle_min, angle_inc))
        idx_max = min(len(ranges) - 1, self.get_index_for_angle(fov_rad, angle_min, angle_inc))

        forward_slice = ranges[idx_min:idx_max + 1]
        if len(forward_slice) == 0:
            return ranges

        # Find closest point
        min_idx_relative = np.argmin(forward_slice)
        min_dist = forward_slice[min_idx_relative]
        closest_idx = idx_min + min_idx_relative

        # If closest obstacle is within threshold, bubble it out
        if min_dist < self.lookahead_dist and min_dist > 0.05:
            # Angle subtended by safety bubble
            half_bubble_angle = math.atan2(self.bubble_radius, min_dist)
            bubble_beams = int(math.ceil(half_bubble_angle / angle_inc))

            start_b = max(0, closest_idx - bubble_beams)
            end_b = min(len(ranges), closest_idx + bubble_beams + 1)
            ranges[start_b:end_b] = 0.0

        return ranges

    def find_max_gap(self, ranges, threshold=1.2):
        """
        Finds the largest contiguous sequence of beams that exceed the threshold.
        """
        above_thresh = ranges > threshold
        if not np.any(above_thresh):
            # Fallback if all beams are close: pick the furthest single beam
            furthest_idx = np.argmax(ranges)
            return furthest_idx, furthest_idx

        # Find continuous slices
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

        # 1. Clean and filter LiDAR data
        ranges = self.preprocess_lidar(scan_msg.ranges, angle_min, angle_inc)

        # 2. Extract forward planning sector (-75 deg to +75 deg)
        fov_rad = math.radians(self.forward_fov_deg / 2.0)
        idx_left = min(total_beams - 1, self.get_index_for_angle(fov_rad, angle_min, angle_inc))
        idx_right = max(0, self.get_index_for_angle(-fov_rad, angle_min, angle_inc))

        # Ensure indices ordered from right to left
        idx_start = min(idx_right, idx_left)
        idx_end = max(idx_right, idx_left)
        sector_ranges = np.copy(ranges[idx_start:idx_end + 1])

        # 3. Apply safety bubble to avoid clipping corner vertices
        sector_ranges = self.apply_safety_bubble(sector_ranges, -fov_rad, angle_inc)

        # 4. Find the largest gap of free drivable space
        gap_start, gap_end = self.find_max_gap(sector_ranges, threshold=1.2)

        # 5. Goal heading: target the deepest opening inside the largest gap
        gap_slice = sector_ranges[gap_start:gap_end + 1]
        best_in_gap = np.argmax(gap_slice)
        goal_sector_idx = gap_start + best_in_gap
        target_global_idx = idx_start + goal_sector_idx

        target_angle = angle_min + target_global_idx * angle_inc

        # 6. Center Bias / Wall Balancing:
        # Check 90 degrees left and 90 degrees right to balance track center
        idx_90_left = min(total_beams - 1, self.get_index_for_angle(math.radians(90.0), angle_min, angle_inc))
        idx_90_right = max(0, self.get_index_for_angle(math.radians(-90.0), angle_min, angle_inc))
        left_dist = np.mean(ranges[idx_90_left - 15:idx_90_left + 15])
        right_dist = np.mean(ranges[idx_90_right - 15:idx_90_right + 15])

        # Lateral error: positive if closer to right wall (steer left), negative if closer to left
        lateral_error = (left_dist - right_dist) * 0.12
        lateral_error = np.clip(lateral_error, -0.25, 0.25)

        # Combined steering goal (in radians)
        steering_rad = target_angle + lateral_error

        # 7. Map to [-1, 1] steering command (1.0 = ~0.52 rad left, -1.0 = ~0.52 rad right)
        max_steer_rad = 0.52
        steering_cmd = steering_rad / max_steer_rad
        steering_cmd = float(np.clip(steering_cmd, -1.0, 1.0))

        # Low-pass filter steering to avoid jitter
        steering_cmd = self.steering_smooth_alpha * steering_cmd + (1.0 - self.steering_smooth_alpha) * self.prev_steering
        self.prev_steering = steering_cmd

        # 8. Adaptive Throttle Control:
        # Direct line-of-sight straight ahead
        center_idx = self.get_index_for_angle(0.0, angle_min, angle_inc)
        forward_clearance = np.min(ranges[center_idx - 20:center_idx + 21])

        # Scale speed based on steering angle and forward clearance
        turn_severity = abs(steering_cmd)
        if forward_clearance > 3.0 and turn_severity < 0.20:
            throttle_cmd = self.max_speed
        elif forward_clearance > 1.8 and turn_severity < 0.45:
            throttle_cmd = (self.max_speed + self.min_speed) * 0.5
        else:
            throttle_cmd = self.min_speed

        throttle_cmd = float(np.clip(throttle_cmd, 0.0, 1.0))

        # Publish commands
        str_msg = Float32()
        str_msg.data = steering_cmd
        self.steering_pub.publish(str_msg)

        thr_msg = Float32()
        thr_msg.data = throttle_cmd
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
