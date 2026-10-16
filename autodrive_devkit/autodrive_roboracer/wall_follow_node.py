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
        self.max_speed = 0.48          # Controlled top speed on straightaways (0.0 to 1.0)
        self.corner_speed = 0.18       # Slow, stable cornering speed to prevent skidding out
        self.min_crawl_speed = 0.10    # Minimum speed when approaching tight hairpin walls
        self.bubble_radius = 0.45      # Safety bubble around close obstacles in meters
        self.max_lidar_range = 8.0     # Cap very far readings to avoid false gaps
        self.forward_fov_deg = 150.0   # Field of view considered for path planning (+- 75 deg)
        self.lookahead_dist = 3.2      # Distance threshold to consider obstacle close

        # Steering smoothing parameters
        self.prev_steering = 0.0
        self.steering_smooth_alpha = 0.75  # Responsive steering without oscillation

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

        self.get_logger().info('WallFollowNode initialized with smooth cornering & speed profiling.')

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

        # 6. Lateral Track Balancing (Centering):
        # Sample side beams at 90 deg left and 90 deg right to keep car centered in corridor
        idx_90_left = min(total_beams - 1, self.get_index_for_angle(math.radians(90.0), angle_min, angle_inc))
        idx_90_right = max(0, self.get_index_for_angle(math.radians(-90.0), angle_min, angle_inc))
        left_dist = np.mean(ranges[max(0, idx_90_left - 20):min(total_beams, idx_90_left + 20)])
        right_dist = np.mean(ranges[max(0, idx_90_right - 20):min(total_beams, idx_90_right + 20)])

        # Lateral correction: steer away from whichever wall is closer
        lateral_error = (left_dist - right_dist) * 0.18
        lateral_error = np.clip(lateral_error, -0.30, 0.30)

        # Combined steering goal (in radians)
        steering_rad = target_angle + lateral_error

        # 7. Map to [-1, 1] steering command (1.0 = full left ~0.52 rad, -1.0 = full right)
        max_steer_rad = 0.52
        raw_steering = steering_rad / max_steer_rad
        raw_steering = float(np.clip(raw_steering, -1.0, 1.0))

        # Smooth steering to prevent jerky zig-zag
        steering_cmd = self.steering_smooth_alpha * raw_steering + (1.0 - self.steering_smooth_alpha) * self.prev_steering
        self.prev_steering = steering_cmd

        # 8. Adaptive Smooth Speed Profiling:
        # Check forward line-of-sight clearance
        center_idx = self.get_index_for_angle(0.0, angle_min, angle_inc)
        forward_clearance = np.min(ranges[max(0, center_idx - 25):min(total_beams, center_idx + 26)])

        turn_severity = abs(steering_cmd)  # 0.0 = straight, 1.0 = full turn lock

        # A) Speed based on steering angle:
        # The harder the turn, the slower the speed (drops down to corner_speed)
        speed_from_steering = self.max_speed - (self.max_speed - self.corner_speed) * (turn_severity ** 1.3)

        # B) Lookahead proactive braking:
        # If an approaching corner or wall is within 4.5 meters, decelerate proactively
        if forward_clearance < 1.4:
            speed_from_clearance = self.min_crawl_speed
        elif forward_clearance < 4.2:
            # Smooth interpolation from max_speed down to corner_speed
            alpha = (forward_clearance - 1.4) / (4.2 - 1.4)
            speed_from_clearance = self.corner_speed + alpha * (self.max_speed - self.corner_speed)
        else:
            speed_from_clearance = self.max_speed

        # Final throttle is the safest of steering limit and sightline limit
        throttle_cmd = min(speed_from_steering, speed_from_clearance)
        throttle_cmd = float(np.clip(throttle_cmd, self.min_crawl_speed, self.max_speed))

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
