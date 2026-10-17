#!/usr/bin/env python3

"""
AutoDRIVE RoboRacer Autonomous Wall-Following & Anticipatory Cornering Controller

Subscribes:
  /autodrive/roboracer_1/lidar (sensor_msgs/LaserScan)

Publishes:
  /autodrive/roboracer_1/throttle_command (std_msgs/Float32) [-1.0, 1.0]
  /autodrive/roboracer_1/steering_command (std_msgs/Float32) [-1.0, 1.0]
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32


class WallFollowNode(Node):
    def __init__(self):
        super().__init__('wall_follow_node')

        # -------------------------------------------------------------
        # ROS 2 Tunable Parameters (reconfigurable via --ros-args -p ...)
        # -------------------------------------------------------------
        self.declare_parameter('follow_side', 'left')          # 'left' or 'right'
        self.declare_parameter('desired_distance', 0.85)       # Target distance to followed wall (meters)
        self.declare_parameter('kp', 1.20)                     # Proportional gain (error in m -> steer [-1, 1])
        self.declare_parameter('kd', 0.35)                     # Derivative gain (d_error/dt -> steer [-1, 1])
        self.declare_parameter('lookahead', 0.85)              # Lookahead projection distance L (meters)
        self.declare_parameter('max_throttle', 0.28)           # Top straightaway speed (conservative for clean laps)
        self.declare_parameter('min_throttle', 0.16)           # Base speed floor in tight corners
        self.declare_parameter('front_threshold_base', 2.20)   # Front clearance to trigger corner logic
        self.declare_parameter('front_threshold_gain', 2.00)   # Speed multiplier for front clearance threshold

        # Internal controller state
        self.prev_error = None
        self.prev_time = None
        self.filtered_d_error = 0.0
        self.prev_steering = 0.0
        self.prev_throttle = 0.20
        self.corner_latch = 0.0        # Locks turn direction during cornering to prevent oscillation
        self.last_log_time = 0.0

        # Steering dynamics: responsive smoothing filter (alpha >= 0.60)
        self.steering_smooth_alpha = 0.65

        # Reliable QoS matching AutoDRIVE Bridge
        qos = QoSProfile(
            durability=QoSDurabilityPolicy.VOLATILE,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers matching teleop_keyboard.py
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

        # LiDAR Subscriber
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/autodrive/roboracer_1/lidar',
            self.lidar_callback,
            qos
        )

        self.get_logger().info('RoboRacer Stabilized Wall Follow Node active.')

    def get_index_for_angle(self, angle_rad, angle_min, angle_inc, total_beams):
        """Compute beam index dynamically from message fields."""
        idx = int(round((angle_rad - angle_min) / angle_inc))
        return max(0, min(total_beams - 1, idx))

    def get_range_window(self, clean_ranges, angle_rad, angle_min, angle_inc, window_deg=2.5, mode='mean'):
        """
        Extract ranges in a small window (+- window_deg) around the requested angle.
        Returns the mean or min to eliminate single-beam noise and dropouts.
        """
        total_beams = len(clean_ranges)
        center_idx = self.get_index_for_angle(angle_rad, angle_min, angle_inc, total_beams)
        win_beams = max(1, int(round(math.radians(window_deg) / angle_inc)))
        start_idx = max(0, center_idx - win_beams)
        end_idx = min(total_beams - 1, center_idx + win_beams)
        window = clean_ranges[start_idx:end_idx + 1]
        if len(window) == 0:
            return 10.0
        if mode == 'min':
            return float(np.min(window))
        return float(np.mean(window))

    def lidar_callback(self, scan_msg: LaserScan):
        total_beams = len(scan_msg.ranges)
        if total_beams == 0:
            return

        angle_min = scan_msg.angle_min
        angle_inc = scan_msg.angle_increment
        range_min = scan_msg.range_min if scan_msg.range_min > 0.05 else 0.06
        range_max = scan_msg.range_max if scan_msg.range_max > 0.0 else 10.0

        # Preprocess LiDAR: replace nan/inf with range_max, filter chassis returns (<0.20m), clip
        clean_ranges = np.array(scan_msg.ranges, dtype=np.float32)
        clean_ranges[np.isnan(clean_ranges)] = range_max
        clean_ranges[np.isinf(clean_ranges)] = range_max
        clean_ranges[clean_ranges < 0.20] = range_max
        clean_ranges = np.clip(clean_ranges, range_min, range_max)

        # Dynamic parameter lookup
        follow_side = self.get_parameter('follow_side').get_parameter_value().string_value.lower()
        desired_dist = self.get_parameter('desired_distance').get_parameter_value().double_value
        kp = self.get_parameter('kp').get_parameter_value().double_value
        kd = self.get_parameter('kd').get_parameter_value().double_value
        lookahead = self.get_parameter('lookahead').get_parameter_value().double_value
        max_throttle = self.get_parameter('max_throttle').get_parameter_value().double_value
        min_throttle = self.get_parameter('min_throttle').get_parameter_value().double_value
        front_thresh_base = self.get_parameter('front_threshold_base').get_parameter_value().double_value
        front_thresh_gain = self.get_parameter('front_threshold_gain').get_parameter_value().double_value

        # -------------------------------------------------------------
        # 1. TWO-BEAM LOOKAHEAD WALL-ANGLE METHOD
        # -------------------------------------------------------------
        # Beam b at 90 deg to followed wall, beam a at 50 deg from followed wall (40 deg forward of side beam).
        # theta = 40 deg (in radians)
        theta = math.radians(40.0)

        if follow_side == 'right':
            angle_b = math.radians(-90.0)
            angle_a = math.radians(-50.0)
        else: # default 'left'
            angle_b = math.radians(90.0)
            angle_a = math.radians(50.0)

        b = self.get_range_window(clean_ranges, angle_b, angle_min, angle_inc, window_deg=2.5, mode='mean')
        a = self.get_range_window(clean_ranges, angle_a, angle_min, angle_inc, window_deg=2.5, mode='mean')

        # Wall angle alpha: alpha = atan2(a*cos(theta) - b, a*sin(theta))
        alpha = math.atan2(a * math.cos(theta) - b, a * math.sin(theta))
        # Current perpendicular distance D to wall
        D = b * math.cos(alpha)
        # Projected future distance D_future at lookahead distance L
        D_future = D + lookahead * math.sin(alpha)

        # -------------------------------------------------------------
        # Sign Conventions & Error Calculation:
        # Steering > 0 is LEFT, Steering < 0 is RIGHT.
        #
        # For LEFT wall:
        #   - Too close (D_future < desired): must steer RIGHT (negative).
        #   - Too far   (D_future > desired): must steer LEFT (positive).
        #   => error = D_future - desired_dist
        #
        # For RIGHT wall:
        #   - Too close (D_future < desired): must steer LEFT (positive).
        #   - Too far   (D_future > desired): must steer RIGHT (negative).
        #   => error = desired_dist - D_future
        # -------------------------------------------------------------
        if follow_side == 'right':
            error = desired_dist - D_future
        else:
            error = D_future - desired_dist

        # Compute derivative safely with a low-pass filter to eliminate single-tick sensor jitter
        now_time = self.get_clock().now().nanoseconds * 1e-9
        if self.prev_time is None:
            dt = 0.0
        else:
            dt = now_time - self.prev_time

        if dt <= 1e-4 or dt > 0.5 or self.prev_error is None:
            raw_d_error = 0.0
        else:
            raw_d_error = (error - self.prev_error) / dt

        # First-order filter on derivative (eliminates high-frequency left-right chatter)
        self.filtered_d_error = 0.20 * raw_d_error + 0.80 * self.filtered_d_error

        self.prev_error = error
        self.prev_time = now_time

        # PD control on wall following error
        steer_pd = kp * error + kd * self.filtered_d_error

        # -------------------------------------------------------------
        # 2. FRONT-DISTANCE CORNER LOGIC WITH DEAD-BAND & LATCHING
        # -------------------------------------------------------------
        # Front distance: min range in +-10 deg cone ahead
        front = self.get_range_window(clean_ranges, 0.0, angle_min, angle_inc, window_deg=10.0, mode='min')

        # Sector clearances:
        # Left sector: +60 deg to +100 deg
        idx_l60 = self.get_index_for_angle(math.radians(60.0), angle_min, angle_inc, total_beams)
        idx_l100 = self.get_index_for_angle(math.radians(100.0), angle_min, angle_inc, total_beams)
        left_sector = clean_ranges[min(idx_l60, idx_l100):max(idx_l60, idx_l100) + 1]
        left_sector_avg = float(np.mean(left_sector)) if len(left_sector) > 0 else 0.0

        # Right sector: -100 deg to -60 deg
        idx_r100 = self.get_index_for_angle(math.radians(-100.0), angle_min, angle_inc, total_beams)
        idx_r60 = self.get_index_for_angle(math.radians(-60.0), angle_min, angle_inc, total_beams)
        right_sector = clean_ranges[min(idx_r100, idx_r60):max(idx_r100, idx_r60) + 1]
        right_sector_avg = float(np.mean(right_sector)) if len(right_sector) > 0 else 0.0

        # Speed-dependent front threshold: front < base + gain * speed (~2.6m at 0.25 throttle)
        front_threshold = front_thresh_base + front_thresh_gain * max(0.0, self.prev_throttle)
        sector_diff = left_sector_avg - right_sector_avg

        if front < front_threshold:
            # Approaching a corner:
            # Use deadband (0.35m) so straightaways with equidistant walls never trigger false turns.
            # Latch turn direction during cornering to prevent rapid left-right switching.
            if self.corner_latch != 0.0:
                open_side_sign = self.corner_latch
            elif sector_diff > 0.35:
                open_side_sign = 1.0   # Left is genuinely open
                self.corner_latch = 1.0
            elif sector_diff < -0.35:
                open_side_sign = -1.0  # Right is genuinely open
                self.corner_latch = -1.0
            else:
                # Both walls are roughly equidistant (straight corridor) -> NO corner override!
                open_side_sign = 0.0
        else:
            # Straightaway: clear corner latch
            self.corner_latch = 0.0
            open_side_sign = 0.0

        if open_side_sign != 0.0:
            # Genuine corner: apply anticipatory steering ramping up to full lock (+-1.0)
            urgency = (front_threshold - front) / max(0.01, front_threshold)
            corner_mag = float(np.clip(urgency * 1.8, 0.45, 1.0))
            corner_steer = open_side_sign * corner_mag

            # Override or supplement PD wall term decisively
            if open_side_sign > 0:  # Turning LEFT: ensure steering is at least corner_steer
                target_steer = max(steer_pd, corner_steer)
            else:                   # Turning RIGHT: ensure steering is at least as negative as corner_steer
                target_steer = min(steer_pd, corner_steer)
        else:
            # Straight corridor: pure PD wall following (drives straight forward without weaving!)
            target_steer = steer_pd

        # -------------------------------------------------------------
        # 4. LIGHT STEERING SMOOTHING (alpha >= 0.6 on new value)
        # -------------------------------------------------------------
        steering_cmd = self.steering_smooth_alpha * target_steer + (1.0 - self.steering_smooth_alpha) * self.prev_steering
        steering_cmd = float(np.clip(steering_cmd, -1.0, 1.0))
        self.prev_steering = steering_cmd

        # -------------------------------------------------------------
        # 3. SPEED SCHEDULING
        # -------------------------------------------------------------
        # Throttle depends on front clearance and steering angle:
        # High when front is clear and steering is small, low when front is close or |steering| is large
        t_front = float(np.clip((front - 0.8) / 3.5, 0.0, 1.0))
        throttle_lerp = min_throttle + t_front * (max_throttle - min_throttle)
        throttle_cmd = throttle_lerp * (1.0 - 0.5 * abs(steering_cmd))
        throttle_cmd = float(np.clip(throttle_cmd, min_throttle * 0.75, max_throttle))
        self.prev_throttle = throttle_cmd

        # -------------------------------------------------------------
        # PUBLISH COMMANDS (Clamped strictly to [-1, 1])
        # -------------------------------------------------------------
        str_msg = Float32()
        str_msg.data = float(np.clip(steering_cmd, -1.0, 1.0))
        self.steering_pub.publish(str_msg)

        thr_msg = Float32()
        thr_msg.data = float(np.clip(throttle_cmd, -1.0, 1.0))
        self.throttle_pub.publish(thr_msg)

        # -------------------------------------------------------------
        # THROTTLED LOGGING (~2 Hz)
        # -------------------------------------------------------------
        if now_time - self.last_log_time >= 0.5:
            self.last_log_time = now_time
            self.get_logger().info(
                f"[WALL_FOLLOW] front={front:4.2f}m | D={D:4.2f}m | D_fut={D_future:4.2f}m | "
                f"steer={str_msg.data:+5.2f} | thr={thr_msg.data:4.2f} | latch={self.corner_latch:+3.0f}"
            )


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
