#!/usr/bin/env python3

"""
AutoDRIVE RoboRacer Autonomous Wall-Following & Anticipatory Cornering Controller
Features:
  1. Direction-Latched Cornering: Eliminates mid-corner coordinate flip-flops.
  2. Progressive Arc Steering: Smooth continuous turn-in from 4.2m down to apex.
  3. Continuous Deceleration Glide: Smooth engine-braking weight transfer onto front tires.
  4. Apex Clearance Safety: Protects against clipping inside corner curbs.
  5. Straightaway Deadband & Damping: Zero left-right oscillation or hunting.

Subscribes:
  /autodrive/roboracer_1/lidar (sensor_msgs/LaserScan)
  /autodrive/reset_command (std_msgs/Bool)

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
from std_msgs.msg import Float32, Bool


class WallFollowNode(Node):
    def __init__(self):
        super().__init__('wall_follow_node')

        # -------------------------------------------------------------
        # ROS 2 Tunable Parameters (reconfigurable via --ros-args -p ...)
        # -------------------------------------------------------------
        self.declare_parameter('follow_side', 'left')          # 'left' (turns right) or 'right' (turns left)
        self.declare_parameter('desired_distance', 0.65)       # Target wall distance on straights (meters)
        self.declare_parameter('kp', 1.00)                     # Proportional gain for straightaway wall following
        self.declare_parameter('kd', 0.40)                     # Derivative damping gain for straightaways
        self.declare_parameter('lookahead', 1.00)              # Lookahead projection distance L (meters)
        self.declare_parameter('max_throttle', 0.28)           # Straightaway cruising speed
        self.declare_parameter('min_throttle', 0.16)           # Calibrated corner speed floor (steady grip)
        self.declare_parameter('corner_entry_dist', 4.20)      # Distance ahead to begin smooth deceleration & turn-in
        self.declare_parameter('corner_apex_dist', 1.30)       # Distance at which full corner steering authority is reached

        # Internal controller state
        self.prev_error = None
        self.prev_time = None
        self.filtered_d_error = 0.0
        self.prev_steering = 0.0
        self.prev_throttle = 0.22
        self.launch_grace_until = 0.0                          # Straight rollout at spawn/reset
        self.last_log_time = 0.0

        # Corner state latching (prevents mid-corner coordinate flip-flop)
        self.in_corner = False
        self.latched_turn_dir = 0.0                            # -1.0 = Right, +1.0 = Left

        # Reliable QoS matching AutoDRIVE Bridge
        qos = QoSProfile(
            durability=QoSDurabilityPolicy.VOLATILE,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers
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

        # Subscribers
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/autodrive/roboracer_1/lidar',
            self.lidar_callback,
            qos
        )
        self.reset_sub = self.create_subscription(
            Bool,
            '/autodrive/reset_command',
            self.reset_callback,
            1
        )

        self.get_logger().info('RoboRacer Direction-Latched Smooth-Arc Controller active.')

    def reset_callback(self, msg: Bool):
        if msg.data:
            now_time = self.get_clock().now().nanoseconds * 1e-9
            self.launch_grace_until = now_time + 2.0
            self.prev_steering = 0.0
            self.prev_throttle = 0.20
            self.in_corner = False
            self.latched_turn_dir = 0.0
            self.get_logger().info('Simulator reset: launch grace active.')

    def get_index_for_angle(self, angle_rad, angle_min, angle_inc, total_beams):
        """Compute beam index dynamically from LaserScan header."""
        idx = int(round((angle_rad - angle_min) / angle_inc))
        return max(0, min(total_beams - 1, idx))

    def get_range_window(self, clean_ranges, angle_rad, angle_min, angle_inc, window_deg=2.5, mode='mean'):
        """Extract smoothed range within a degree window."""
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
        now_time = self.get_clock().now().nanoseconds * 1e-9
        if self.launch_grace_until == 0.0:
            self.launch_grace_until = now_time + 2.0

        raw_ranges = np.array(scan_msg.ranges, dtype=np.float32)
        total_beams = len(raw_ranges)
        if total_beams == 0:
            return

        angle_min = scan_msg.angle_min
        angle_inc = scan_msg.angle_increment

        # -------------------------------------------------------------
        # 1. LIDAR PREPROCESSING & FILTERING
        # -------------------------------------------------------------
        clean_ranges = np.copy(raw_ranges)
        clean_ranges[np.isnan(clean_ranges)] = 10.0
        clean_ranges[np.isinf(clean_ranges)] = 10.0
        clean_ranges[clean_ranges < 0.20] = 10.0  # Ignore self-chassis reflections

        # 5-beam moving average smoothing kernel
        kernel = np.ones(5, dtype=np.float32) / 5.0
        clean_ranges = np.convolve(clean_ranges, kernel, mode='same')

        # Read tunable parameters
        follow_side = self.get_parameter('follow_side').get_parameter_value().string_value
        desired_dist = self.get_parameter('desired_distance').get_parameter_value().double_value
        kp = self.get_parameter('kp').get_parameter_value().double_value
        kd = self.get_parameter('kd').get_parameter_value().double_value
        lookahead = self.get_parameter('lookahead').get_parameter_value().double_value
        max_throttle = self.get_parameter('max_throttle').get_parameter_value().double_value
        min_throttle = self.get_parameter('min_throttle').get_parameter_value().double_value
        corner_entry_dist = self.get_parameter('corner_entry_dist').get_parameter_value().double_value
        corner_apex_dist = self.get_parameter('corner_apex_dist').get_parameter_value().double_value

        # -------------------------------------------------------------
        # 2. TRACK GEOMETRY EVALUATION
        # -------------------------------------------------------------
        # Direct front clearance (+-10 deg cone ahead)
        front = self.get_range_window(clean_ranges, 0.0, angle_min, angle_inc, window_deg=10.0, mode='min')
        # Broader forward corridor (+-15 deg cone ahead)
        front_corridor = self.get_range_window(clean_ranges, 0.0, angle_min, angle_inc, window_deg=15.0, mode='mean')

        # Forward-diagonal sectors (+20° to +70° left, -70° to -20° right)
        idx_l20 = self.get_index_for_angle(math.radians(20.0), angle_min, angle_inc, total_beams)
        idx_l70 = self.get_index_for_angle(math.radians(70.0), angle_min, angle_inc, total_beams)
        left_sector = clean_ranges[min(idx_l20, idx_l70):max(idx_l20, idx_l70) + 1]
        left_sector_avg = float(np.mean(left_sector)) if len(left_sector) > 0 else 0.0

        idx_r70 = self.get_index_for_angle(math.radians(-70.0), angle_min, angle_inc, total_beams)
        idx_r20 = self.get_index_for_angle(math.radians(-20.0), angle_min, angle_inc, total_beams)
        right_sector = clean_ranges[min(idx_r70, idx_r20):max(idx_r70, idx_r20) + 1]
        right_sector_avg = float(np.mean(right_sector)) if len(right_sector) > 0 else 0.0

        is_launching = (now_time < self.launch_grace_until)

        # -------------------------------------------------------------
        # 3. CORNER DETECTION & DIRECTION LATCHING
        # -------------------------------------------------------------
        if is_launching:
            self.in_corner = False
            self.latched_turn_dir = 0.0
        else:
            if not self.in_corner:
                # Approach trigger: wall ahead is closer than corner_entry_dist (4.2m)
                if front < corner_entry_dist:
                    sector_diff = right_sector_avg - left_sector_avg
                    if sector_diff > 0.25:
                        self.latched_turn_dir = -1.0  # Open corridor on the Right
                    elif sector_diff < -0.25:
                        self.latched_turn_dir = 1.0   # Open corridor on the Left
                    else:
                        # Fallback to circuit loop direction
                        self.latched_turn_dir = -1.0 if follow_side == 'left' else 1.0
                    self.in_corner = True
            else:
                # Corner Exit Condition:
                # The car has rotated through the turn and sees the new straightaway open ahead
                if front > 3.60 and front_corridor > 3.50:
                    self.in_corner = False
                    self.latched_turn_dir = 0.0

        # -------------------------------------------------------------
        # 4. STEERING COMPUTATION
        # -------------------------------------------------------------
        inside_clearance = 10.0
        if is_launching:
            target_steer = 0.0
            alpha_steer = 0.50
            D = 0.0
            D_future = 0.0
        elif self.in_corner:
            # Progressive corner arc: smoothly increases from 0.0 at entry to 1.0 at apex
            progress = float(np.clip((corner_entry_dist - front) / (corner_entry_dist - corner_apex_dist), 0.0, 1.0))

            # Base progressive steering: starts early at 0.35 and smoothly builds to 1.00 full lock
            base_steer_mag = 0.35 + 0.65 * progress

            # Inside Corner Apex Protection:
            # Protect against clipping the inside curb if the car turns in too tightly
            if self.latched_turn_dir < 0:  # Turning RIGHT
                idx_in_start = self.get_index_for_angle(math.radians(-85.0), angle_min, angle_inc, total_beams)
                idx_in_end = self.get_index_for_angle(math.radians(-30.0), angle_min, angle_inc, total_beams)
                inside_sector = clean_ranges[min(idx_in_start, idx_in_end):max(idx_in_start, idx_in_end) + 1]
                inside_clearance = float(np.min(inside_sector)) if len(inside_sector) > 0 else 1.0

                if inside_clearance < 0.35:
                    # Too close to inside apex curb: ease steering slightly to widen exit arc
                    steer_mag = min(base_steer_mag, 0.75)
                else:
                    steer_mag = base_steer_mag

                # STRICT ANTI-INVERSION: Can ONLY steer right (negative)
                target_steer = -steer_mag
            else:  # Turning LEFT
                idx_in_start = self.get_index_for_angle(math.radians(30.0), angle_min, angle_inc, total_beams)
                idx_in_end = self.get_index_for_angle(math.radians(85.0), angle_min, angle_inc, total_beams)
                inside_sector = clean_ranges[min(idx_in_start, idx_in_end):max(idx_in_start, idx_in_end) + 1]
                inside_clearance = float(np.min(inside_sector)) if len(inside_sector) > 0 else 1.0

                if inside_clearance < 0.35:
                    steer_mag = min(base_steer_mag, 0.75)
                else:
                    steer_mag = base_steer_mag

                # STRICT ANTI-INVERSION: Can ONLY steer left (positive)
                target_steer = steer_mag

            alpha_steer = 0.75  # Responsive turn-in following the progressive arc
            D = 0.0
            D_future = 0.0
        else:
            # -------------------------------------------------------------
            # STRAIGHTAWAY TWO-BEAM WALL FOLLOWING (Calm & Stable)
            # -------------------------------------------------------------
            theta = math.radians(40.0)
            if follow_side == 'right':
                angle_b = math.radians(-90.0)
                angle_a = math.radians(-50.0)
            else:
                angle_b = math.radians(90.0)
                angle_a = math.radians(50.0)

            b = self.get_range_window(clean_ranges, angle_b, angle_min, angle_inc, window_deg=2.5, mode='mean')
            a = self.get_range_window(clean_ranges, angle_a, angle_min, angle_inc, window_deg=2.5, mode='mean')

            # Wall angle and projected future distance
            alpha = math.atan2(a * math.cos(theta) - b, a * math.sin(theta))
            D = b * math.cos(alpha)
            D_future = D + lookahead * math.sin(alpha)

            if follow_side == 'right':
                error = desired_dist - D_future
            else:
                error = D_future - desired_dist

            # Filtered derivative
            dt = 0.0 if self.prev_time is None else (now_time - self.prev_time)
            if dt <= 1e-4 or dt > 0.5 or self.prev_error is None:
                raw_d_error = 0.0
            else:
                raw_d_error = (error - self.prev_error) / dt

            self.filtered_d_error = 0.20 * raw_d_error + 0.80 * self.filtered_d_error
            self.prev_error = error
            self.prev_time = now_time

            # 3cm deadband prevents micro-hunting or alternating left-right on straights
            if abs(error) < 0.03:
                steer_pd = 0.0
            else:
                steer_pd = kp * error + kd * self.filtered_d_error

            # Limit straightaway steering authority so the car never veers abruptly
            target_steer = float(np.clip(steer_pd, -0.45, 0.45))
            alpha_steer = 0.35  # Gentle smoothing on straights

        # Apply steering filter
        steering_cmd = alpha_steer * target_steer + (1.0 - alpha_steer) * self.prev_steering
        steering_cmd = float(np.clip(steering_cmd, -1.0, 1.0))
        self.prev_steering = steering_cmd

        # -------------------------------------------------------------
        # 5. CONTINUOUS PROGRESSIVE DECELERATION GLIDE
        # -------------------------------------------------------------
        # Speed smoothly scales with forward clearance from corner_entry_dist down to 1.5m
        t_front = float(np.clip((front - 1.5) / (corner_entry_dist - 1.5), 0.0, 1.0))
        throttle_lerp = min_throttle + t_front * (max_throttle - min_throttle)
        target_throttle = throttle_lerp * (1.0 - 0.25 * abs(steering_cmd))
        target_throttle = float(np.clip(target_throttle, min_throttle, max_throttle))

        # First-order throttle filter: smooth gradual speed transition
        throttle_filter_alpha = 0.15
        throttle_cmd = throttle_filter_alpha * target_throttle + (1.0 - throttle_filter_alpha) * self.prev_throttle
        throttle_cmd = float(np.clip(throttle_cmd, min_throttle, max_throttle))
        self.prev_throttle = throttle_cmd

        # -------------------------------------------------------------
        # PUBLISH COMMANDS
        # -------------------------------------------------------------
        str_msg = Float32()
        str_msg.data = float(steering_cmd)
        self.steering_pub.publish(str_msg)

        thr_msg = Float32()
        thr_msg.data = float(throttle_cmd)
        self.throttle_pub.publish(thr_msg)

        # -------------------------------------------------------------
        # THROTTLED LOGGING (~2 Hz)
        # -------------------------------------------------------------
        if now_time - self.last_log_time >= 0.5:
            self.last_log_time = now_time
            mode_str = f"CORNER({self.latched_turn_dir:+.0f})" if self.in_corner else "STRAIGHT"
            self.get_logger().info(
                f"[{mode_str}] front={front:4.2f}m | in_clr={inside_clearance:4.2f}m | "
                f"steer={str_msg.data:+5.2f} | thr={thr_msg.data:4.2f}"
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
