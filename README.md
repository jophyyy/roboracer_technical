# Penn RoboRacer Technical Challenge — Autonomous Sim-Racing

Deterministic waypoint-following controller for the F1TENTH-style vehicle in the AutoDRIVE Ecosystem.

---

## 🏎️ Performance & Submission Details
- **Fastest Clean Lap Time:** `13.22 seconds`
- **YouTube Demonstration (Driver's Eye View):** [Watch the clean lap](https://www.youtube.com/watch?v=Ku6pj1r-kwg)

---

## 🧠 Controller Architecture (`wall_follow_node.py`)

The controller implements deterministic **Pure Pursuit** over a fixed Porto track path:

1. **Fixed Porto Path:**
   - Uses the 21 ordered Porto checkpoint centers extracted from the simulator scene.
   - Converts Unity X/Z coordinates into the ROS X/Y coordinate frame.
   - Densifies the closed loop into 397 path points for consistent tracking.

2. **Ground-Truth Localization:**
   - Subscribes to `/autodrive/roboracer_1/odom` for position and orientation.
   - Finds the nearest point on the closed path and selects a target ahead of the vehicle.

3. **Pure-Pursuit Steering:**
   - Transforms the target into the vehicle frame and computes curvature from the lateral displacement.
   - Converts curvature into a normalized steering command using the RoboRacer wheelbase and steering limit.

4. **Completion-First Crawl Mode:**
   - Uses a constant `0.11` throttle and a short `0.70 m` steering lookahead.
   - Prioritizes stable path tracking and collision-free lap completion over speed.
   - Does not depend on LiDAR wall or corner detection.

---

## 🚀 How to Run

### 1. Launch Devkit & Headless Bridge
```bash
ros2 launch autodrive_roboracer bringup_headless.launch.py
```

### 2. Connect the Simulator
- Launch AutoDRIVE Simulator (Practice Track).
- Set IP to `127.0.0.1`, Port `4567`, and click **Connect**.
- Switch Vehicle Mode to **Autonomous**.

### 3. Launch the Wall Follower
```bash
ros2 run autodrive_roboracer wall_follow_node
```
