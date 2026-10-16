# Penn RoboRacer Technical Challenge — Autonomous Sim-Racing

Autonomous wall-following and obstacle avoidance controller for the F1TENTH-style vehicle in the AutoDRIVE Ecosystem.

---

## 🏎️ Performance & Submission Details
- **Fastest Clean Lap Time:** `[TBD — e.g. 18.42s]` *(Update with your best time from the HUD)*
- **YouTube Demonstration (Driver's Eye View):** `[Insert Unlisted YouTube Link Here]`

---

## 🧠 Controller Architecture (`wall_follow_node.py`)

The controller implements an enhanced **Follow-the-Gap Method (FGM)** combined with **Lateral Wall Balancing** and **Adaptive Speed Profiling**:

1. **LiDAR Preprocessing & Filtering:**
   - Filters 1,080 scan beams across the 270° FOV, discarding invalid measurements (NaNs/infs) and clamping readings to a maximum effective range (8.0m).
   - Applies a 5-beam moving-average smoothing kernel to filter high-frequency speckle noise.

2. **Obstacle Detection & Dynamic Safety Bubble:**
   - Detects the nearest obstacle point within the forward heading arc ($\pm 75^\circ$).
   - Projects a safety bubble (radius 0.38m) around the closest point by zeroing out beams within the subtended angle, ensuring the vehicle does not clip corner vertices or apexes.

3. **Maximum Gap Extraction & Deepest Point Targeting:**
   - Identifies the widest contiguous corridor of navigable free space exceeding the clearance threshold.
   - Computes target heading toward the deepest open beam within the largest gap.

4. **Lateral Track Balancing:**
   - Evaluates perpendicular LiDAR beams at $\pm 90^\circ$ to calculate distance to the left and right walls.
   - Introduces a lateral error correction term to keep the car centered along track straights.

5. **Adaptive Throttle Profiling:**
   - Dynamically modulates throttle based on direct line-of-sight forward clearance and steering severity:
     - **Full Throttle (0.85):** When forward clearance $> 3.0\text{m}$ and steering angle $< 0.20$.
     - **Intermediate Throttle (0.62):** Moderate turns with clearance $> 1.8\text{m}$.
     - **Cornering Throttle (0.40):** Sharp turns or tight obstacles to maximize grip.

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
