# Custom DSLPIDControl (RL-Tuned Version)

This repository contains a **modified and extended version** of the original `DSLPIDControl` controller from  
 **https://github.com/utiasDSL/gym-pybullet-drones**

Our version introduces **Reinforcement-Learning–tuned gains**, improved structure, preserved author notes, and clearly documented control constraints.

---

##  Overview

`DSLPIDControl` implements a **cascaded PD controller** for Crazyflie 2.x drones:

### **Outer Loop — Position PD**
- Computes position error  
- Computes velocity error  
- Produces *desired acceleration*  
- Produces *desired thrust vector*  
- Generates *desired orientation* (roll, pitch, yaw)

### **Inner Loop — Attitude PD on SO(3)**
- Computes attitude error from rotation matrices  
- Computes body-rate error  
- Computes torques using PD  
- Uses Crazyflie mixer matrix to produce motor PWM  
- Converts PWM → RPM  

---

##  Modifications in This Fork

### **1️⃣ Reinforcement Learning–Tuned Gains**

A custom RL loop was used to optimize `(Kp, Kd)` values for position and attitude loops.

The RL agent was rewarded for:
- Fast convergence  
- No overshoot  
- No oscillation  
- Zero final error  
- Convergence under 4–5 seconds  

**Observed performance:**

| Axis | Convergence Time | Final Error |
|------|------------------|-------------|
| x    | ~3.0 s           | ≈ 0         |
| y    | ~3.0–4.0 s       | ≈ 0         |
| z    | ~4.5 s           | ≈ 0         |

These gains are now the **default** in this fork.

---

### **2️⃣ Cleaner and More Transparent Control Code**

We reorganized and clarified:

- Gravity-compensated thrust vector computation  
- Desired attitude generation via geometric control  
- SO(3)-based attitude error  
- Torque clamping  
- Mixer architecture  

This makes the control **more stable**, **more readable**, and **more predictable**.

---

### **3️⃣ Preserved Original Author Notes**

All original DSL comments and contributor credits remain intact for reference.

---

## 📎 Constraints and Assumptions

### ✔ Supported Drone Models
- `DroneModel.CF2X`  
- `DroneModel.CF2P`

### ✔ Actuator Limits
- `MIN_PWM = 20000`  
- `MAX_PWM = 65535`  
- Torque saturation at ±3200  

### ✔ No Integral Terms
This controller remains **strictly PD** (no integral), matching both course requirements and Crazyflie stability guidelines.

### ✔ Attitude Error on SO(3)
Uses:

e_R = 0.5 * vee(R_d^T * R - R^T * R_d)
e_omega = ω − R^T * R_d * ω_d

shell
Copy code

### ✔ Desired Rotation Derived from Thrust Vector

T_d = m * (g * e3 + desired_acc)
z_B_d = T_d / ||T_d||

yaml
Copy code

Special handling avoids alignment singularities.

---

## RL-Tuned Gains (Default)

| Loop | Gain Type | Values |
|------|-----------|--------|
| Position | `Kp_pos` | `[2, 2, 2]` |
| Position | `Kd_pos` | `[2.35, 2.35, 2.35]` |
| Attitude | `Kp_att` | `[4800, 4800, 4800]` |
| Attitude | `Kd_att` | `[600, 600, 600]` |

Earlier manual gains + RL search logs remain commented in the control file.

---

##  Original Repository (Fork Source)

This implementation is an extension of:

 **https://github.com/utiasDSL/gym-pybullet-drones**

### Please cite if used academically:

```bibtex
@INPROCEEDINGS{panerati2021learning,
      title={Learning to Fly---a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control}, 
      author={Jacopo Panerati and Hehui Zheng and SiQi Zhou and James Xu and Amanda Prorok and Angela P. Schoellig},
      booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
      year={2021},
      pages={7512-7519},
      doi={10.1109/IROS51168.2021.9635857}
}
Example Usage
python
Copy code
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel
import numpy as np

ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)

rpm, pos_error, att_error = ctrl.computeControl(
    control_timestep=1/240,
    cur_pos=current_position,
    cur_quat=current_quaternion,
    cur_vel=current_velocity,
    cur_ang_vel=current_bodyrates,
    target_pos=np.array([1.0, 0.0, 1.0])
)

print("Motor RPM:", rpm)
print("Position Error:", pos_error)
print("Attitude Error:", att_error)
 File Summary
DSLPIDControl.py

Contains default RL-tuned gains

Contains original manually tuned gains (commented)

Implements cascaded PD control

Includes gravity compensation, mixer, torque limits, etc.

Preserves all original author notes

Optional Future Additions
Add integral corrections for trajectory tracking

Add trajectory-smoothing spline input

Add wind/drag disturbance models

Add domain-randomized RL retuning
