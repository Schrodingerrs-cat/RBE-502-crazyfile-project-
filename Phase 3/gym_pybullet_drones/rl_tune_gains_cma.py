"""
Pure random search tuner for DSLPIDControl gains.
Fixed version: ensures circle.circle() always receives scalar time.
"""

import traceback

import numpy as np

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.examples import trajectory_tracking as traj_mod
from gym_pybullet_drones.trajectory import circle


# ----------------------------------------------------------------------
# Episode runner
# ----------------------------------------------------------------------
def run_episode_and_extract_logs(duration_sec: float = 20.0, gui: bool = False):
    logger = traj_mod.run(
        gui=gui,
        plot=False,
        duration_sec=int(duration_sec),
    )

    states = np.array(logger.states)
    timestamps = np.array(logger.timestamps).astype(float).reshape(-1)
    T = timestamps.size

    # Make sure states is (12, T)
    if states.ndim == 3:
        states = states[0]
    if states.ndim != 2 or states.shape[0] < 12:
        raise RuntimeError(f"Unexpected states shape {states.shape}")

    s = states[:12, :]

    pos = s[0:3, :]
    rpy = s[3:6, :]
    vel = s[6:9, :]
    omega = s[9:12, :]

    # Desired trajectories
    des_pos = np.zeros((3, T))
    des_vel = np.zeros((3, T))
    des_rpy = np.zeros((3, T))

    for k, tk in enumerate(timestamps):
        tk_scalar = float(np.asarray(tk).reshape(()))
        des = circle.circle(tk_scalar)
        des_pos[:, k] = des["pos"]
        des_vel[:, k] = des["vel"]
        # des_rpy stays all zeros (circle trajectory)

    return timestamps, pos, vel, rpy, omega, des_pos, des_vel, des_rpy


# ----------------------------------------------------------------------
# Cost function
# ----------------------------------------------------------------------
def episode_cost(t, pos, vel, rpy, omega, des_pos, des_vel, des_rpy, horizon=10.0):
    t = np.asarray(t, float)
    mask = t <= horizon
    if not np.any(mask):
        mask = slice(None)

    if isinstance(mask, slice):
        idx = slice(None)
        dt = 1.0 / 48.0
    else:
        idx = mask
        dt = float(np.mean(np.diff(t[idx])))

    pos_e = pos[:, idx] - des_pos[:, idx]
    vel_e = vel[:, idx] - des_vel[:, idx]
    rpy_e = rpy[:, idx] - des_rpy[:, idx]
    omega_e = omega[:, idx]

    w_pos, w_vel, w_rpy, w_omega = 5.0, 2.0, 1.0, 0.5

    total = (
        w_pos * np.sum(pos_e**2) * dt
        + w_vel * np.sum(vel_e**2) * dt
        + w_rpy * np.sum(rpy_e**2) * dt
        + w_omega * np.sum(omega_e**2) * dt
    )

    # Terminal penalty
    if isinstance(mask, slice):
        last = -1
    else:
        last = int(np.where(mask)[0][-1])

    final_pos_e = pos[:, last] - des_pos[:, last]
    final_vel_e = vel[:, last] - des_vel[:, last]

    total += 10.0 * (
        np.linalg.norm(final_pos_e) ** 2 + np.linalg.norm(final_vel_e) ** 2
    )

    return float(total)


# ----------------------------------------------------------------------
# Gain sampling
# ----------------------------------------------------------------------
def sample_log10_gains():
    return np.concatenate(
        [
            np.random.uniform(-2, 1, size=4),
            np.random.uniform(0, 4, size=4),
        ]
    )


def log_to_gains(theta_log):
    return 10.0 ** np.asarray(theta_log, float)


# ----------------------------------------------------------------------
# Main search loop
# ----------------------------------------------------------------------
def main():
    np.random.seed(0)

    num_samples = 50
    best_cost = float("inf")
    best_log = None
    best_gains = None

    for i in range(num_samples):
        theta_log = sample_log10_gains()
        gains = log_to_gains(theta_log)
        DSLPIDControl.set_gains_from_vector(gains)

        try:
            t, pos, vel, rpy, omega, des_pos, des_vel, des_rpy = (
                run_episode_and_extract_logs()
            )
            cost = episode_cost(t, pos, vel, rpy, omega, des_pos, des_vel, des_rpy)

        except Exception as e:
            print("[random_search] Exception:", repr(e))
            traceback.print_exc()
            cost = 1e9

        print(f"Sample {i + 1}/{num_samples}")
        print("  theta_log =", theta_log)
        print("  gains     =", gains)
        print("  cost      =", cost)
        print()

        if cost < best_cost:
            best_cost = cost
            best_log = theta_log.copy()
            best_gains = gains.copy()
            print("  >>> New best cost:", best_cost)
            print()

    print("\nFinal best cost:", best_cost)
    print("Best theta_log:", best_log)
    print("Best gains:", best_gains)


if __name__ == "__main__":
    main()
