import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from trajectory.circle import circle
from trajectory.diamond import diamond


def sample_trajectory(traj_func, tf, dt=0.01):
    """
    Samples a trajectory function from t=0 to t=tf with time step dt.

    Returns:
        t_vals: time array
        pos, vel, acc: Nx3 arrays for position, velocity, acceleration
    """
    t_vals = np.arange(0, tf, dt)

    pos_list = []
    vel_list = []
    acc_list = []

    for t in t_vals:
        st = traj_func(t)
        pos_list.append(st["pos"])
        vel_list.append(st["vel"])
        acc_list.append(st["acc"])

    pos = np.array(pos_list)
    vel = np.array(vel_list)
    acc = np.array(acc_list)

    return t_vals, pos, vel, acc


def plot_trajectory(t, pos, vel, acc, title_prefix="Trajectory"):
    """Generates 2D plots for x,y,z and their derivatives."""
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    # ---------- Position ----------
    axs[0].plot(t, pos[:, 0], label="x(t)")
    axs[0].plot(t, pos[:, 1], label="y(t)")
    axs[0].plot(t, pos[:, 2], label="z(t)")
    axs[0].set_title(f"{title_prefix}: Position")
    axs[0].set_ylabel("meters")
    axs[0].legend()
    axs[0].grid()

    # ---------- Velocity ----------
    axs[1].plot(t, vel[:, 0], label="vx(t)")
    axs[1].plot(t, vel[:, 1], label="vy(t)")
    axs[1].plot(t, vel[:, 2], label="vz(t)")
    axs[1].set_title(f"{title_prefix}: Velocity")
    axs[1].set_ylabel("m/s")
    axs[1].legend()
    axs[1].grid()

    # ---------- Acceleration ----------
    axs[2].plot(t, acc[:, 0], label="ax(t)")
    axs[2].plot(t, acc[:, 1], label="ay(t)")
    axs[2].plot(t, acc[:, 2], label="az(t)")
    axs[2].set_title(f"{title_prefix}: Acceleration")
    axs[2].set_xlabel("time (s)")
    axs[2].set_ylabel("m/s^2")
    axs[2].legend()
    axs[2].grid()

    plt.tight_layout()
    plt.show()


def plot_3d(pos, title_prefix="Trajectory 3D"):
    """Generates a 3D plot of the path."""
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], linewidth=2)

    ax.set_title(f"{title_prefix}: 3D Path")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.grid()

    # equal aspect ratio
    max_range = (
        np.array(
            [
                pos[:, 0].max() - pos[:, 0].min(),
                pos[:, 1].max() - pos[:, 1].min(),
                pos[:, 2].max() - pos[:, 2].min(),
            ]
        ).max()
        / 2.0
    )
    mid_x = (pos[:, 0].max() + pos[:, 0].min()) * 0.5
    mid_y = (pos[:, 1].max() + pos[:, 1].min()) * 0.5
    mid_z = (pos[:, 2].max() + pos[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()


if __name__ == "__main__":
    # ---------------------
    # CIRCLE TRAJECTORY
    # ---------------------
    print("Sampling circle trajectory...")
    t_circle, pos_circle, vel_circle, acc_circle = sample_trajectory(circle, tf=15.0)

    plot_trajectory(
        t_circle, pos_circle, vel_circle, acc_circle, title_prefix="Circular Trajectory"
    )

    plot_3d(pos_circle, title_prefix="Circular Trajectory")

    # ---------------------
    # DIAMOND TRAJECTORY
    # ---------------------
    print("Sampling diamond trajectory...")
    t_diamond, pos_diamond, vel_diamond, acc_diamond = sample_trajectory(
        diamond, tf=8.0
    )

    plot_trajectory(
        t_diamond,
        pos_diamond,
        vel_diamond,
        acc_diamond,
        title_prefix="Diamond Trajectory",
    )

    plot_3d(pos_diamond, title_prefix="Diamond Trajectory")
