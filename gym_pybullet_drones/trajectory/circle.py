import numpy as np


# ---------------- Polynomial Basis Matrix ---------------- #
def M(t):
    return np.array(
        [
            [1, t, t**2, t**3, t**4, t**5],
            [0, 1, 2 * t, 3 * t**2, 4 * t**3, 5 * t**4],
            [0, 0, 2, 6 * t, 12 * t**2, 20 * t**3],
        ]
    )


def solve_poly(p0, v0, a0, pf, vf, af, t0, tf):
    A = np.vstack((M(t0), M(tf)))  # 6×6
    b = np.array([p0, v0, a0, pf, vf, af])
    return np.linalg.solve(A, b)


def eval_poly(coeff, t):
    T = np.array([1, t, t**2, t**3, t**4, t**5])
    Td = np.array([0, 1, 2 * t, 3 * t**2, 4 * t**3, 5 * t**4])
    Tdd = np.array([0, 0, 2, 6 * t, 12 * t**2, 20 * t**3])
    return T @ coeff, Td @ coeff, Tdd @ coeff


# ------------------- MAIN TRAJECTORY --------------------- #
def circle(t, tf=15):
    """
    3-phase trajectory strictly following the assignment:
        Phase 1: Hover → Circle start (0–5 s)
        Phase 2: Circular trajectory (5–10 s)
        Phase 3: Return to hover (10–15 s)
    """

    # Phase timings
    t1, t2, t3 = 5, 10, 15

    # ---------------------- Phase 1 ---------------------- #
    if t <= t1:
        start = np.array([0, 0, 0.5])
        end = np.array([1, 0, 1])

        cx = solve_poly(start[0], 0, 0, end[0], 0, 0, 0, t1)
        cy = solve_poly(start[1], 0, 0, end[1], 0, 0, 0, t1)
        cz = solve_poly(start[2], 0, 0, end[2], 0, 0, 0, t1)

        x, vx, ax = eval_poly(cx, t)
        y, vy, ay = eval_poly(cy, t)
        z, vz, az = eval_poly(cz, t)

        return {
            "pos": np.array([x, y, z]),
            "vel": np.array([vx, vy, vz]),
            "acc": np.array([ax, ay, az]),
            "jerk": np.zeros(3),
            "yaw": 0,
            "yawdot": 0,
        }

    # ---------------------- Phase 2 ---------------------- #
    if t1 < t <= t2:
        R = 1
        z = 1
        w = 2 * np.pi / (t2 - t1)
        tc = t - t1

        # FIXED: phase-shift so start and end points differ
        theta0 = -np.pi / 2
        theta = w * tc + theta0

        x = R * np.cos(theta)
        y = R * np.sin(theta)

        vx = -R * w * np.sin(theta)
        vy = R * w * np.cos(theta)
        vz = 0

        ax = -R * (w**2) * np.cos(theta)
        ay = -R * (w**2) * np.sin(theta)
        az = 0

        return {
            "pos": np.array([x, y, z]),
            "vel": np.array([vx, vy, vz]),
            "acc": np.array([ax, ay, az]),
            "jerk": np.zeros(3),
            "yaw": 0,
            "yawdot": 0,
        }

    # ---------------------- Phase 3 ---------------------- #
    R = 1
    w = 2 * np.pi / (t2 - t1)
    theta0 = -np.pi / 2
    theta_end = w * (t2 - t1) + theta0

    end_circle = np.array([R * np.cos(theta_end), R * np.sin(theta_end), 1])
    hover = np.array([0, 0, 0.5])

    cx = solve_poly(end_circle[0], 0, 0, hover[0], 0, 0, t2, t3)
    cy = solve_poly(end_circle[1], 0, 0, hover[1], 0, 0, t2, t3)
    cz = solve_poly(end_circle[2], 0, 0, hover[2], 0, 0, t2, t3)

    x, vx, ax = eval_poly(cx, t)
    y, vy, ay = eval_poly(cy, t)
    z, vz, az = eval_poly(cz, t)

    return {
        "pos": np.array([x, y, z]),
        "vel": np.array([vx, vy, vz]),
        "acc": np.array([ax, ay, az]),
        "jerk": np.zeros(3),
        "yaw": 0,
        "yawdot": 0,
    }
