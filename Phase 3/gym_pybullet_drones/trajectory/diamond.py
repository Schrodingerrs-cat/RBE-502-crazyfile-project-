# import numpy as np


# def M(t):
#     return np.array(
#         [
#             [1, t, t**2, t**3, t**4, t**5],
#             [0, 1, 2 * t, 3 * t**2, 4 * t**3, 5 * t**4],
#             [0, 0, 2, 6 * t, 12 * t**2, 20 * t**3],
#         ]
#     )


# def solve_poly(p0, v0, a0, pf, vf, af, t0, tf):
#     A = np.vstack((M(t0), M(tf)))
#     b = np.array([p0, v0, a0, pf, vf, af])
#     return np.linalg.solve(A, b)


# def eval_poly(coeff, t):
#     T = np.array([1, t, t**2, t**3, t**4, t**5])
#     Td = np.array([0, 1, 2 * t, 3 * t**2, 4 * t**3, 5 * t**4])
#     Tdd = np.array([0, 0, 2, 6 * t, 12 * t**2, 20 * t**3])
#     return T @ coeff, Td @ coeff, Tdd @ coeff


# def diamond(t, tfinal=8):
#     """
#     One-shot diamond trajectory.
#     4 polynomial segments, each tfinal/4.
#     After t > tfinal: drone hovers at (0,0,0.5).
#     """

#     # After diamond completes → hover at (0,0,0.5)
#     if t >= tfinal:
#         return {
#             "pos": np.array([0.0, 0.0, 0.5]),
#             "vel": np.zeros(3),
#             "acc": np.zeros(3),
#             "jerk": np.zeros(3),
#             "yaw": 0.0,
#             "yawdot": 0.0,
#         }

#     # Waypoints in y,z for diamond
#     u = 1 / np.sqrt(2)
#     p = [
#         np.array([0, 0]),
#         np.array([u, u]),
#         np.array([0, 2 * u]),
#         np.array([-u, u]),
#         np.array([0, 0]),
#     ]

#     segT = tfinal / 4

#     # Segment selection
#     if t <= segT:
#         i, t0, tf = 0, 0, segT
#     elif t <= 2 * segT:
#         i, t0, tf = 1, segT, 2 * segT
#     elif t <= 3 * segT:
#         i, t0, tf = 2, 2 * segT, 3 * segT
#     else:
#         i, t0, tf = 3, 3 * segT, tfinal

#     # Y polynomial
#     cy = solve_poly(p[i][0], 0, 0, p[i + 1][0], 0, 0, t0, tf)
#     y, vy, ay = eval_poly(cy, t)

#     # Z polynomial
#     cz = solve_poly(p[i][1], 0, 0, p[i + 1][1], 0, 0, t0, tf)
#     z, vz, az = eval_poly(cz, t)

#     # X moves linearly 0 → 1 over tfinal
#     x = t / tfinal
#     vx = 1 / tfinal
#     ax = 0

#     pos = np.array([x, y, z])
#     vel = np.array([vx, vy, vz])
#     acc = np.array([ax, ay, az])

#     return {
#         "pos": pos,
#         "vel": vel,
#         "acc": acc,
#         "jerk": np.zeros(3),
#         "yaw": 0,
#         "yawdot": 0,
#     }


import numpy as np


def M(t):
    return np.array(
        [
            [1, t, t**2, t**3, t**4, t**5],
            [0, 1, 2 * t, 3 * t**2, 4 * t**3, 5 * t**4],
            [0, 0, 2, 6 * t, 12 * t**2, 20 * t**3],
        ]
    )


def solve_poly(p0, v0, a0, pf, vf, af, t0, tf):
    A = np.vstack((M(t0), M(tf)))
    b = np.array([p0, v0, a0, pf, vf, af])
    return np.linalg.solve(A, b)


def eval_poly(coeff, t):
    T = np.array([1, t, t**2, t**3, t**4, t**5])
    Td = np.array([0, 1, 2 * t, 3 * t**2, 4 * t**3, 5 * t**4])
    Tdd = np.array([0, 0, 2, 6 * t, 12 * t**2, 20 * t**3])
    return T @ coeff, Td @ coeff, Tdd @ coeff


def diamond(t, tfinal=8):
    """
    4 polynomial segments, each 2 seconds.
    Diamond defined in y-z plane:
        p0 → p1 → p2 → p3 → p4
    x(t) moves linearly from 0 → 1.
    """
    u = 1 / np.sqrt(2)

    # y,z waypoints
    p = [
        np.array([0, 0]),
        np.array([u, u]),
        np.array([0, 2 * u]),
        np.array([-u, u]),
        np.array([0, 0]),
    ]

    segT = tfinal / 4

    if t <= segT:
        i = 0
        t0 = 0
        tf = segT
    elif t <= 2 * segT:
        i = 1
        t0 = segT
        tf = 2 * segT
    elif t <= 3 * segT:
        i = 2
        t0 = 2 * segT
        tf = 3 * segT
    else:
        i = 3
        t0 = 3 * segT
        tf = tfinal

    # Solve polynomial in Y
    cy = solve_poly(p[i][0], 0, 0, p[i + 1][0], 0, 0, t0, tf)
    y, vy, ay = eval_poly(cy, t)

    # Solve polynomial in Z
    cz = solve_poly(p[i][1], 0, 0, p[i + 1][1], 0, 0, t0, tf)
    z, vz, az = eval_poly(cz, t)

    # X moves linearly
    x = t / tfinal
    vx = 1 / tfinal
    ax = 0

    pos = np.array([x, y, z])
    vel = np.array([vx, vy, vz])
    acc = np.array([ax, ay, az])

    return {
        "pos": pos,
        "vel": vel,
        "acc": acc,
        "jerk": np.zeros(3),
        "yaw": 0,
        "yawdot": 0,
    }
