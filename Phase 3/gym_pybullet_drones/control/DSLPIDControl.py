import math

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel


class DSLPIDControl(BaseControl):
    """PD position + PD attitude controller for CF2X/CF2P.

    Outer loop (position):
        T_d = K_p e_r + K_d e_v + m a_d + F_g

    Inner loop (attitude on SO(3)):
        τ = -K_p,att e_R - K_d,att e_ω

    Motor mixing:
        PWM_i = PWM_base + mixer_row_i · τ
    """

    ###########################################################################
    def __init__(self, drone_model: DroneModel, g: float = 9.8):
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL not in [DroneModel.CF2X, DroneModel.CF2P]:
            print("[ERROR] Only CF2X or CF2P supported.")
            exit()

        # ---- Physical parameters from URDF ----
        self.mass = float(self._getURDFParameter("m"))
        self.Ixx = float(self._getURDFParameter("ixx"))
        self.Iyy = float(self._getURDFParameter("iyy"))
        self.Izz = float(self._getURDFParameter("izz"))
        self.I = np.diag([self.Ixx, self.Iyy, self.Izz])

        # BaseControl stores GRAVITY as total force m*g
        self.g_lin = self.GRAVITY / self.mass  # ≈ 9.8 m/s^2

        # ------------------------------------------------------------------ #
        #                        CONTROLLER GAINS                            #
        # ------------------------------------------------------------------ #
        # Position PD gains (outer loop)
        self.Kp_pos = np.array([2, 2, 2])
        self.Kd_pos = np.array([2.35, 2.35, 2.35])
        self.Kp_att = np.array([4800.0, 4800.0, 4800.0])
        self.Kd_att = np.array([600.0, 600.0, 600.0])
        # self.Kp_pos = np.array([2.0, 2.0, 2.0])
        # self.Kd_pos = np.array([2.35, 2.35, 2.35])

        # # Attitude PD gains (inner loop, roll / pitch / yaw)
        # self.Kp_att = np.array([4800.0, 4800.0, 4800.0])
        # self.Kd_att = np.array([600.0, 600.0, 600.0])

        # ------------------------------------------------------------------ #
        #                        MIXER AND PWM CONVERSION                    #
        # ------------------------------------------------------------------ #
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535

        if self.DRONE_MODEL == DroneModel.CF2X:
            # Standard Crazyflie “X” mixer
            self.MIXER_MATRIX = np.array(
                [
                    [-0.5, -0.5, -1.0],
                    [-0.5, 0.5, 1.0],
                    [0.5, 0.5, -1.0],
                    [0.5, -0.5, 1.0],
                ]
            )
        else:  # CF2P
            self.MIXER_MATRIX = np.array(
                [
                    [0.0, -1.0, -1.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, -1.0],
                    [-1.0, 0.0, 1.0],
                ]
            )

        # Desired rotation from outer loop (R_d)
        self.R_d_des = np.eye(3)

        self.reset()

    ###########################################################################
    def reset(self):
        super().reset()
        self.last_rpy = np.zeros(3)

        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)

        self.R_d_des = np.eye(3)

    ###########################################################################
    def computeControl(
        self,
        control_timestep,
        cur_pos,
        cur_quat,
        cur_vel,
        cur_ang_vel,
        target_pos,
        target_rpy=np.zeros(3),
        target_vel=np.zeros(3),
        target_rpy_rates=np.zeros(3),
        target_acc=np.zeros(3),
    ):
        """Main entry: returns motor RPMs, position error, and desired yaw.

        This implements the cascaded “position → attitude → motor” structure.
        """

        self.control_counter += 1

        # ---------------- OUTER LOOP: position PD -------------------------
        T_d_vec, R_d, pos_e, R_cur = self._outer_position_loop(
            cur_pos=cur_pos,
            cur_quat=cur_quat,
            cur_vel=cur_vel,
            target_pos=target_pos,
            target_vel=target_vel,
            target_acc=target_acc,
            target_rpy=target_rpy,
        )

        # Project desired thrust vector onto current body z-axis
        # a3^B in world frame is the 3rd column of R_cur
        scalar_thrust = max(0.0, float(np.dot(T_d_vec, R_cur[:, 2])))

        # Convert desired thrust (Newton) to a PWM baseline
        # F_total = sum_i KF * RPM_i^2, assume equal share: F_total ≈ 4 KF RPM^2
        pwm_base = (
            math.sqrt(scalar_thrust / (4.0 * self.KF)) - self.PWM2RPM_CONST
        ) / self.PWM2RPM_SCALE

        # ---------------- INNER LOOP: attitude PD -------------------------
        rpm = self._inner_attitude_loop(
            control_timestep=control_timestep,
            pwm_base=pwm_base,
            cur_quat=cur_quat,
            cur_ang_vel=cur_ang_vel,
            R_d=R_d,
            target_rpy_rates=target_rpy_rates,
        )

        # For logging we return desired yaw from R_d
        des_rpy = Rotation.from_matrix(R_d).as_euler("xyz", degrees=False)
        return rpm, pos_e, des_rpy

    ###########################################################################
    #                           OUTER LOOP                                   #
    ###########################################################################
    def _outer_position_loop(
        self,
        cur_pos,
        cur_quat,
        cur_vel,
        target_pos,
        target_vel,
        target_acc,
        target_rpy,
    ):
        """Implements Eq. (3), (4)–(9), (10)–(15) in the handout."""

        # Current rotation R (world←body)
        R = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)

        # Errors
        e_r = target_pos - cur_pos
        e_v = target_vel - cur_vel

        # Feedforward + PD acceleration (a_d)
        a_d = target_acc + self.Kp_pos * e_r + self.Kd_pos * e_v

        # Gravity compensation
        e3 = np.array([0.0, 0.0, 1.0])
        F_g = self.mass * self.g_lin * e3

        # Desired total thrust vector in world frame
        T_d_vec = self.mass * a_d + F_g

        # Desired body z-axis (11)
        T_norm = np.linalg.norm(T_d_vec)
        if T_norm < 1e-6:
            z_B_d = e3.copy()
        else:
            z_B_d = T_d_vec / T_norm

        # Desired heading x_C^d from yaw ψ_d (12)
        psi_d = target_rpy[2]
        x_C_d = np.array([math.cos(psi_d), math.sin(psi_d), 0.0])

        # Avoid degeneracy when z_B_d almost parallel to x_C_d
        if abs(np.dot(x_C_d, z_B_d)) > 0.99:
            x_C_d = np.array([1.0, 0.0, 0.0])

        # Orthogonal basis via Gram–Schmidt (13)–(15)
        y_B_d = np.cross(z_B_d, x_C_d)
        y_B_d /= np.linalg.norm(y_B_d)
        x_B_d = np.cross(y_B_d, z_B_d)

        R_d = np.column_stack((x_B_d, y_B_d, z_B_d))
        self.R_d_des = R_d

        return T_d_vec, R_d, e_r, R

    ###########################################################################
    #                           INNER LOOP                                   #
    ###########################################################################
    def _inner_attitude_loop(
        self,
        control_timestep,
        pwm_base,
        cur_quat,
        cur_ang_vel,
        R_d,
        target_rpy_rates,
    ):
        """Implements Eq. (16)–(21): attitude PD on SO(3), then motor mixing."""

        # Current rotation and angular velocity
        R = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        omega = cur_ang_vel

        # Rotation error R_e and vee-map e_R (16)–(19)
        R_e = R_d.T @ R
        skew = 0.5 * (R_e - R_e.T)
        e_R = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])

        # Desired angular velocity in body frame (usually zero) (20)
        omega_d_des = target_rpy_rates  # [p_d, q_d, r_d]
        omega_d = R.T @ (R_d @ omega_d_des)
        e_omega = omega - omega_d

        # PD torques (21)
        tau = -self.Kp_att * e_R - self.Kd_att * e_omega - 0.05 * omega
        tau = np.clip(tau, -3200.0, 3200.0)

        # Motor mixing: PWM_i = pwm_base + mixer_row_i · tau
        pwm = pwm_base + self.MIXER_MATRIX @ tau
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)

        # Convert PWM → RPM
        rpm = self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
        return rpm

    ###########################################################################
    #                    Utility: 1, 2, or 3D thrust interface                #
    ###########################################################################
    def _one23DInterface(self, thrust):
        """Utility to convert 1, 2, or 3D thrust inputs into 4 motor PWM values."""

        DIM = len(np.array(thrust))
        pwm = np.clip(
            (np.sqrt(np.array(thrust) / (self.KF * (4 / DIM))) - self.PWM2RPM_CONST)
            / self.PWM2RPM_SCALE,
            self.MIN_PWM,
            self.MAX_PWM,
        )
        if DIM in [1, 4]:
            return np.repeat(pwm, 4 / DIM)
        elif DIM == 2:
            return np.hstack([pwm, np.flip(pwm)])
        else:
            print("[ERROR] in DSLPIDControl._one23DInterface()")
            exit()


# import math

# import numpy as np
# import pybullet as p
# from scipy.spatial.transform import Rotation

# from gym_pybullet_drones.control.BaseControl import BaseControl
# from gym_pybullet_drones.utils.enums import DroneModel


# class DSLPIDControl(BaseControl):
#     """PD position + PD attitude controller for CF2X/CF2P.

#     Outer loop (position):
#         T_d = m * (a_d + g*e3)

#     Inner loop (attitude on SO(3)):
#         τ = -Kp_att * e_R - Kd_att * e_ω

#     Motor mixing:
#         PWM_i = PWM_base + mixer_row_i · τ
#     """

#     ###########################################################################
#     def __init__(self, drone_model: DroneModel, g: float = 9.8):
#         super().__init__(drone_model=drone_model, g=g)
#         if self.DRONE_MODEL not in [DroneModel.CF2X, DroneModel.CF2P]:
#             print("[ERROR] DSLPIDControl supports only CF2X or CF2P.")
#             exit()

#         # ---- Physical parameters from URDF ----
#         self.mass = float(self._getURDFParameter("m"))
#         self.Ixx = float(self._getURDFParameter("ixx"))
#         self.Iyy = float(self._getURDFParameter("iyy"))
#         self.Izz = float(self._getURDFParameter("izz"))
#         self.I = np.diag([self.Ixx, self.Iyy, self.Izz])

#         # BaseControl stores GRAVITY as total force m*g
#         self.g_lin = self.GRAVITY / self.mass  # ≈ 9.8 m/s^2

#         # ------------------------------------------------------------------ #
#         #                        CONTROLLER GAINS                            #
#         #  Tuned to work for both hover and smooth circle tracking.         #
#         # ------------------------------------------------------------------ #
#         # Position PD gains (outer loop)
#         self.Kp_pos = np.array([3.0, 3.0, 4.0])
#         self.Kd_pos = np.array([3.0, 3.0, 3.0])

#         # Attitude PD gains (inner loop, roll / pitch / yaw)
#         self.Kp_att = np.array([6000.0, 6000.0, 3000.0])
#         self.Kd_att = np.array([800.0, 800.0, 500.0])

#         # ------------------------------------------------------------------ #
#         #                        MIXER AND PWM CONVERSION                    #
#         # ------------------------------------------------------------------ #
#         self.PWM2RPM_SCALE = 0.2685
#         self.PWM2RPM_CONST = 4070.3
#         self.MIN_PWM = 20000
#         self.MAX_PWM = 65535

#         if self.DRONE_MODEL == DroneModel.CF2X:
#             # Standard Crazyflie “X” mixer
#             self.MIXER_MATRIX = np.array(
#                 [
#                     [-0.5, -0.5, -1.0],
#                     [-0.5, 0.5, 1.0],
#                     [0.5, 0.5, -1.0],
#                     [0.5, -0.5, 1.0],
#                 ]
#             )
#         else:  # CF2P
#             self.MIXER_MATRIX = np.array(
#                 [
#                     [0.0, -1.0, -1.0],
#                     [1.0, 0.0, 1.0],
#                     [0.0, 1.0, -1.0],
#                     [-1.0, 0.0, 1.0],
#                 ]
#             )

#         # Desired rotation from outer loop (R_d)
#         self.R_d_des = np.eye(3)

#         self.reset()

#     ###########################################################################
#     def reset(self):
#         """Reset control internal state."""
#         super().reset()
#         self.last_rpy = np.zeros(3)

#         self.last_pos_e = np.zeros(3)
#         self.integral_pos_e = np.zeros(3)
#         self.last_rpy_e = np.zeros(3)
#         self.integral_rpy_e = np.zeros(3)

#         self.R_d_des = np.eye(3)

#     ###########################################################################
#     def computeControl(
#         self,
#         control_timestep,
#         cur_pos,
#         cur_quat,
#         cur_vel,
#         cur_ang_vel,
#         target_pos,
#         target_rpy=np.zeros(3),
#         target_vel=np.zeros(3),
#         target_rpy_rates=np.zeros(3),
#         target_acc=np.zeros(3),
#     ):
#         """Main entry: returns motor RPMs, position error, and desired rpy.

#         This implements the cascaded “position → attitude → motor” structure.
#         """

#         self.control_counter += 1

#         # ---------------- OUTER LOOP: position PD -------------------------
#         T_d_vec, R_d, pos_e, R_cur = self._outer_position_loop(
#             cur_pos=cur_pos,
#             cur_quat=cur_quat,
#             cur_vel=cur_vel,
#             target_pos=target_pos,
#             target_vel=target_vel,
#             target_acc=target_acc,
#             target_rpy=target_rpy,
#         )

#         # Physically correct scalar thrust: projection of desired thrust
#         # onto current body z-axis (3rd column of R_cur).
#         scalar_thrust = max(0.0, float(np.dot(T_d_vec, R_cur[:, 2])))

#         # Convert desired scalar thrust (Newton) to a PWM baseline
#         # F_total ≈ sum_i KF * RPM_i^2, assume equal share: F_total ≈ 4 * KF * RPM^2
#         # Invert the CF firmware's PWM→RPM mapping.
#         pwm_base = (
#             math.sqrt(scalar_thrust / (4.0 * self.KF)) - self.PWM2RPM_CONST
#         ) / self.PWM2RPM_SCALE

#         # ---------------- INNER LOOP: attitude PD -------------------------
#         rpm = self._inner_attitude_loop(
#             control_timestep=control_timestep,
#             pwm_base=pwm_base,
#             cur_quat=cur_quat,
#             cur_ang_vel=cur_ang_vel,
#             R_d=R_d,
#             target_rpy_rates=target_rpy_rates,
#         )

#         # For logging we return desired rpy from R_d
#         des_rpy = Rotation.from_matrix(R_d).as_euler("xyz", degrees=False)
#         return rpm, pos_e, des_rpy

#     ###########################################################################
#     #                           OUTER LOOP                                   #
#     ###########################################################################
#     def _outer_position_loop(
#         self,
#         cur_pos,
#         cur_quat,
#         cur_vel,
#         target_pos,
#         target_vel,
#         target_acc,
#         target_rpy,
#     ):
#         """Position loop: computes desired thrust vector and desired attitude."""

#         # Current rotation R (world←body)
#         R = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)

#         # Errors
#         e_r = target_pos - cur_pos
#         e_v = target_vel - cur_vel

#         # Feedforward + PD acceleration (a_d)
#         a_d = target_acc + self.Kp_pos * e_r + self.Kd_pos * e_v

#         # Gravity compensation
#         e3 = np.array([0.0, 0.0, 1.0])
#         F_g = self.mass * self.g_lin * e3

#         # Desired total thrust vector in world frame
#         T_d_vec = self.mass * a_d + F_g

#         # Desired body z-axis (eq. 11 type)
#         T_norm = np.linalg.norm(T_d_vec)
#         if T_norm < 1e-6:
#             z_B_d = e3.copy()
#         else:
#             z_B_d = T_d_vec / T_norm

#         # Desired heading x_C^d from yaw ψ_d
#         psi_d = target_rpy[2]
#         x_C_d = np.array([math.cos(psi_d), math.sin(psi_d), 0.0])

#         # Avoid degeneracy when z_B_d almost parallel to x_C_d
#         if abs(np.dot(x_C_d, z_B_d)) > 0.99:
#             x_C_d = np.array([1.0, 0.0, 0.0])

#         # Orthogonal basis via Gram–Schmidt
#         y_B_d = np.cross(z_B_d, x_C_d)
#         y_B_d /= np.linalg.norm(y_B_d)
#         x_B_d = np.cross(y_B_d, z_B_d)

#         R_d = np.column_stack((x_B_d, y_B_d, z_B_d))
#         self.R_d_des = R_d

#         return T_d_vec, R_d, e_r, R

#     ###########################################################################
#     #                           INNER LOOP                                   #
#     ###########################################################################
#     def _inner_attitude_loop(
#         self,
#         control_timestep,
#         pwm_base,
#         cur_quat,
#         cur_ang_vel,
#         R_d,
#         target_rpy_rates,
#     ):
#         """Attitude PD on SO(3), then motor mixing."""

#         # Current rotation and angular velocity
#         R = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
#         omega = cur_ang_vel

#         # Rotation error R_e and vee-map e_R
#         R_e = R_d.T @ R
#         skew = 0.5 * (R_e - R_e.T)
#         e_R = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])

#         # Desired angular velocity in body frame (usually zero)
#         omega_d_des = target_rpy_rates  # [p_d, q_d, r_d]
#         omega_d = R.T @ (R_d @ omega_d_des)
#         e_omega = omega - omega_d

#         # PD torques
#         tau = -self.Kp_att * e_R - self.Kd_att * e_omega
#         tau = np.clip(tau, -3200.0, 3200.0)

#         # Motor mixing: PWM_i = pwm_base + mixer_row_i · τ
#         pwm = pwm_base + self.MIXER_MATRIX @ tau
#         pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)

#         # Convert PWM → RPM
#         rpm = self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
#         return rpm

#     ###########################################################################
#     #                    Utility: 1, 2, or 3D thrust interface                #
#     ###########################################################################
#     def _one23DInterface(self, thrust):
#         """Utility to convert 1, 2, or 3D thrust inputs into 4 motor PWM values."""

#         DIM = len(np.array(thrust))
#         pwm = np.clip(
#             (np.sqrt(np.array(thrust) / (self.KF * (4 / DIM))) - self.PWM2RPM_CONST)
#             / self.PWM2RPM_SCALE,
#             self.MIN_PWM,
#             self.MAX_PWM,
#         )
#         if DIM in [1, 4]:
#             return np.repeat(pwm, int(4 / DIM))
#         elif DIM == 2:
#             return np.hstack([pwm, np.flip(pwm)])
#         else:
#             print("[ERROR] in DSLPIDControl._one23DInterface()")
#             exit()


# # ------------------------------------------------------------------ #
# #                  RL tuning                                         #
# # ------------------------------------------------------------------ #
# import math

# import numpy as np
# import pybullet as p
# from scipy.spatial.transform import Rotation

# from gym_pybullet_drones.control.BaseControl import BaseControl
# from gym_pybullet_drones.utils.enums import DroneModel


# class DSLPIDControl(BaseControl):
#     """PID control class for Crazyflies (CF2X/CF2P).

#     Outer loop: position → thrust vector and desired attitude
#     Inner loop: attitude control on SO(3) → motor torques → PWM → RPM

#     This version matches your original stable controller but adds a
#     class-level interface `set_gains_from_vector()` so the tuner
#     can change the PD gains between episodes.
#     """

#     # ------------------------------------------------------------------
#     # Class-level default gains (can be overwritten by tuner)
#     #
#     # Order: kp_pos_xy, kp_pos_z, kd_pos_xy, kd_pos_z,
#     #        kp_att_xy, kp_att_yaw, kd_att_xy, kd_att_yaw
#     # ------------------------------------------------------------------
#     KP_POS = np.array([2.0, 2.0, 2.0])
#     KD_POS = np.array([2.35, 2.35, 2.35])
#     KP_ATT = np.array([4800.0, 4800.0, 4800.0])
#     KD_ATT = np.array([600.0, 600.0, 600.0])

#     ####################################################################
#     @classmethod
#     def set_gains_from_vector(cls, gains: np.ndarray):
#         """Set position and attitude gains from an 8D vector.

#         gains = [
#             kp_pos_xy,
#             kp_pos_z,
#             kd_pos_xy,
#             kd_pos_z,
#             kp_att_xy,
#             kp_att_yaw,
#             kd_att_xy,
#             kd_att_yaw,
#         ]
#         """
#         gains = np.asarray(gains, dtype=float)
#         if gains.shape[0] != 8:
#             raise ValueError(
#                 "gains must be length 8: "
#                 "[kp_pos_xy, kp_pos_z, kd_pos_xy, kd_pos_z, "
#                 " kp_att_xy, kp_att_yaw, kd_att_xy, kd_att_yaw]"
#             )

#         (
#             kp_pos_xy,
#             kp_pos_z,
#             kd_pos_xy,
#             kd_pos_z,
#             kp_att_xy,
#             kp_att_yaw,
#             kd_att_xy,
#             kd_att_yaw,
#         ) = gains

#         # Position gains: [x, y, z]
#         cls.KP_POS = np.array([kp_pos_xy, kp_pos_xy, kp_pos_z], dtype=float)
#         cls.KD_POS = np.array([kd_pos_xy, kd_pos_xy, kd_pos_z], dtype=float)

#         # Attitude gains: [roll, pitch, yaw]
#         cls.KP_ATT = np.array([kp_att_xy, kp_att_xy, kp_att_yaw], dtype=float)
#         cls.KD_ATT = np.array([kd_att_xy, kd_att_xy, kd_att_yaw], dtype=float)

#         print("[DSLPIDControl] Updated gains:")
#         print("  Kp_pos:", cls.KP_POS)
#         print("  Kd_pos:", cls.KD_POS)
#         print("  Kp_att:", cls.KP_ATT)
#         print("  Kd_att:", cls.KD_ATT)

#     ####################################################################
#     def __init__(self, drone_model: DroneModel, g: float = 9.8):
#         """Common control class __init__."""

#         super().__init__(drone_model=drone_model, g=g)
#         if self.DRONE_MODEL not in [DroneModel.CF2X, DroneModel.CF2P]:
#             print(
#                 "[ERROR] in DSLPIDControl.__init__(), "
#                 "DSLPIDControl requires DroneModel.CF2X or DroneModel.CF2P"
#             )
#             exit()

#         # ---- Physical parameters from URDF ----
#         self.mass = float(self._getURDFParameter("m"))
#         self.Ixx = float(self._getURDFParameter("ixx"))
#         self.Iyy = float(self._getURDFParameter("iyy"))
#         self.Izz = float(self._getURDFParameter("izz"))
#         self.I = np.diag([self.Ixx, self.Iyy, self.Izz])

#         # BaseControl stores GRAVITY as m*g (force)
#         self.g_lin = self.GRAVITY / self.mass  # ≈ 9.8

#         # ------------------------------------------------------------------
#         # Gains are read from the class-level defaults (can be tuned)
#         # ------------------------------------------------------------------
#         self.Kp_pos = self.__class__.KP_POS.copy()
#         self.Kd_pos = self.__class__.KD_POS.copy()
#         self.Kp_att = self.__class__.KP_ATT.copy()
#         self.Kd_att = self.__class__.KD_ATT.copy()

#         # ------------------------------------------------------------------
#         # Motor / PWM parameters (do not change)
#         # ------------------------------------------------------------------
#         self.PWM2RPM_SCALE = 0.2685
#         self.PWM2RPM_CONST = 4070.3
#         self.MIN_PWM = 20000
#         self.MAX_PWM = 65535
#         if self.DRONE_MODEL == DroneModel.CF2X:
#             self.MIXER_MATRIX = np.array(
#                 [
#                     [-0.5, -0.5, -1],
#                     [-0.5, 0.5, 1],
#                     [0.5, 0.5, -1],
#                     [0.5, -0.5, 1],
#                 ]
#             )
#         else:  # CF2P
#             self.MIXER_MATRIX = np.array(
#                 [
#                     [0, -1, -1],
#                     [1, 0, 1],
#                     [0, 1, -1],
#                     [-1, 0, 1],
#                 ]
#             )

#         self.R_d_des = np.eye(3)
#         self.reset()

#     ####################################################################
#     def reset(self):
#         """Reset integral and previous errors."""
#         super().reset()
#         self.last_rpy = np.zeros(3)
#         self.last_pos_e = np.zeros(3)
#         self.integral_pos_e = np.zeros(3)
#         self.last_rpy_e = np.zeros(3)
#         self.integral_rpy_e = np.zeros(3)
#         self.R_d_des = np.eye(3)

#     ####################################################################
#     def computeControl(
#         self,
#         control_timestep,
#         cur_pos,
#         cur_quat,
#         cur_vel,
#         cur_ang_vel,
#         target_pos,
#         target_rpy=np.zeros(3),
#         target_vel=np.zeros(3),
#         target_rpy_rates=np.zeros(3),
#         target_acc=np.zeros(3),
#     ):
#         """Compute RPMs, position error, and desired RPY from current state."""

#         self.control_counter += 1

#         mass = self._getURDFParameter("m")
#         target_thrust, computed_target_rpy, pos_e, cur_rotation = (
#             self._dslPIDPositionControl(
#                 control_timestep,
#                 cur_pos,
#                 cur_quat,
#                 cur_vel,
#                 target_pos,
#                 target_rpy,
#                 target_vel,
#                 target_acc,
#                 mass=mass,
#             )
#         )

#         # Project desired thrust vector onto current body z-axis
#         scalar_thrust = max(0.0, float(np.dot(target_thrust, cur_rotation[:, 2])))

#         # Convert scalar thrust → baseline PWM for all 4 motors
#         thrust = (
#             math.sqrt(scalar_thrust / (4 * self.KF)) - self.PWM2RPM_CONST
#         ) / self.PWM2RPM_SCALE

#         # Inner loop: attitude / SO(3)
#         rpm = self._dslPIDAttitudeControl(
#             control_timestep,
#             thrust,
#             cur_quat,
#             cur_ang_vel,
#             computed_target_rpy,
#             target_rpy_rates,
#         )

#         return rpm, pos_e, computed_target_rpy

#     ####################################################################
#     def _dslPIDPositionControl(
#         self,
#         control_timestep,
#         cur_pos,
#         cur_quat,
#         cur_vel,
#         target_pos,
#         target_rpy,
#         target_vel,
#         target_acc,
#         mass=0.29,
#     ):
#         """Outer-loop position control (generates thrust vector and R_d)."""

#         cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)

#         pos_e = target_pos - cur_pos
#         vel_e = target_vel - cur_vel

#         a_des = target_acc + self.Kp_pos * pos_e + self.Kd_pos * vel_e

#         # Gravity compensation
#         e3 = np.array([0.0, 0.0, 1.0])
#         T_d = mass * (self.g_lin * e3 + a_des)

#         T_mag = np.linalg.norm(T_d)
#         if T_mag < 1e-6:
#             z_B_d = e3.copy()
#         else:
#             z_B_d = T_d / T_mag

#         psi_d = target_rpy[2]
#         x_C_d = np.array([math.cos(psi_d), math.sin(psi_d), 0.0])

#         # Avoid degeneracy when z_B_d almost parallel to x_C_d
#         if abs(np.dot(x_C_d, z_B_d)) > 0.99:
#             x_C_d = np.array([1.0, 0.0, 0.0])

#         y_B_d = np.cross(z_B_d, x_C_d)
#         y_B_d /= np.linalg.norm(y_B_d)
#         x_B_d = np.cross(y_B_d, z_B_d)

#         R_d = np.column_stack((x_B_d, y_B_d, z_B_d))
#         self.R_d_des = R_d

#         target_rpy_out = Rotation.from_matrix(R_d).as_euler("xyz", degrees=False)

#         return T_d, target_rpy_out, pos_e, cur_rotation

#     ####################################################################
#     def _dslPIDAttitudeControl(
#         self,
#         control_timestep,
#         thrust,
#         cur_quat,
#         cur_ang_vel,
#         target_euler,
#         target_rpy_rates,
#     ):
#         """Inner-loop attitude control on SO(3)."""

#         R = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)

#         # Desired rotation from outer loop
#         R_d = self.R_d_des

#         R_e = R_d.T @ R
#         skew = 0.5 * (R_e - R_e.T)
#         e_R = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])

#         omega = cur_ang_vel
#         omega_d_desired = target_rpy_rates  # usually [0, 0, 0]
#         omega_d = R.T @ (R_d @ omega_d_desired)
#         e_omega = omega - omega_d

#         target_torques = -self.Kp_att * e_R - self.Kd_att * e_omega

#         # Saturate torques
#         target_torques = np.clip(target_torques, -3200, 3200)

#         # Mixer: PWM = thrust + M * tau
#         pwm = thrust + self.MIXER_MATRIX @ target_torques
#         pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)

#         return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST

#     ####################################################################
#     def _one23DInterface(self, thrust):
#         """Utility to convert 1, 2, or 4D thrust commands into 4 PWM values."""

#         DIM = len(np.array(thrust))
#         pwm = np.clip(
#             (np.sqrt(np.array(thrust) / (self.KF * (4 / DIM))) - self.PWM2RPM_CONST)
#             / self.PWM2RPM_SCALE,
#             self.MIN_PWM,
#             self.MAX_PWM,
#         )
#         if DIM in [1, 4]:
#             return np.repeat(pwm, 4 / DIM)
#         elif DIM == 2:
#             return np.hstack([pwm, np.flip(pwm)])
#         else:
#             print("[ERROR] in DSLPIDControl._one23DInterface()")
#             exit()
