import math

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel


class DSLPIDControl(BaseControl):
    """PID control class for Crazyflies.

    Contributors: SiQi Zhou, James Xu, Tracy Du, Mario Vukosavljev, Calvin Ngan, and Jingyuan Hou.

    """

    ################################################################################

    def __init__(self, drone_model: DroneModel, g: float = 9.8):
        """Common control classes __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.

        """
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL != DroneModel.CF2X and self.DRONE_MODEL != DroneModel.CF2P:
            print(
                "[ERROR] in DSLPIDControl.__init__(), DSLPIDControl requires DroneModel.CF2X or DroneModel.CF2P"
            )
            exit()

        self.mass = float(self._getURDFParameter("m"))
        self.Ixx = float(self._getURDFParameter("ixx"))
        self.Iyy = float(self._getURDFParameter("iyy"))
        self.Izz = float(self._getURDFParameter("izz"))
        self.I = np.diag([self.Ixx, self.Iyy, self.Izz])

        # BaseControl stores GRAVITY as m*g (force). Get linear gravity:
        self.g_lin = self.GRAVITY / self.mass  # ≈ 9.8

        """ These gains started from the best set (given below) discovered by the RL tuner.
        I then manually fine-tuned them for smoother damping and faster convergence.
        With these adjustments, the system settles in ~3.0 s - 3.3 s , with near-zero steady-state error. """
        self.Kp_pos = np.array([2, 2, 2])
        self.Kd_pos = np.array([2.35, 2.35, 2.35])
        self.Kp_att = np.array([4800.0, 4800.0, 4800.0])
        self.Kd_att = np.array([600.0, 600.0, 600.0])

        """These PD gains were tuned using a custom RL loop.
        The RL agent proposes (Kp, Kd) values by running the drone simulation and it receives a reward based on:
             • Fast convergence (reward if error → 0 in < ~4 seconds)
             • Penalties for overshoot
             • Penalties for oscillations
             • Penalties for taking too long (> 5 seconds)

         The RL search converged on gains which currently gave:
             - x-axis settles in ~3.0 s (≈0 error)
             - y-axis settles in ~3.0–4.0 s (≈0 error)
             - z-axis settles in ~4.5 s (≈0 error)

         (Note: My earlier manually-tuned gains are still included below as comments
         for reference.)
        """
        # self.Kp_pos = np.array(
        #     [
        #         2.00000000,
        #         1.18139780,
        #         2.20000000,
        #     ]
        # )

        # self.Kd_pos = np.array(
        #     [
        #         2.33988142,
        #         1.89824841,
        #         1.97971997,
        #     ]
        # )

        # self.Kp_att = np.array(
        #     [
        #         4814.15005028,
        #         4814.15005028,
        #         2647.78252766,
        #     ]
        # )

        # self.Kd_att = np.array(
        #     [
        #         600.0,
        #         600.0,
        #         330.0,
        #     ]
        # )

        """Manually Tuned Gains (approx. 5.15 seconds. with error approx. 0.03)"""
        # self.Kp_pos = np.array([0.325, 0.325, 1.3])
        # self.Kd_pos = np.array([0.98, 0.98, 2.0])
        # self.Kp_att = np.array([800.0, 800.0, 420.0])
        # self.Kd_att = np.array([900.0, 900.0, 450.0])

        self.R_d_des = np.eye(3)

        ######################################################
        # Do not change these parameters below
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MIXER_MATRIX = np.array(
                [[-0.5, -0.5, -1], [-0.5, 0.5, 1], [0.5, 0.5, -1], [0.5, -0.5, 1]]
            )
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array(
                [[0, -1, -1], [+1, 0, 1], [0, 1, -1], [-1, 0, 1]]
            )
        self.reset()

    ################################################################################

    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        super().reset()
        #### Store the last roll, pitch, and yaw ###################
        self.last_rpy = np.zeros(3)
        #### Initialized PID control variables #####################
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)
        #### Desired rotation from outer loop ######################
        self.R_d_des = np.eye(3)

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
        """Computes the PID control action (as RPMs) for a single drone.

        This methods sequentially calls `_dslPIDPositionControl()` and `_dslPIDAttitudeControl()`.
        Parameter `cur_ang_vel` is unused.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        ndarray
            (3,1)-shaped array of floats containing the current XYZ position error.
        float
            The current yaw error.

        """
        self.control_counter += 1
        mass = self._getURDFParameter("m")

        # OUTER LOOP: position → thrust vector T_d, desired R_d
        target_thrust, computed_target_rpy, pos_e, cur_rotation = (
            self._dslPIDPositionControl(
                control_timestep,
                cur_pos,
                cur_quat,
                cur_vel,
                target_pos,
                target_rpy,
                target_vel,
                target_acc,
                mass=mass,
            )
        )

        # Project desired thrust vector onto current body z-axis
        scalar_thrust = max(0.0, np.dot(target_thrust, cur_rotation[:, 2]))

        # Convert scalar thrust → baseline PWM for all 4 motors
        thrust = (
            math.sqrt(scalar_thrust / (4 * self.KF)) - self.PWM2RPM_CONST
        ) / self.PWM2RPM_SCALE

        # INNER LOOP: attitude control on SO(3)
        rpm = self._dslPIDAttitudeControl(
            control_timestep,
            thrust,
            cur_quat,
            cur_ang_vel,
            computed_target_rpy,
            target_rpy_rates,
        )

        return rpm, pos_e, computed_target_rpy

    def _dslPIDPositionControl(
        self,
        control_timestep,
        cur_pos,
        cur_quat,
        cur_vel,
        target_pos,
        target_rpy,
        target_vel,
        target_acc,
        mass=0.29,
    ):
        """DSL's CF2.x PID position control.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray
            (3,1)-shaped array of floats containing the desired velocity.
        target_acc : ndarray
            (3,1)-shaped array of floats containing the desired acceleration.

        Returns
        -------
        float
            The target thrust along the drone z-axis.
        ndarray
            (3,1)-shaped array of floats containing the target roll, pitch, and yaw.
        float
            The current position error.
        ndarray
            (3,3)-shaped array of floats representing the current rotation matrix (from quaternion).
        """

        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)

        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel

        a_des = target_acc + self.Kp_pos * pos_e + self.Kd_pos * vel_e

        # Gravity compensation vector
        e3 = np.array([0.0, 0.0, 1.0])

        T_d = mass * (self.g_lin * e3 + a_des)

        T_mag = np.linalg.norm(T_d)
        if T_mag < 1e-6:
            z_B_d = e3.copy()
        else:
            z_B_d = T_d / T_mag

        psi_d = target_rpy[2]

        x_C_d = np.array([math.cos(psi_d), math.sin(psi_d), 0.0])
        if abs(np.dot(x_C_d, z_B_d)) > 0.99:  # avoid degeneracy
            x_C_d = np.array([1.0, 0.0, 0.0])

        y_B_d = np.cross(z_B_d, x_C_d)
        y_B_d /= np.linalg.norm(y_B_d)
        x_B_d = np.cross(y_B_d, z_B_d)

        R_d = np.column_stack((x_B_d, y_B_d, z_B_d))
        self.R_d_des = R_d  # store for attitude loop

        target_rpy_out = Rotation.from_matrix(R_d).as_euler("xyz", degrees=False)

        return T_d, target_rpy_out, pos_e, cur_rotation

    def _dslPIDAttitudeControl(
        self,
        control_timestep,
        thrust,
        cur_quat,
        cur_ang_vel,
        target_euler,
        target_rpy_rates,
    ):
        """DSL's CF2.x PID attitude control.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        thrust : float
            The target thrust along the drone z-axis.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        target_euler : ndarray
            (3,1)-shaped array of floats containing the computed target Euler angles.
        target_rpy_rates : ndarray
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.

        """

        R = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)

        # Desired rotation from outer loop (already stored)
        R_d = self.R_d_des

        R_e = R_d.T @ R
        skew = 0.5 * (R_e - R_e.T)
        e_R = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])

        omega = cur_ang_vel
        omega_d_desired = target_rpy_rates  # usually [0, 0, 0]
        omega_d = R.T @ (R_d @ omega_d_desired)
        e_omega = omega - omega_d

        target_torques = -self.Kp_att * e_R - self.Kd_att * e_omega

        ################################################################################

        target_torques = np.clip(target_torques, -3200, 3200)
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)

        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST

    ################################################################################

    def _one23DInterface(self, thrust):
        """Utility function interfacing 1, 2, or 3D thrust input use cases.

        Parameters
        ----------
        thrust : ndarray
            Array of floats of length 1, 2, or 4 containing a desired thrust input.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the PWM (not RPMs) to apply to each of the 4 motors.

        """
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
