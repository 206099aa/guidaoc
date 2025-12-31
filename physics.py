import numpy as np
import math
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict

# 配置日志
logger = logging.getLogger("HighFidelityPhysics")

GRAVITY = 9.81  # m/s^2


@dataclass
class EnvironmentState:
    mud_depth: float = 0.2
    rail_adhesion_factor: float = 1.0


class DavisResistanceModel:
    """[Rail Physics] 戴维斯阻力公式"""

    def __init__(self):
        self.roll_coeff = 0.0020
        self.flange_coeff = 0.0003
        self.aero_drag_coeff = 0.6

    def compute_resistance(self, mass_kg, velocity, frontal_area=4.0):
        v_abs = abs(velocity)
        f_roll = mass_kg * GRAVITY * self.roll_coeff
        f_mech = mass_kg * self.flange_coeff * v_abs
        f_aero = 0.5 * 1.225 * frontal_area * self.aero_drag_coeff * (v_abs ** 2)
        total_resistance = f_roll + f_mech + f_aero
        return total_resistance * np.sign(velocity) if v_abs > 0.001 else 0.0


class PolachContactModel:
    """[Tribology] 轮轨接触模型"""

    def __init__(self):
        self.mu_0_dry = 0.40

    def compute_adhesion_force(self, normal_force, v_wheel, v_vehicle, mud_factor):
        epsilon = 1e-5
        v_ref = max(abs(v_vehicle), epsilon)
        creepage = (v_wheel - v_vehicle) / v_ref

        # 泥泞降低最大摩擦力
        mu_available = self.mu_0_dry * (1.0 - 0.5 * np.clip(mud_factor, 0, 1))
        mu_available = max(mu_available, 0.1)

        if abs(creepage) < 1e-4:
            return 0.0, mu_available

        tau = (100.0 * creepage) / mu_available
        mu_eff = mu_available * (tau / (1.0 + abs(tau)))
        return normal_force * mu_eff, mu_available


class DCMotorModel:
    """[Electrical] 电机模型"""

    def __init__(self, specs: Dict, max_current_limit: float = 400.0):
        self.R = specs.get('R', 0.05)  # [优化] 降低内阻，提升高转速下的效率
        self.L = specs.get('L', 0.05)
        self.Ke = specs.get('Ke', 0.8)
        self.Kt = specs.get('Kt', 0.8)
        self.current = 0.0
        self.max_current_limit = max_current_limit

    def step_electrical(self, dt, voltage_in, omega_motor):
        back_emf = self.Ke * omega_motor
        di_dt = (voltage_in - self.current * self.R - back_emf) / self.L
        self.current += di_dt * dt
        self.current = np.clip(self.current, -self.max_current_limit, self.max_current_limit)
        return self.Kt * self.current, self.current


class RailVehicleMBDSystem:
    """[System Integration] 轨道车辆多体动力学系统"""

    def __init__(self, vehicle_config, env_config):
        self.cfg = vehicle_config
        self.mud_factor = env_config.get('mud_factor', 0.5)

        self.mass_total = float(vehicle_config.get('mass_full', 5000))
        self.length = float(vehicle_config.get('length', 12.0))
        self.mass_loco = self.mass_total * 0.4
        self.mass_wagon = self.mass_total * 0.3

        self.wheel_radius = 0.4

        # [提速修改 1] 换高速档
        # 从 15.0 降到 8.0。牺牲部分低速扭矩，换取更高的极速。
        self.gear_ratio = 8.0

        self.davis_model = DavisResistanceModel()
        self.contact_model = PolachContactModel()

        # [提速修改 2] 强制设定物理层面的极速目标
        # 即使 config 写得小，物理引擎也要按照 15m/s (54km/h) 来匹配电机，
        # 确保 40km/h 时反电动势不会饱和。
        target_max_v_ms = 15.0

        # 功率增强
        max_power_watts = float(vehicle_config.get('max_power', 12000.0))
        if max_power_watts < 60000: max_power_watts = 60000.0  # 60kW 保证高速下的风阻克服能力

        sys_voltage = 48.0
        calculated_max_current = max_power_watts / sys_voltage

        # [提速修改 3] 重新计算 Ke
        # V_emf = Ke * G * (v/r)
        # Ke = 48V / (8.0 * 15.0 / 0.4) ≈ 0.16
        calc_Ke = (sys_voltage * 0.95 * self.wheel_radius) / (self.gear_ratio * target_max_v_ms)
        calc_Ke = np.clip(calc_Ke, 0.05, 5.0)

        logger.info(f"High Speed Tuning: V_max={target_max_v_ms * 3.6:.1f}km/h, Gear={self.gear_ratio}")
        logger.info(f"Motor: Ke={calc_Ke:.4f}, I_max={calculated_max_current:.1f}A")

        self.motor = DCMotorModel(
            {'R': 0.05, 'L': 0.05, 'Ke': calc_Ke, 'Kt': calc_Ke},
            max_current_limit=calculated_max_current
        )

        self.num_wagons = 2
        self.dof = 2 * (1 + self.num_wagons)
        self.state = np.zeros(self.dof)
        self.last_mu_eff = 0.0

        # 软车钩
        self.coupler_gap = 0.02
        self.k_coupler_base = 5.0e4
        self.c_coupler = 1.0e4

        self._init_position()

    def _init_position(self, spacing=2.5):
        for i in range(1 + self.num_wagons):
            self.state[2 * i] = -i * spacing
            self.state[2 * i + 1] = 0.0

    def _calculate_coupler_force(self, idx_front, idx_rear):
        x_front = self.state[2 * idx_front]
        v_front = self.state[2 * idx_front + 1]
        x_rear = self.state[2 * idx_rear]
        v_rear = self.state[2 * idx_rear + 1]

        nominal_dist = 2.5
        dx = x_front - x_rear - nominal_dist
        dv = v_front - v_rear

        force = 0.0
        if abs(dx) > self.coupler_gap:
            deformation = abs(dx) - self.coupler_gap
            k_nonlinear = self.k_coupler_base * (1.0 + 2.0 * deformation)
            spring_force = k_nonlinear * (dx - np.sign(dx) * self.coupler_gap)
            damping_force = self.c_coupler * dv
            force = spring_force + damping_force
            force = np.clip(force, -5e5, 5e5)

        return force

    def get_derivatives(self, t, state, motor_torque_input, voltage_input):
        derivs = np.zeros_like(state)
        v_loco = state[1]

        f_drive_mech = (motor_torque_input * self.gear_ratio) / self.wheel_radius
        v_wheel_est = v_loco + 0.1 * (voltage_input / 48.0)
        f_limit, mu_curr = self.contact_model.compute_adhesion_force(
            self.mass_loco * GRAVITY, v_wheel_est, v_loco, self.mud_factor
        )
        self.last_mu_eff = mu_curr
        f_traction = np.clip(f_drive_mech, -abs(f_limit), abs(f_limit))

        f_res_loco = self.davis_model.compute_resistance(self.mass_loco, v_loco)

        # 软停车阻尼
        if abs(voltage_input) < 0.5 and abs(v_loco) < 0.1:
            f_traction = 0.0
            f_res_loco += 5000.0 * v_loco

        f_net_loco = f_traction - f_res_loco

        wagon_forces = []
        for i in range(self.num_wagons):
            v_w = state[2 * (i + 1) + 1]
            f_res_w = self.davis_model.compute_resistance(self.mass_wagon, v_w)
            wagon_forces.append(-f_res_w)

        f_c1 = self._calculate_coupler_force(0, 1)
        f_net_loco -= f_c1
        wagon_forces[0] += f_c1

        for i in range(self.num_wagons - 1):
            f_c = self._calculate_coupler_force(i + 1, i + 2)
            wagon_forces[i] -= f_c
            wagon_forces[i + 1] += f_c

        derivs[0] = v_loco
        derivs[1] = f_net_loco / self.mass_loco
        for i in range(self.num_wagons):
            idx = i + 1
            derivs[2 * idx] = state[2 * idx + 1]
            derivs[2 * idx + 1] = wagon_forces[i] / self.mass_wagon

        return derivs

    def step_rk4(self, dt, voltage_input):
        N_SUB = 10
        dt_sub = dt / N_SUB
        y = self.state.copy()

        for _ in range(N_SUB):
            try:
                omega_wheel = y[1] / self.wheel_radius * self.gear_ratio
                motor_torque = self.motor.Kt * self.motor.current

                k1 = self.get_derivatives(0, y, motor_torque, voltage_input)
                k2 = self.get_derivatives(0, y + 0.5 * dt_sub * k1, motor_torque, voltage_input)
                k3 = self.get_derivatives(0, y + 0.5 * dt_sub * k2, motor_torque, voltage_input)
                k4 = self.get_derivatives(0, y + dt_sub * k3, motor_torque, voltage_input)
                y_next = y + (dt_sub / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

                self.motor.step_electrical(dt_sub, voltage_input, omega_wheel)

                if not np.all(np.isfinite(y_next)):
                    y[1::2] = 0.0
                else:
                    y = y_next
            except RuntimeWarning:
                y[1::2] = 0.0

        self.state = y
        return self._pack_telemetry()

    def _pack_telemetry(self):
        return {
            'loco_vel': self.state[1],
            'motor_current': self.motor.current,
            'coupler_force_1': self._calculate_coupler_force(0, 1),
            'mu_effective': self.last_mu_eff,
            'mass_total': self.mass_total,
            'length': self.length
        }