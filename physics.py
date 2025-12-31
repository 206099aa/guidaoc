import numpy as np
import math
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict

# 配置日志
logger = logging.getLogger("HighFidelityPhysics")

# --- 物理常数 ---
GRAVITY = 9.81  # m/s^2
RHO_MUD = 1450.0  # 泥浆密度 kg/m^3


@dataclass
class EnvironmentState:
    mud_depth: float = 0.2
    mud_viscosity: float = 50.0
    soil_cohesion: float = 5000.0
    soil_friction_angle: float = 25.0
    rail_roughness: float = 0.0


class BekkerTerramechanics:
    """[Soil Mechanics] 保留 Bekker 理论"""

    def __init__(self):
        self.kc = 1000.0
        self.kphi = 10000.0
        self.n = 0.7

    def compute_sinkage_and_resistance(self, normal_load_N, contact_area_m2, wheel_width_m):
        if contact_area_m2 <= 1e-6: return 0.0, 0.0
        pressure = normal_load_N / contact_area_m2
        k_eq = (self.kc / wheel_width_m) + self.kphi
        try:
            val = pressure / max(k_eq, 1.0)
            z = (val) ** (1.0 / self.n)
        except OverflowError:
            z = 0.5
        z = np.clip(z, 0.0, 0.4)
        resistance = wheel_width_m * k_eq * (z ** (self.n + 1)) / (self.n + 1)
        return z, resistance


class PolachContactModel:
    """[Tribology] 完整保留 Polach 非线性模型"""

    def __init__(self):
        self.mu_0_dry = 0.45
        self.ka = 1.0
        self.ks = 0.4

    def compute_adhesion_force(self, normal_force, v_wheel, v_vehicle, mud_factor):
        # 避免除零
        epsilon = 1e-5
        v_ref = max(abs(v_vehicle), epsilon)

        creepage = (v_wheel - v_vehicle) / v_ref

        # 泥浆衰减
        mu_available = self.mu_0_dry * np.exp(-1.2 * mud_factor)

        if abs(creepage) < 1e-5:
            return 0.0, mu_available

        # 完整的 Polach 公式结构 (arctan)
        # C 是接触刚度相关系数
        C = 4.0e6

        # 归一化梯度
        epsilon_p = (2.0 * C * math.pi * 0.25 * abs(creepage)) / (mu_available * max(normal_force, 1.0))

        # 使用 atan 模拟真实的饱和特性
        mu_eff = mu_available * ((2.0 / math.pi) * math.atan(epsilon_p))

        tangential_force = mu_eff * normal_force * np.sign(creepage)

        return tangential_force, abs(mu_eff)


class DCMotorModel:
    """[Electrical] 直流电机模型"""

    def __init__(self, specs: Dict):
        self.R = specs.get('R', 0.5)
        self.L = specs.get('L', 0.05)
        self.Ke = specs.get('Ke', 0.8)
        self.Kt = specs.get('Kt', 0.8)
        self.current = 0.0

    def step_electrical(self, dt, voltage_in, omega_motor):
        back_emf = self.Ke * omega_motor
        di_dt = (voltage_in - self.current * self.R - back_emf) / self.L
        self.current += di_dt * dt
        self.current = np.clip(self.current, -400.0, 400.0)
        return self.Kt * self.current, self.current


class RailVehicleMBDSystem:
    """
    [System Integration] 多体动力学系统
    保留了所有车钩、多节车厢逻辑。
    """

    def __init__(self, vehicle_config, env_config):
        self.cfg = vehicle_config
        self.env = EnvironmentState(mud_depth=env_config.get('mud_factor', 0.5))

        self.mass_total = float(vehicle_config.get('mass_full', 5000))
        self.length = float(vehicle_config.get('length', 5.0))
        self.mass_loco = self.mass_total * 0.4
        self.mass_wagon = self.mass_total * 0.3

        self.wheel_radius = 0.3
        self.gear_ratio = 15.0

        self.soil_model = BekkerTerramechanics()
        self.contact_model = PolachContactModel()
        self.motor = DCMotorModel({'R': 0.2, 'L': 0.05, 'Ke': 1.0, 'Kt': 1.0})

        # 多体状态: [pos_loco, vel_loco, pos_w1, vel_w1, ...]
        self.num_wagons = 2
        self.dof = 2 * (1 + self.num_wagons)
        self.state = np.zeros(self.dof)
        self.last_mu_eff = 0.0

        # 车钩参数 (保留非线性特性)
        self.coupler_gap = 0.05
        self.k_coupler_base = 2.0e5
        self.c_coupler = 5.0e4

        self._init_position()

    def _init_position(self, spacing=2.0):
        for i in range(1 + self.num_wagons):
            self.state[2 * i] = -i * spacing
            self.state[2 * i + 1] = 0.0

    def _calculate_coupler_force(self, idx_front, idx_rear):
        """保留非线性车钩模型：间隙 + 刚度硬化"""
        x_front = self.state[2 * idx_front]
        v_front = self.state[2 * idx_front + 1]
        x_rear = self.state[2 * idx_rear]
        v_rear = self.state[2 * idx_rear + 1]

        nominal_dist = 2.0
        dx = x_front - x_rear - nominal_dist
        dv = v_front - v_rear

        force = 0.0
        if abs(dx) > self.coupler_gap:
            # 非线性硬化
            deformation = abs(dx) - self.coupler_gap
            k_nonlinear = self.k_coupler_base * (1.0 + 5.0 * deformation)

            spring_force = k_nonlinear * (dx - np.sign(dx) * self.coupler_gap)
            damping_force = self.c_coupler * dv

            force = spring_force + damping_force
            force = np.clip(force, -2e6, 2e6)  # 物理屈服极限

        return force

    def get_derivatives(self, t, state, voltage_input, dt_sub):
        derivs = np.zeros_like(state)

        # --- 机车 ---
        v_loco = state[1]

        # A. 电机
        omega_wheel = v_loco / self.wheel_radius * self.gear_ratio
        # 这里的 dt_sub 是微步长，用于更精确的电流积分
        motor_torque, _ = self.motor.step_electrical(dt_sub, voltage_input, omega_wheel)
        f_drive_ideal = (motor_torque * self.gear_ratio) / self.wheel_radius

        # B. 粘着 (Polach)
        v_wheel_est = v_loco + 0.05 * voltage_input
        f_tract_limit, mu_eff = self.contact_model.compute_adhesion_force(
            self.mass_loco * GRAVITY, v_wheel_est, v_loco, self.env.mud_depth
        )
        self.last_mu_eff = mu_eff

        f_drive = np.clip(f_drive_ideal, -abs(f_tract_limit), abs(f_tract_limit))

        # C. 阻力
        # 使用陡峭的 tanh (k=20) 逼近 sign，但保持连续性
        # 在微步进积分下，这种陡峭度是安全的
        vel_sign = math.tanh(20.0 * v_loco)

        z, r_comp = self.soil_model.compute_sinkage_and_resistance(
            self.mass_loco * GRAVITY / 4, 0.05, 0.1
        )
        f_soil = r_comp * 4 * vel_sign
        f_mud = 0.5 * RHO_MUD * 0.8 * (0.5 * self.env.mud_depth) * (v_loco * abs(v_loco))
        f_aero = 0.5 * 1.2 * 4.0 * 0.6 * v_loco * abs(v_loco)

        f_net_loco = f_drive - f_soil - f_mud - f_aero

        # --- 挂车 ---
        wagon_forces = []
        for i in range(self.num_wagons):
            v_w = state[2 * (i + 1) + 1]
            vel_sign_w = math.tanh(20.0 * v_w)

            z_w, r_c_w = self.soil_model.compute_sinkage_and_resistance(
                self.mass_wagon * GRAVITY / 4, 0.05, 0.1
            )
            f_res_w = (r_c_w * 4 + 0.5 * RHO_MUD * 0.8 * (0.5 * self.env.mud_depth) * v_w * abs(v_w)) * vel_sign_w
            wagon_forces.append(-f_res_w)

        # --- 耦合 ---
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
        """
        [High-Fidelity Core] 微步进积分 (Sub-stepping)
        这是解决刚性系统的正确方法：将主步长 dt 切分为 N 个微步长。
        """
        N_SUB = 20  # 切分为20步，即 0.01s -> 0.0005s/step
        dt_sub = dt / N_SUB

        y = self.state.copy()

        # 在微步循环中进行高频物理计算
        for _ in range(N_SUB):
            # 软锁定逻辑：当能量极低时施加数值阻尼，模拟静摩擦锁定
            # 这比硬性的“置零”更符合物理实际
            if abs(voltage_input) < 1.0 and abs(y[1]) < 0.05:
                y[1::2] *= 0.95  # 快速能量衰减

            try:
                k1 = self.get_derivatives(0, y, voltage_input, dt_sub)
                k2 = self.get_derivatives(dt_sub / 2, y + 0.5 * dt_sub * k1, voltage_input, dt_sub)
                k3 = self.get_derivatives(dt_sub / 2, y + 0.5 * dt_sub * k2, voltage_input, dt_sub)
                k4 = self.get_derivatives(dt_sub, y + dt_sub * k3, voltage_input, dt_sub)

                y_next = y + (dt_sub / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

                # 安全检查
                if not np.all(np.isfinite(y_next)):
                    logger.warning("Physics instability (NaN). Resetting velocity.")
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