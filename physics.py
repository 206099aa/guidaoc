import numpy as np
import math
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict

# 配置日志
logger = logging.getLogger("HighFidelityPhysics")

# --- 物理常数 ---
GRAVITY = 9.81  # m/s^2
RHO_MUD = 1450.0  # 泥浆密度 kg/m^3 (比水重)


@dataclass
class EnvironmentState:
    """
    [Environment Layer]
    描述水田轨道的微观环境状态。
    SCI Point: 引入空间异质性 (Spatial Heterogeneity) 和流变特性。
    """
    mud_depth: float = 0.2  # 泥层深度 (m)
    mud_viscosity: float = 50.0  # 泥浆动力粘度 (Pa·s, 非牛顿流体)
    soil_cohesion: float = 5000.0  # 土壤粘聚力 c (Pa)
    soil_friction_angle: float = 25.0  # 内摩擦角 (deg)
    rail_roughness: float = 0.0  # 轨道不平顺度 (PSD input)


class BekkerTerramechanics:
    """
    [Soil Mechanics Module]
    基于 Bekker-Wong 理论的水田软土承载力与阻力计算。
    Reference: Wong, J. Y. "Theory of Ground Vehicles", 4th Ed.
    """

    def __init__(self):
        # 水田软土典型参数
        self.kc = 1000.0  # 粘聚模量
        self.kphi = 10000.0   # 摩擦模量
        self.n = 0.7  # 沉陷指数 (0.5~0.8 for paddy)

    def compute_sinkage_and_resistance(self, normal_load_N, contact_area_m2, wheel_width_m):
        """
        计算沉陷量(z)和压实阻力(Rc)
        :return: (sinkage_z, resistance_force)
        """
        if contact_area_m2 <= 1e-6: return 0.0, 0.0
        pressure = normal_load_N / contact_area_m2

        # 2. 计算等效模量 k_eq = kc/b + kphi
        k_eq = (self.kc / wheel_width_m) + self.kphi

        # 3. 计算沉陷量 z = (p / k_eq)^(1/n)
        try:
            val = pressure / max(k_eq, 1.0)
            z = (val) ** (1.0 / self.n)
        except OverflowError:
            z = 0.5

        z = np.clip(z, 0.0, 0.4)  # 物理限制：最大沉陷不超过轨道高度

        # 4. 计算压实阻力 R_c = b * k_eq * z^(n+1) / (n+1)
        resistance = wheel_width_m * k_eq * (z ** (self.n + 1)) / (self.n + 1)

        return z, resistance


class PolachContactModel:
    """
    [Tribology Module]
    修正版 Polach 轮轨蠕滑模型，考虑泥浆介质 (Third-body Layer)。
    用于计算牵引力/制动力上限。
    """

    def __init__(self):
        # 干轨标准参数
        self.mu_0_dry = 0.45  # 最大静摩擦
        self.ka = 1.0  # 饱和区参数
        self.ks = 0.4  # 滑动区参数

    def compute_adhesion_force(self, normal_force, creepage, mud_factor):
        """
        :param creepage: 蠕滑率 (v_wheel - v_vehicle) / v_vehicle
        :param mud_factor: 泥浆污染指数 [0, 1]
        :return: (force, mu_effective)
        """
        # 1. 泥浆导致的粘着系数衰减 (Adhesion Degradation)
        # SCI EQ: mu = mu0 * exp(-k * mud)
        mu_available = self.mu_0_dry * np.exp(-1.2 * mud_factor)

        if abs(creepage) < 1e-5:
            # [SCI ADD] 返回 mu_available 以便遥测
            return 0.0, mu_available

        # 2. Polach 公式计算 (简化版，用于纵向力)
        try:
            C = 4.0e6 * (normal_force ** 0.66)
            epsilon = (C * abs(creepage)) / (mu_available * max(normal_force, 1.0))

            # 摩擦系数 mu
            mu_polach = mu_available * ((2.0 / math.pi) * math.atan(epsilon))
        except (ValueError, OverflowError):
            mu_polach = mu_available

        # 3. 混合摩擦后的切向力
        tangential_force = mu_polach * normal_force * np.sign(creepage)

        # [SCI ADD] 返回计算出的实时摩擦系数 mu_polach
        return tangential_force, mu_polach


class DCMotorModel:
    """
    [Electrical Module]
    直流串励电机动态模型。
    方程组:
    1. V_in = I*R + L*(dI/dt) + K_e * omega
    2. T_elec = K_t * I
    """

    def __init__(self, specs: Dict):
        self.R = specs.get('R', 0.5)  # 电枢电阻
        self.L = specs.get('L', 0.1)  # 电枢电感
        self.Ke = specs.get('Ke', 0.8)  # 反电动势常数
        self.Kt = specs.get('Kt', 0.8)  # 扭矩常数
        self.J_rotor = specs.get('J', 0.1)  # 转子惯量
        self.current = 0.0  # 状态变量：电流

    def step_electrical(self, dt, voltage_in, omega_motor):
        """
        计算电流变化 (dI/dt)
        """
        back_emf = self.Ke * omega_motor
        di_dt = (voltage_in - self.current * self.R - back_emf) / self.L
        self.current += di_dt * dt

        # 饱和与保护
        self.current = np.clip(self.current, -400.0, 400.0)

        torque = self.Kt * self.current
        return torque, self.current


class RailVehicleMBDSystem:
    """
    [System Integration] Multi-Body Dynamics for Rail Vehicle.
    """

    def __init__(self, vehicle_config, env_config):
        self.cfg = vehicle_config
        self.env = EnvironmentState(mud_depth=env_config.get('mud_factor', 0.5))

        # 车辆参数解析
        self.num_wagons = 2  # 默认2节挂车

        # [SCI ADD] 显式保存这些属性，供 vehicle.py 调用
        self.mass_total = float(vehicle_config.get('mass_full', 5000))
        self.length = float(vehicle_config.get('length', 5.0))

        self.mass_loco = self.mass_total * 0.4
        self.mass_wagon = self.mass_total * 0.3
        self.wheel_radius = 0.3
        self.gear_ratio = 20.0

        # 初始化子模型
        self.soil_model = BekkerTerramechanics()
        self.contact_model = PolachContactModel()
        self.motor = DCMotorModel({'R': 0.5, 'L': 0.02, 'Ke': 1.2, 'Kt': 2.4})

        # 状态向量 Y: [x0, v0, x1, v1, x2, v2...]
        # 长度 = 2 * (1 + num_wagons)
        self.dof = 2 * (1 + self.num_wagons)
        self.state = np.zeros(self.dof)

        # [SCI ADD] 缓存上一帧的有效摩擦系数
        self.last_mu_eff = 0.0

        # 车钩参数 (Non-linear Spring-Damper)
        self.coupler_gap = 0.05  # 5cm 虚接间隙
        self.k_coupler = 2.0e5  # 1 MN/m (软化以防震荡)
        self.c_coupler = 5.0e5  # 50 kNs/m

        self._init_position()

    def _init_position(self, spacing=2.0):
        """初始化列车编组位置"""
        for i in range(1 + self.num_wagons):
            self.state[2 * i] = -i * spacing
            self.state[2 * i + 1] = 0.0

    def _calculate_coupler_force(self, idx_front, idx_rear):
        """计算非线性车钩力 (含间隙模型)"""
        x_front = self.state[2 * idx_front]
        v_front = self.state[2 * idx_front + 1]
        x_rear = self.state[2 * idx_rear]
        v_rear = self.state[2 * idx_rear + 1]

        # 标称间距
        nominal_dist = 2.0
        dx = x_front - x_rear - nominal_dist
        dv = v_front - v_rear

        force = 0.0
        # 死区模型 (Dead-zone): 只有超出间隙才有力
        if abs(dx) > self.coupler_gap:
            # Stiffening logic
            stiffening = 1.0 + min(10.0 * abs(dx), 50.0)
            k_eff = self.k_coupler * stiffening

            # F = K*x + C*v
            spring_force = k_eff * (dx - np.sign(dx) * self.coupler_gap)
            damping_force = self.c_coupler * dv
            force = spring_force + damping_force

            # Clamping Force
            force = np.clip(force, -1e6, 1e6)

        return force

    def get_derivatives(self, t, state, voltage_input):
        """
        计算状态导数 DY/DT = [v0, a0, v1, a1...]
        """
        derivs = np.zeros_like(state)

        # --- 1. 机车动力学 (Locomotive) ---
        idx_loco = 0
        v_loco = state[1]

        v_safe = np.clip(v_loco, -50.0, 50.0)

        # A. 电机扭矩
        omega_wheel = v_safe / self.wheel_radius * self.gear_ratio
        motor_torque, _ = self.motor.step_electrical(0.01, voltage_input, omega_wheel)
        force_wheel_ideal = (motor_torque * self.gear_ratio) / self.wheel_radius

        # B. 粘着限制
        creep_est = 0.02 * np.sign(voltage_input)

        # [SCI ADD] 获取 mu_eff
        f_adhesion_limit, mu_eff = self.contact_model.compute_adhesion_force(
            self.mass_loco * GRAVITY, creep_est, self.env.mud_depth
        )
        self.last_mu_eff = mu_eff  # 记录

        f_tract = np.clip(force_wheel_ideal, -abs(f_adhesion_limit), abs(f_adhesion_limit))

        # C. 阻力计算 (使用 v_safe)
        sign_v = np.sign(v_safe)

        # 压实阻力
        z, r_compaction = self.soil_model.compute_sinkage_and_resistance(
            self.mass_loco * GRAVITY / 4, 0.05, 0.1
        )
        f_soil = r_compaction * 4 * sign_v

        # 泥浆粘性阻力
        f_mud_drag = 0.5 * RHO_MUD * 0.8 * (0.5 * self.env.mud_depth) * (v_safe ** 2) * sign_v

        # 气动阻力
        f_aero = 0.5 * 1.2 * 4.0 * 0.6 * (v_safe ** 2) * sign_v

        f_net_loco = f_tract - f_soil - f_mud_drag - f_aero

        # --- 2. 挂车动力学 ---
        wagon_forces = []
        for i in range(self.num_wagons):
            v_w = state[2 * (i + 1) + 1]
            v_w_safe = np.clip(v_w, -50.0, 50.0)
            sign_w = np.sign(v_w_safe) if abs(v_w_safe) > 0.01 else 0

            z_w, r_c_w = self.soil_model.compute_sinkage_and_resistance(
                self.mass_wagon * GRAVITY / 4, 0.05, 0.1
            )
            f_res_w = (r_c_w * 4 + 0.5 * RHO_MUD * 0.8 * (0.5 * self.env.mud_depth) * v_w_safe ** 2) * sign_w
            wagon_forces.append(-f_res_w)

        # --- 3. 车钩力耦合 ---
        f_c1 = self._calculate_coupler_force(0, 1)
        f_net_loco -= f_c1
        wagon_forces[0] += f_c1

        for i in range(self.num_wagons - 1):
            f_c_next = self._calculate_coupler_force(i + 1, i + 2)
            wagon_forces[i] -= f_c_next
            wagon_forces[i + 1] += f_c_next

        # --- 4. 组装导数 ---
        derivs[0] = v_safe
        derivs[1] = f_net_loco / self.mass_loco

        for i in range(self.num_wagons):
            idx = i + 1
            derivs[2 * idx] = state[2 * idx + 1]  # dx/dt = v
            derivs[2 * idx + 1] = wagon_forces[i] / self.mass_wagon  # dv/dt = a

        return derivs

    def step_rk4(self, dt, voltage_input):
        """
        [Numerical Core] Runge-Kutta 4th Order Solver with Safety Clamp.
        """
        y = self.state.copy()

        try:
            k1 = self.get_derivatives(0, y, voltage_input)
            k2 = self.get_derivatives(dt / 2, y + (dt / 2) * k1, voltage_input)
            k3 = self.get_derivatives(dt / 2, y + (dt / 2) * k2, voltage_input)
            k4 = self.get_derivatives(dt, y + dt * k3, voltage_input)

            new_state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Check for NaN/Inf
            if not np.all(np.isfinite(new_state)):
                logger.warning("Physics NaN detected! Resetting step.")
                self.state[1::2] *= 0.9  # Damping energy
            else:
                self.state = new_state

        except RuntimeWarning:
            logger.warning("Physics Overflow. Clamping state.")
            self.state[1::2] = np.clip(self.state[1::2], -20.0, 20.0)

        # Stiction Logic
        for i in range(1 + self.num_wagons):
            vel_idx = 2 * i + 1
            # [Fix] 移除了强制归零的逻辑，或者将阈值设得极低，保证微小移动能被看到
            if abs(self.state[vel_idx]) < 0.0001 and abs(voltage_input) < 1.0:
                self.state[vel_idx] = 0.0

        return {
            'loco_vel': self.state[1],
            'motor_current': self.motor.current,
            'coupler_force_1': self._calculate_coupler_force(0, 1),
            # [SCI ADD] 增加这些字段，供可视化使用
            'mu_effective': self.last_mu_eff,
            'mass_total': self.mass_total,
            'length': self.length
        }