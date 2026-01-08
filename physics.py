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
        # [调优] 轻量化板车参数：滚阻和风阻都较小
        self.roll_coeff = 0.0015
        self.flange_coeff = 0.0001
        self.aero_drag_coeff = 0.3

    def compute_resistance(self, mass_kg, velocity, frontal_area=2.0):
        # 迎风面积默认设为 2.0 (板车)
        v_abs = abs(velocity)
        f_roll = mass_kg * GRAVITY * self.roll_coeff
        f_mech = mass_kg * self.flange_coeff * v_abs
        f_aero = 0.5 * 1.225 * frontal_area * self.aero_drag_coeff * (v_abs ** 2)
        total_resistance = f_roll + f_mech + f_aero
        return total_resistance * np.sign(velocity) if v_abs > 0.001 else 0.0


class PolachContactModel:
    """[Tribology] 轮轨接触模型"""

    def __init__(self):
        # [调优] 基础干摩擦系数提升至 0.65 (模拟良好像接触)
        self.mu_0_dry = 0.65

    def compute_adhesion_force(self, normal_force, v_wheel, v_vehicle, mud_factor):
        epsilon = 1e-5
        v_ref = max(abs(v_vehicle), epsilon)
        creepage = (v_wheel - v_vehicle) / v_ref

        # [物理真实性] 泥泞会导致摩擦下降，但我们保证下限不低于 0.3
        # 0.3 的摩擦系数配合 900kg 的驱动轴重，足够产生强劲推力
        mu_available = self.mu_0_dry * (1.0 - 0.2 * np.clip(mud_factor, 0, 1))
        mu_available = max(mu_available, 0.3)

        if abs(creepage) < 1e-4:
            return 0.0, 0.0

        # 蠕滑特性曲线 (较陡峭，响应快)
        k_creep = 30.0
        tau = k_creep * creepage
        # 使用 tanh 模拟饱和特性
        mu_eff = mu_available * np.tanh(abs(tau))

        return normal_force * mu_eff * np.sign(creepage), mu_eff


class DCMotorModel:
    """[Electrical] 电机模型"""

    def __init__(self, specs: Dict, max_current_limit: float = 400.0):
        self.R = specs.get('R', 0.02)  # 低内阻
        self.L = specs.get('L', 0.05)
        self.Ke = specs.get('Ke', 1.2)
        self.Kt = specs.get('Kt', 1.2)
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

        # ==========================================
        # 🛠️ [修复] 恢复丢失的属性定义
        # ==========================================
        self.length = float(vehicle_config.get('length', 8.0))

        # ==========================================
        # ⚖️ [轻量化改装 - LIGHTWEIGHT MOD]
        # ==========================================
        # 强制覆盖 Config 中的 5000kg 设置
        self.mass_total = 1200.0  # 总重 1.2吨

        # 质量分布优化：让“电机铁壳”占据大部分重量以获取抓地力
        self.mass_loco = 900.0  # 驱动车头 900kg (75%)
        self.mass_wagon = 300.0  # 平板拖车 300kg (25%)

        self.wheel_radius = 0.35  # 小轮径，起步更快

        # ==========================================
        # ⚡ [动力总成匹配]
        # ==========================================
        # 1. 系统电压: 400V
        self.sys_voltage = 400.0

        # 2. 齿轮比:
        self.gear_ratio = 6.0

        # 3. 电机 Kv 匹配
        calc_Ke = 1.2

        # 4. 电流限制
        max_current = 500.0

        logger.info(f"[Physics] Lightweight Mod (1.2T). 400V System. Ke={calc_Ke}, Gear={self.gear_ratio}")

        self.davis_model = DavisResistanceModel()
        self.contact_model = PolachContactModel()

        self.motor = DCMotorModel(
            {'R': 0.02, 'L': 0.05, 'Ke': calc_Ke, 'Kt': calc_Ke},
            max_current_limit=max_current
        )

        # 恢复多车厢支持逻辑 (虽然这里只配了1节拖车，但代码逻辑保留通用性)
        self.num_wagons = 1
        self.dof = 2 * (1 + self.num_wagons)
        self.state = np.zeros(self.dof)
        self.last_mu_eff = 0.0

        # 软车钩参数
        self.coupler_gap = 0.02
        self.k_coupler_base = 2.0e4
        self.c_coupler = 5.0e3

        self._init_position()

    def _init_position(self, spacing=3.0):
        for i in range(1 + self.num_wagons):
            self.state[2 * i] = -i * spacing
            self.state[2 * i + 1] = 0.0

    def _calculate_coupler_force(self, idx_front, idx_rear):
        x_front = self.state[2 * idx_front]
        v_front = self.state[2 * idx_front + 1]
        x_rear = self.state[2 * idx_rear]
        v_rear = self.state[2 * idx_rear + 1]

        nominal_dist = 3.0
        dx = x_front - x_rear - nominal_dist
        dv = v_front - v_rear

        force = 0.0
        if abs(dx) > self.coupler_gap:
            deformation = abs(dx) - self.coupler_gap
            k_nonlinear = self.k_coupler_base * (1.0 + 2.0 * deformation)
            spring_force = k_nonlinear * (dx - np.sign(dx) * self.coupler_gap)
            damping_force = self.c_coupler * dv
            force = spring_force + damping_force
            force = np.clip(force, -5e4, 5e4)  # 限制最大力防止数值爆炸

        return force

    def get_derivatives(self, t, state, motor_torque_input, voltage_input):
        derivs = np.zeros_like(state)
        v_loco = state[1]

        # 1. 机械驱动力
        f_drive_mech = (motor_torque_input * self.gear_ratio) / self.wheel_radius

        # 2. 粘着控制 (含电压对转速的微扰)
        v_wheel_est = v_loco + 1.0 * (voltage_input / 400.0)
        f_limit, mu_curr = self.contact_model.compute_adhesion_force(
            self.mass_loco * GRAVITY, v_wheel_est, v_loco, self.mud_factor
        )
        self.last_mu_eff = mu_curr
        f_traction = np.clip(f_drive_mech, -abs(f_limit), abs(f_limit))

        # 3. 车头阻力
        f_res_loco = self.davis_model.compute_resistance(self.mass_loco, v_loco, frontal_area=2.0)

        # 停车阻尼
        if abs(voltage_input) < 0.5 and abs(v_loco) < 0.1:
            f_traction = 0.0
            f_res_loco += 2000.0 * v_loco

        f_net_loco = f_traction - f_res_loco

        # 4. 拖车动力学 (通用循环)
        wagon_forces = []
        for i in range(self.num_wagons):
            v_w = state[2 * (i + 1) + 1]
            # 拖车风阻较小，面积设为1.0
            f_res_w = self.davis_model.compute_resistance(self.mass_wagon, v_w, frontal_area=1.0)
            wagon_forces.append(-f_res_w)

        # 5. 车钩力传递 (通用循环)
        # 车头 <-> 第一节
        f_c1 = self._calculate_coupler_force(0, 1)
        f_net_loco -= f_c1
        wagon_forces[0] += f_c1

        # 后续车厢间
        for i in range(self.num_wagons - 1):
            f_c = self._calculate_coupler_force(i + 1, i + 2)
            wagon_forces[i] -= f_c
            wagon_forces[i + 1] += f_c

        # 6. 填充导数
        derivs[0] = v_loco
        derivs[1] = f_net_loco / self.mass_loco
        for i in range(self.num_wagons):
            idx = i + 1
            derivs[2 * idx] = state[2 * idx + 1]
            derivs[2 * idx + 1] = wagon_forces[i] / self.mass_wagon

        return derivs

    def step_rk4(self, dt, control_signal_48v):
        """
        物理步进 (RK4 积分器)
        """
        N_SUB = 10
        dt_sub = dt / N_SUB
        y = self.state.copy()

        # [电压映射] 48V -> 400V
        scaling_factor = self.sys_voltage / 48.0
        bus_voltage = control_signal_48v * scaling_factor
        bus_voltage = np.clip(bus_voltage, -self.sys_voltage, self.sys_voltage)

        for _ in range(N_SUB):
            try:
                omega_wheel = y[1] / self.wheel_radius * self.gear_ratio
                motor_torque = self.motor.Kt * self.motor.current

                k1 = self.get_derivatives(0, y, motor_torque, bus_voltage)
                k2 = self.get_derivatives(0, y + 0.5 * dt_sub * k1, motor_torque, bus_voltage)
                k3 = self.get_derivatives(0, y + 0.5 * dt_sub * k2, motor_torque, bus_voltage)
                k4 = self.get_derivatives(0, y + dt_sub * k3, motor_torque, bus_voltage)
                y_next = y + (dt_sub / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

                self.motor.step_electrical(dt_sub, bus_voltage, omega_wheel)

                if np.all(np.isfinite(y_next)):
                    y = y_next
                else:
                    # 数值不稳定时重置速度，防止崩溃
                    logger.warning("Numerical Instability detected, resetting velocity.")
                    y[1::2] = 0.0
            except RuntimeWarning:
                y[1::2] = 0.0

        self.state = y
        return self._pack_telemetry()

    def _pack_telemetry(self):
        # 确保这里用到的属性都在 __init__ 中定义了
        return {
            'loco_vel': self.state[1],
            'motor_current': self.motor.current,
            'coupler_force_1': self._calculate_coupler_force(0, 1),
            'mu_effective': self.last_mu_eff,
            'mass_total': self.mass_total,
            'length': self.length
        }