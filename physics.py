import numpy as np
import math
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict

# 配置日志
logger = logging.getLogger("Edge.Physics")

GRAVITY = 9.81  # m/s^2


class DavisResistanceModel:
    """
    [Rail Physics] 戴维斯阻力公式 (Davis Equation)
    R = A + B*v + C*v^2
    其中 A 项与质量成正比 (滚动阻力)
    """

    def __init__(self):
        self.roll_coeff = 0.0015  # 滚动摩擦系数
        self.flange_coeff = 0.0001  # 轮缘摩擦系数 (机械阻力)
        self.aero_drag_coeff = 0.3  # 空气阻力系数

    def compute_resistance(self, mass_kg, velocity, frontal_area=2.0):
        v_abs = abs(velocity)
        # A项: 滚动阻力 (F = mu * N)
        f_roll = mass_kg * GRAVITY * self.roll_coeff
        # B项: 机械/轮缘阻力
        f_mech = mass_kg * self.flange_coeff * v_abs
        # C项: 空气阻力
        f_aero = 0.5 * 1.225 * frontal_area * self.aero_drag_coeff * (v_abs ** 2)

        total_resistance = f_roll + f_mech + f_aero
        # 阻力方向始终与速度相反
        return total_resistance * np.sign(velocity) if v_abs > 0.001 else 0.0


class PolachContactModel:
    """
    [Tribology] Polach 轮轨接触模型
    模拟非线性粘着特性，包含泥泞因子的影响
    """

    def __init__(self):
        self.mu_0_dry = 0.45  # 干轨基础粘着系数

    def compute_adhesion_force(self, normal_force, v_wheel, v_vehicle, mud_factor):
        epsilon = 1e-5
        v_ref = max(abs(v_vehicle), epsilon)
        # 蠕滑率 (Creepage)
        creepage = (v_wheel - v_vehicle) / v_ref

        # 环境衰减模型
        # 泥泞度 (0.1-0.9) 越高，最大可用粘着越小
        # 0.9 mud -> factor = 0.4 * (1 - 0.9) = 0.04 (极低摩擦)
        env_decay = 1.0 - 0.8 * np.clip(mud_factor, 0, 1.0)
        mu_available = max(0.05, self.mu_0_dry * env_decay)

        if abs(creepage) < 1e-4:
            return 0.0, 0.0

        # Polach 简化公式 (双曲正切拟合)
        k_creep = 40.0  # 刚度系数
        mu_eff = mu_available * np.tanh(abs(k_creep * creepage))

        # 粘着力 = 正压力 * 有效粘着系数
        return normal_force * mu_eff * np.sign(creepage), mu_eff


class DCMotorModel:
    """
    [Electrical] 直流电机模型
    V = I*R + L*di/dt + Ke*omega
    T = Kt*I
    """

    def __init__(self, specs: Dict, max_current_limit: float = 500.0):
        self.R = specs.get('R', 0.05)
        self.L = specs.get('L', 0.05)
        self.Ke = specs.get('Ke', 1.0)  # 反电动势常数
        self.Kt = specs.get('Kt', 1.0)  # 转矩常数
        self.current = 0.0
        self.max_current_limit = max_current_limit

    def step_electrical(self, dt, voltage_in, omega_motor):
        # 反电动势
        back_emf = self.Ke * omega_motor
        # 电流变化率 di/dt = (V - IR - EMF) / L
        di_dt = (voltage_in - self.current * self.R - back_emf) / self.L

        # 更新电流
        self.current += di_dt * dt
        # 物理限流 (饱和)
        self.current = np.clip(self.current, -self.max_current_limit, self.max_current_limit)

        torque = self.Kt * self.current
        return torque, self.current


class RailVehicleMBDSystem:
    """
    [System Integration] 轨道车辆多体动力学系统 (Multibody Dynamics)
    核心逻辑：车头(Locomotive) + N节车厢(Wagons)
    """

    def __init__(self, vehicle_config, env_config):
        self.cfg = vehicle_config
        self.env_config = env_config
        self.mud_factor = env_config.get('mud_factor', 0.5)

        # ==========================================
        # ⚖️ [关键修正：异构质量模型]
        # ==========================================
        # 读取配置中的总重 (例如 Heavy=5000kg, Scout=2000kg)
        self.target_mass_full = float(vehicle_config.get('mass_full', 5000.0))
        self.target_mass_empty = float(vehicle_config.get('mass_empty', 2000.0))

        # 车型判断
        self.is_heavy = self.target_mass_full > 4000
        # 设定车厢数量：重车10节，轻车3节
        self.num_wagons = 10 if self.is_heavy else 3

        # --- 质量分配 (Mass Distribution) ---
        # 1. 车头质量 (Constant): 假设为空载总重的 40%
        #    车头必须足够重才能提供牵引摩擦力 (F = mu * N)
        self.mass_loco = self.target_mass_empty * 0.4

        # 2. 车厢质量 (Variable):
        #    空载时，剩余 60% 重量平分给所有车厢
        self.mass_wagon_empty = (self.target_mass_empty * 0.6) / self.num_wagons

        #    货物质量: (满载 - 空载) 平分给所有车厢
        total_cargo = max(0, self.target_mass_full - self.target_mass_empty)
        self.mass_cargo_per_wagon = total_cargo / self.num_wagons

        # 初始化为【空载】状态
        self.mass_wagon_current = self.mass_wagon_empty
        self.is_loaded = False

        # 计算当前总重
        self.mass_total = self.mass_loco + self.num_wagons * self.mass_wagon_current

        logger.info(f"[Physics Init] {vehicle_config.get('type_name', 'Vehicle')}: "
                    f"Loco={self.mass_loco:.1f}kg, "
                    f"Wagon(Empty)={self.mass_wagon_empty:.1f}kg x {self.num_wagons}, "
                    f"Cargo/Wagon={self.mass_cargo_per_wagon:.1f}kg")

        # ==========================================
        # 动力传动系统 (Powertrain)
        # ==========================================
        self.wheel_radius = 0.35
        self.gear_ratio = 12.0 if self.is_heavy else 8.0
        self.sys_voltage = 400.0  # V

        # 自动调参：根据满载爬坡需求反推电机扭矩常数
        # 目标：在满载下能爬 5度坡 (sin(5)~0.09)
        req_traction = self.target_mass_full * GRAVITY * 0.09
        req_torque = req_traction * self.wheel_radius / self.gear_ratio
        # 假设此时电流 400A
        calc_Kt = req_torque / 400.0

        self.davis_model = DavisResistanceModel()
        self.contact_model = PolachContactModel()
        self.motor = DCMotorModel(
            {'R': 0.05, 'L': 0.05, 'Ke': calc_Kt, 'Kt': calc_Kt},
            max_current_limit=500.0
        )

        # ==========================================
        # 多体状态向量 (State Vector)
        # ==========================================
        # State: [x_loco, v_loco, x_w1, v_w1, ..., x_wN, v_wN]
        # 自由度 = 2 * (1 + N)
        self.dof = 2 * (1 + self.num_wagons)
        self.state = np.zeros(self.dof)

        self.length = float(vehicle_config.get('length', 12.0))
        self.last_mu_eff = 0.0

        # 车钩参数 (Spring-Damper)
        self.coupler_gap = 0.02  # 间隙
        self.k_coupler = 8.0e4  # 刚度
        self.c_coupler = 2.0e4  # 阻尼

        self._init_position()

    def set_load_status(self, loaded: bool):
        """
        [External Interface] 改变装载状态
        这是产生物理差异的关键：满载后车厢变重，死重增加。
        """
        self.is_loaded = loaded
        if loaded:
            self.mass_wagon_current = self.mass_wagon_empty + self.mass_cargo_per_wagon
        else:
            self.mass_wagon_current = self.mass_wagon_empty

        # 更新总重记录
        self.mass_total = self.mass_loco + self.num_wagons * self.mass_wagon_current
        logger.info(f"[Physics] Status Update: {'FULL LOAD' if loaded else 'EMPTY'}. "
                    f"Wagon Mass: {self.mass_wagon_current:.1f}kg")

    def _init_position(self, spacing=3.0):
        """初始化列车队列位置"""
        # 车头在 0，后续车厢依次排在 -3, -6, ...
        for i in range(1 + self.num_wagons):
            self.state[2 * i] = -i * spacing
            self.state[2 * i + 1] = 0.0

    def _calculate_coupler_force(self, idx_front, idx_rear, state_vec):
        """计算两节车之间的车钩力"""
        x_front = state_vec[2 * idx_front]
        v_front = state_vec[2 * idx_front + 1]
        x_rear = state_vec[2 * idx_rear]
        v_rear = state_vec[2 * idx_rear + 1]

        nominal_dist = 3.0  # 标准间隔
        dx = x_front - x_rear - nominal_dist
        dv = v_front - v_rear

        force = 0.0
        # 弹簧阻尼模型 (仅在拉伸/压缩超过间隙时生效)
        if abs(dx) > self.coupler_gap:
            # 刚度项 + 阻尼项
            # dx > 0: 拉伸，力为正(拉力)
            force = self.k_coupler * (dx - np.sign(dx) * self.coupler_gap) + self.c_coupler * dv

        return force

    def get_derivatives(self, t, state, motor_torque, voltage):
        """
        计算状态导数 (RK4 核心)
        输入: 当前状态, 电机扭矩, 电压
        输出: 状态导数 [v_loco, a_loco, v_w1, a_w1...]
        """
        derivs = np.zeros_like(state)

        # --- 1. 车头动力学 (Locomotive) ---
        v_loco = state[1]

        # A. 驱动力 (Drive Force)
        f_drive_ideal = (motor_torque * self.gear_ratio) / self.wheel_radius

        # B. 粘着限制 (Adhesion Limit)
        # 车轮空转速度近似 (假设无滑动时 v_wheel = v_loco)
        # 电压越高，空转倾向越大
        v_wheel_spin = v_loco + (voltage / self.sys_voltage) * 2.0

        # 计算最大可用牵引力 (受限于车头重量 mass_loco)
        f_adh_limit, mu = self.contact_model.compute_adhesion_force(
            self.mass_loco * GRAVITY, v_wheel_spin, v_loco, self.mud_factor
        )
        self.last_mu_eff = mu

        # 实际牵引力 = min(电机力, 粘着力)
        f_traction = np.clip(f_drive_ideal, -abs(f_adh_limit), abs(f_adh_limit))

        # C. 运行阻力 (Resistance) - 仅基于车头重量
        f_res_loco = self.davis_model.compute_resistance(self.mass_loco, v_loco, frontal_area=2.5)

        # D. 驻车/制动辅助 (低速高阻尼)
        if abs(voltage) < 1.0 and abs(v_loco) < 0.1:
            f_traction = 0.0
            f_res_loco += 5000.0 * v_loco

        f_net_loco = f_traction - f_res_loco

        # --- 2. 车厢动力学 (Wagons) ---
        wagon_forces = []

        for i in range(self.num_wagons):
            v_w = state[2 * (i + 1) + 1]
            # 阻力基于【当前车厢质量】(空载/满载不同)
            f_res_w = self.davis_model.compute_resistance(self.mass_wagon_current, v_w, frontal_area=1.8)
            wagon_forces.append(-f_res_w)  # 初始仅受阻力

        # --- 3. 车钩力传递 (Coupler Interaction) ---
        # Loco <-> Wagon_0
        f_c0 = self._calculate_coupler_force(0, 1, state)
        f_net_loco -= f_c0  # 车头被向后拉 (阻力)
        wagon_forces[0] += f_c0  # 第一节车厢被向前拉 (动力)

        # Wagon_i <-> Wagon_i+1
        for i in range(self.num_wagons - 1):
            f_c = self._calculate_coupler_force(i + 1, i + 2, state)
            wagon_forces[i] -= f_c  # 前车被后车拉
            wagon_forces[i + 1] += f_c  # 后车被前车拉

        # --- 4. 组装导数 (a = F/m) ---
        # Loco
        derivs[0] = v_loco
        derivs[1] = f_net_loco / self.mass_loco

        # Wagons
        for i in range(self.num_wagons):
            idx = i + 1
            derivs[2 * idx] = state[2 * idx + 1]
            # 使用当前车厢质量计算加速度
            derivs[2 * idx + 1] = wagon_forces[i] / self.mass_wagon_current

        return derivs

    def step_rk4(self, dt, control_signal):
        """
        四阶龙格-库塔积分器 (Runge-Kutta 4)
        """
        # 输入映射: 控制信号(-48V ~ +48V) -> 电机电压
        voltage = (control_signal / 48.0) * self.sys_voltage
        voltage = np.clip(voltage, -self.sys_voltage, self.sys_voltage)

        y = self.state.copy()

        # 1. 更新电机状态 (Current loop is faster, integrated once per step)
        omega = y[1] / self.wheel_radius * self.gear_ratio
        trq, _ = self.motor.step_electrical(dt, voltage, omega)

        # 2. RK4 Integration for MBD
        k1 = self.get_derivatives(0, y, trq, voltage)
        k2 = self.get_derivatives(0, y + 0.5 * dt * k1, trq, voltage)
        k3 = self.get_derivatives(0, y + 0.5 * dt * k2, trq, voltage)
        k4 = self.get_derivatives(0, y + dt * k3, trq, voltage)

        self.state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # 返回遥测数据
        return {
            'loco_vel': self.state[1],  # 车头速度
            'motor_current': self.motor.current,  # 电机电流
            'coupler_force_1': self._calculate_coupler_force(0, 1, self.state),  # 首节车钩力
            'mu_effective': self.last_mu_eff,  # 粘着系数
            'mass_total': self.mass_total  # 当前总重
        }