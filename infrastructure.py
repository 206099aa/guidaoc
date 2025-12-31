import numpy as np
import logging
import math
from collections import deque
from enum import Enum, auto
from dataclasses import dataclass

# 配置日志
logger = logging.getLogger("DeepEdgeInfra")


class SwitchState(Enum):
    LOCKED_NORMAL = auto()  # 定位锁闭
    LOCKED_REVERSE = auto()  # 反位锁闭
    UNLOCKING = auto()  # 机械解锁过程
    MOVING = auto()  # 转换过程
    LOCKING = auto()  # 机械闭锁过程
    STALLED = auto()  # 堵转故障
    MAINTENANCE = auto()  # 停机维护


@dataclass
class ZD6PhysicalParams:
    """
    [Mechanism Model] ZD6-D Type Switch Machine Parameters.
    Source: Railway Signal Engineering Handbook.
    """
    # 电气参数
    R_armature_20C: float = 4.5  # 20℃时的电枢电阻 (Ohm)
    L_armature: float = 0.05  # 电枢电感 (H)
    Ke: float = 0.85  # 反电动势常数 (V/(rad/s))
    Kt: float = 0.85  # 扭矩常数 (Nm/A)
    thermal_capacity: float = 500.0  # 热容 (J/K)
    heat_dissipation: float = 2.5  # 散热系数 (W/K)

    # 机械参数
    J_rotor: float = 0.02  # 转子惯量 (kg m^2)
    gear_ratio: float = 45.0  # 减速比
    screw_pitch: float = 0.01  # 丝杠螺距 (m)
    stroke: float = 0.16  # 动程 (m)

    # 负载特性
    locking_force_peak: float = 2000.0  # 解锁/闭锁所需的额外机械阻力 (N)


class ElectroThermalMotor:
    """
    [Sub-Model] Coupled Electrical & Thermal Dynamics.
    T(t) -> R(T) -> I(t) -> Torque(t) -> Heat(t) -> T(t)
    SCI Point: Temperature feedback loop affecting reliability.
    """

    def __init__(self, params: ZD6PhysicalParams):
        self.p = params
        self.current = 0.0
        self.temperature = 25.0  # Ambient start
        self.resistance = params.R_armature_20C

    def update_resistance(self):
        # Copper temperature coefficient approx 0.00393
        self.resistance = self.p.R_armature_20C * (1.0 + 0.004 * (self.temperature - 20.0))

    def step(self, dt, voltage_in, omega_rotor):
        # 1. Update Thermal State
        # Q_gen = I^2 * R
        heat_gen = (self.current ** 2) * self.resistance
        # Q_loss = h * (T - T_amb)
        heat_loss = self.p.heat_dissipation * (self.temperature - 25.0)
        delta_T = (heat_gen - heat_loss) / self.p.thermal_capacity * dt
        self.temperature += delta_T
        self.update_resistance()

        # 2. Electrical Dynamics (RK4 integration for current)
        # dI/dt = (V - I*R - Ke*w) / L
        def di_dt(i, v, w, r):
            return (v - i * r - self.p.Ke * w) / self.p.L_armature

        k1 = di_dt(self.current, voltage_in, omega_rotor, self.resistance)
        k2 = di_dt(self.current + 0.5 * dt * k1, voltage_in, omega_rotor, self.resistance)
        k3 = di_dt(self.current + 0.5 * dt * k2, voltage_in, omega_rotor, self.resistance)
        k4 = di_dt(self.current + dt * k3, voltage_in, omega_rotor, self.resistance)

        self.current += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Saturation (Supply limit)
        self.current = np.clip(self.current, -30.0, 30.0)

        return self.p.Kt * self.current  # Torque


class ZD6Mechanism:
    """
    [Sub-Model] Mechanical Transmission with Non-linear Load.
    Includes Stribeck friction (Mud) and Mechanical Locking Curve.
    """

    def __init__(self, params: ZD6PhysicalParams, mud_factor: float):
        self.p = params
        self.mud = mud_factor
        self.omega = 0.0  # Rotor angular vel
        self.pos = 0.0  # Linear position

        # 转换效率 (泥浆会降低机械效率)
        self.efficiency = 0.8 * (1.0 - 0.3 * mud_factor)

    def get_mechanical_load(self, v_linear):
        """
        计算总负载力 (Load Force)
        F_total = F_friction(Mud) + F_locking(Position) + F_external
        """
        # 1. 泥浆介质 Stribeck 摩擦
        # [Fix] 使用 tanh 平滑，防止除零和数值震荡
        F_c = 300.0 + 500.0 * self.mud
        sigma = 1000.0 * (1.0 + 2.0 * self.mud)

        # 简化后的稳定摩擦模型
        f_fric = F_c * math.tanh(10.0 * v_linear) + sigma * v_linear

        # 2. 机械锁闭阻力曲线 (Locking Curve)
        f_lock = 0.0
        if self.pos < 0.01 or self.pos > (self.p.stroke - 0.01):
            f_lock = self.p.locking_force_peak
            if abs(v_linear) > 1e-4:
                f_lock *= np.sign(v_linear)
            else:
                f_lock = 0.0

        return f_fric + f_lock

    def step(self, dt, motor_torque, external_force_N=0.0):
        # 传动关系
        k_linear = self.p.screw_pitch / (2 * np.pi * self.p.gear_ratio)

        v_linear = self.omega * k_linear

        # [CRITICAL FIX] 物理限位逻辑 (Hard Stops)
        # 防止道岔到达终点后位置继续增加导致的数值溢出
        if (self.pos <= 0 and motor_torque < 0) or \
                (self.pos >= self.p.stroke and motor_torque > 0):
            self.omega = 0.0
            v_linear = 0.0
            return self.pos, self.omega

        # 折算到电机轴的负载力矩
        force_total = self.get_mechanical_load(v_linear) + external_force_N
        torque_load = force_total * k_linear / self.efficiency

        # 动力学方程: J * dw/dt = T_motor - T_load - B*w
        dw_dt = (motor_torque - torque_load - 0.05 * self.omega) / self.p.J_rotor

        self.omega += dw_dt * dt
        self.pos += (self.omega * k_linear) * dt

        # Pos Clamping
        self.pos = np.clip(self.pos, 0.0, self.p.stroke)

        return self.pos, self.omega


class EdgeSwitchAgent:
    """
    [Edge Computing Core] Decentralized Infrastructure Agent.

    Capabilities:
    1. V2I Distributed Locking (Wound-Wait Protocol).
    2. Adaptive Control (Voltage Boosting for Muddy conditions).
    3. PHM (Current Signature Analysis for Stall Detection).
    """

    def __init__(self, node_id, env_config):
        self.id = node_id
        self.mud = env_config.get('mud_factor', 0.5)

        self.params = ZD6PhysicalParams()
        self.motor = ElectroThermalMotor(self.params)
        self.mechanism = ZD6Mechanism(self.params, self.mud)

        # Logic State
        self.state = SwitchState.LOCKED_NORMAL
        self.target_pos = 0.0

        # Edge Resource State (Distributed Lock)
        self.owner_id = None
        self.lock_expiry = 0.0
        self.timestamp_table = {}  # {vid: timestamp} to implement Wound-Wait

        # PHM Data
        self.health_index = 1.0
        self.current_buffer = deque(maxlen=100)  # 1s data at 100Hz

        # Internal counters
        self.stall_counter = 0.0

    # --- V2I Negotiation Interface ---

    def handle_access_request(self, packet: dict) -> dict:
        """
        [Protocol] 处理车辆的路权请求
        packet: {vid, priority, timestamp, action='LOCK'/'UNLOCK', duration}
        """
        vid = packet['vid']
        ts = packet['timestamp']
        action = packet.get('action', 'LOCK')

        # 1. 维护分布式时间戳表
        self.timestamp_table[vid] = ts

        current_time = packet.get('global_time', 0.0)  # 仿真时间

        if action == 'UNLOCK':
            if self.owner_id == vid:
                self.owner_id = None
                return {'status': 'RELEASED'}
            return {'status': 'IGNORED'}

        # 2. 死锁预防逻辑 (Wound-Wait)
        # 如果锁空闲，或者是自己，直接给
        if self.owner_id is None or self.owner_id == vid:
            self.owner_id = vid
            self.lock_expiry = current_time + packet['duration']
            # 触发动作
            direction = packet.get('direction', 'NORMAL')
            self._set_target(direction)
            return {'status': 'GRANTED'}

        # 如果被占用，比较优先级/时间戳
        holder_ts = self.timestamp_table.get(self.owner_id, 0.0)

        # 策略：老事务(ts小) 抢占 新事务(ts大)
        # "Wound": If Requesting_TS < Holder_TS -> Preempt
        if ts < holder_ts:
            logger.warning(f"Switch {self.id}: {vid} (Old) PREEMPTS {self.owner_id} (New)")
            self.owner_id = vid
            self.lock_expiry = current_time + packet['duration']
            self._set_target(packet.get('direction', 'NORMAL'))
            return {'status': 'GRANTED_PREEMPT'}

        # "Wait": If Requesting_TS > Holder_TS -> Wait
        return {'status': 'WAIT_BUSY', 'holder': self.owner_id}

    def _set_target(self, direction):
        needed = self.params.stroke if direction == 'REVERSE' else 0.0
        if abs(self.mechanism.pos - needed) > 0.005:
            self.state = SwitchState.UNLOCKING
            self.target_pos = needed

    # --- Real-time Simulation Step (Renamed to update) ---

    def update(self, dt, current_time):
        """
        执行物理仿真与边缘控制逻辑
        (Previously named step, renamed to match call signature in map_core.py)
        """
        # 1. 自动释放过期锁
        if self.owner_id and current_time > self.lock_expiry:
            # logger.info(f"Switch {self.id}: Lock EXPIRED for {self.owner_id}")
            self.owner_id = None

        # 2. PHM: 健康监测
        if self.state == SwitchState.STALLED:
            return  # 故障停机

        # 3. 控制律 (Control Law)
        voltage_cmd = 0.0

        if self.state in [SwitchState.UNLOCKING, SwitchState.MOVING, SwitchState.LOCKING]:
            # PID or Bang-Bang with Overdrive
            err = self.target_pos - self.mechanism.pos

            # 基础电压 24V
            u_base = 24.0

            # 自适应增强: 如果泥浆厚且速度慢，提升电压至 48V
            if abs(self.motor.current) > 10.0 and abs(self.mechanism.omega) < 5.0:
                u_base = 48.0  # Boost for high torque
                self.health_index -= 0.001 * dt  # 寿命损耗

            voltage_cmd = u_base * np.sign(err)

            # 接近终点时减速 (Soft landing)
            if abs(err) < 0.01: voltage_cmd *= 0.5

        # 4. 物理求解 (RK4 inside motor)
        # 随机故障注入: 异物卡滞
        ext_force = 0.0
        if np.random.random() < 0.0001 * self.mud:
            ext_force = 5000.0  # Stone jam

        torque = self.motor.step(dt, voltage_cmd, self.mechanism.omega)
        pos, vel = self.mechanism.step(dt, torque, ext_force)

        # 5. 状态机流转
        self._update_state_machine(pos, vel, self.motor.current, dt)

        # 6. 数据记录
        self.current_buffer.append(self.motor.current)

    def _update_state_machine(self, pos, vel, current, dt):
        # Unlocking -> Moving
        if self.state == SwitchState.UNLOCKING:
            # 假设前 10mm 是解锁行程
            if (self.target_pos > 0.1 and pos > 0.01) or (self.target_pos < 0.1 and pos < self.params.stroke - 0.01):
                self.state = SwitchState.MOVING

        # Moving -> Locking
        elif self.state == SwitchState.MOVING:
            dist = abs(pos - self.target_pos)
            if dist < 0.01:
                self.state = SwitchState.LOCKING

            # 堵转检测
            if abs(current) > 20.0 and abs(vel) < 0.1:
                self.stall_counter += dt
                if self.stall_counter > 1.5:
                    self.state = SwitchState.STALLED
                    logger.error(f"Switch {self.id} STALLED! Thermal overload imminent.")
            else:
                self.stall_counter = 0.0

        # Locking -> Locked
        elif self.state == SwitchState.LOCKING:
            if abs(pos - self.target_pos) < 0.001:
                self.state = SwitchState.LOCKED_REVERSE if self.target_pos > 0.1 else SwitchState.LOCKED_NORMAL

    def get_telemetry(self):
        """Dashboard Data"""
        return {
            'id': self.id,
            'state': self.state.name,
            'temp': self.motor.temperature,
            'current': self.motor.current,
            'pos': self.mechanism.pos,
            'health': self.health_index,
            'owner': self.owner_id
        }