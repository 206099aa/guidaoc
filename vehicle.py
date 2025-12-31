import numpy as np
import math
import logging
import random
from enum import Enum, auto
from collections import deque
from dataclasses import dataclass


# --- 引入底层物理内核 ---
from physics import RailVehicleMBDSystem

# 配置日志
logger = logging.getLogger("DeepVehicleAgent")


class VehicleState(Enum):
    """
    [FSM] Finite State Machine for Autonomous Edge Agent.
    """
    IDLE = auto()  # 休眠/待命
    PLANNING = auto()  # 路径规划中
    NEGOTIATING = auto()  # 边缘协同中 (V2I Handshake)
    TRACTION_CONTROL = auto()  # 正常行驶 (SMC Control)
    COASTING = auto()  # 惰行节能
    BRAKING_NORMAL = auto()  # 进站制动
    BRAKING_EMERGENCY = auto()  # ATP 触发紧急制动
    FAULT_RECOVERY = auto()  # 故障降级模式


@dataclass
class EnergyAudit:
    """[Energy Model] Fine-grained energy consumption tracking."""
    traction_joules: float = 0.0  # 电机做功
    compute_joules: float = 0.0  # 边缘计算功耗
    comm_joules: float = 0.0  # 通信射频功耗

    @property
    def total(self): return self.traction_joules + self.compute_joules + self.comm_joules


class SlidingModeController:
    """
    [Control Algo] Robust Sliding Mode Controller (SMC).
    Designed to handle high uncertainty in mud friction.
    Control Law: u = u_eq + u_sw
    u_sw = -K * sat(s/phi)
    """

    def __init__(self, mass, max_voltage, k_gain=15.0, lambda_s=0.5):
        self.mass = mass
        self.max_voltage = max_voltage
        self.K = k_gain  # 鲁棒增益 (应对扰动上限)
        self.lam = lambda_s  # 滑模面收敛率
        self.phi = 2.0  # 边界层厚度 (抑制抖振 Chattering)
        self.integral_e = 0.0  # 误差积分

    def compute(self, target_v, current_v, dt, estimated_resistance):
        """
        计算电机电压指令
        """
        # 1. 定义误差与滑模面
        e = target_v - current_v
        self.integral_e += e * dt
        s = e + self.lam * self.integral_e

        # 2. 等效控制 (Equivalent Control)
        u_eq = estimated_resistance * 0.05  # 粗略前馈

        # 3. 切换控制 (Switching Control)
        sat_s = np.clip(s / self.phi, -1.0, 1.0)
        u_sw = self.K * sat_s

        # 4. 总输出与饱和
        u_total = u_eq + u_sw
        u_clamped = np.clip(u_total, -self.max_voltage, self.max_voltage)

        return u_clamped, s


class VehicleAgent:
    """
    [Edge AI Agent]
    Autonomous Rail Vehicle for Paddy Fields.
    Features:
    - V2I Distributed Locking (Wound-Wait).
    - On-board Safety Layer (ATP).
    - High-Fidelity Physics Integration.
    - [NEW] SCI-Grade Energy Auditing (SEC, Power Analysis)
    """

    def __init__(self,
                 agent_id: str,
                 vehicle_type_cfg: dict,
                 env_config: dict,
                 start_node: str,
                 map_graph,
                 infra_agents: dict):  # {node_id: EdgeSwitchAgent}

        self.id = agent_id
        self.cfg = vehicle_type_cfg

        # [SCI Requirement] Ensure critical attributes exist
        self.cfg.setdefault('mass_full', 12000.0)
        self.cfg.setdefault('length', 8.5)
        self.cfg.setdefault('max_speed', 15.0)

        self.env = env_config
        self.map = map_graph
        self.infra = infra_agents  # 感知范围内的基础设施

        # --- 1. 物理内核实例化 ---
        self.physics = RailVehicleMBDSystem(
            vehicle_config=self.cfg,
            env_config=self.env
        )

        # 初始位置设定
        if start_node in self.map.nodes:
            self.pos_2d = np.array(self.map.nodes[start_node].pos, dtype=float)
            self.physics._init_position(spacing=2.0)
        else:
            self.pos_2d = np.array([0.0, 0.0])

        # --- 2. 边缘智能核心 ---
        self.timestamp = 0.0 + random.random()
        self.state = VehicleState.IDLE
        self.path_queue = deque()
        self.current_lock = None

        # --- 3. 控制与安全 ---
        self.controller = SlidingModeController(
            mass=self.cfg['mass_full'],
            max_voltage=48.0,
            k_gain=10.0
        )
        self.energy = EnergyAudit()

        # ATP 安全参数
        self.safe_braking_decel = 0.5  # m/s^2 (泥地保守值)
        self.comm_range = 500.0  # V2I 通信距离

        # --- 4. [NEW] SCI Telemetry Accumulators ---
        self.dist_accumulated = 0.0
        self.time_active = 0.0
        self.last_telemetry = {}

    def step(self, dt, global_time):
        """
        [Real-time Loop] 10Hz - 50Hz Control Loop.
        Perception -> Negotiation -> Planning -> Control -> Actuation.
        """
        # 1. 状态感知 (Perception)
        real_v = self.physics.state[1]
        sensor_v = real_v + np.random.normal(0, 0.05)  # 速度传感器噪声

        # 2. 边缘计算功耗
        self.energy.compute_joules += 2.0 * dt

        # --- 状态机逻辑 ---
        u_cmd = 0.0

        if self.state == VehicleState.IDLE:
            if global_time > 1.0 and not self.path_queue:
                self._plan_mission()
                self.state = VehicleState.NEGOTIATING

        elif self.state == VehicleState.NEGOTIATING:
            self.energy.compute_joules += 5.0 * dt
            if not self.path_queue:
                self.state = VehicleState.IDLE
            else:
                target_node = self.path_queue[0]
                if self._v2i_handshake(target_node, global_time):
                    logger.info(f"[{self.id}] Edge Lock Acquired: {target_node}")
                    self.state = VehicleState.TRACTION_CONTROL
                else:
                    u_cmd = 0.0

        elif self.state == VehicleState.TRACTION_CONTROL:
            if not self.path_queue:
                self.state = VehicleState.BRAKING_NORMAL
            else:
                dist_to_target = self._dist_to(self.path_queue[0])
                braking_dist = (sensor_v ** 2) / (2 * self.safe_braking_decel * (1 - 0.5 * self.env['mud_factor']))

                if dist_to_target < braking_dist + 2.0:
                    self.state = VehicleState.BRAKING_NORMAL
                else:
                    target_v = self.cfg['max_speed']
                    est_resist = 50.0 + 10.0 * sensor_v + 0.5 * sensor_v ** 2
                    u_cmd, _ = self.controller.compute(target_v, sensor_v, dt, est_resist)
                    # Arrived Check
                    if dist_to_target < 2.0:
                        self.path_queue.popleft()

        elif self.state == VehicleState.BRAKING_NORMAL:
            if not self.path_queue:
                u_cmd = 0.0
                self.state = VehicleState.IDLE
            else:
                dist_to_target = self._dist_to(self.path_queue[0])
                if dist_to_target < 0.5:
                    self.path_queue.popleft()
                    if not self.path_queue:
                        self.state = VehicleState.IDLE
                        u_cmd = 0.0
                    else:
                        self.state = VehicleState.NEGOTIATING
                else:
                    u_cmd = -24.0

        # --- 3. 物理执行 (Actuation) ---
        dynamics = self.physics.step_rk4(dt, u_cmd)

        # --- 4. 能耗审计与上报 [SCI Upgrade] ---
        # 瞬时功率 P_inst = U * I
        p_inst = abs(u_cmd * dynamics['motor_current'])  # Watts
        self.energy.traction_joules += p_inst * dt

        # 运动学更新
        v_mps = dynamics['loco_vel']

        # [CRITICAL FIX] 移除了 > 0.01 的阈值判断。
        # 只要有任何微小的速度，都进行里程累加和位置更新。
        # 这解决了“电机在转但车看起来不动”的 Bug。
        if abs(v_mps) > 0.0:
            self.time_active += dt
            dist_step = abs(v_mps * dt)
            self.dist_accumulated += dist_step
            self._update_kinematics(v_mps, dt)

        # 5. 打包全量遥测数据
        self.last_telemetry = {
            'id': self.id,
            'state': self.state.name,
            'pos': self.pos_2d,
            'vel': v_mps,
            'force': dynamics['coupler_force_1'],
            'current': dynamics['motor_current'],
            'rssi': -65.0,
            'energy_total': self.energy.total,
            'mud': self.env['mud_factor'],

            # [NEW SCI METRICS]
            'p_inst': p_inst,  # 瞬时功率 (W)
            'mu': dynamics['mu_effective'],  # 微观摩擦系数
            'mass': dynamics['mass_total'],  # 车辆质量
            'length': dynamics['length'],  # 车辆长度
            'dist_accum': self.dist_accumulated,  # 里程 (m)
            'time_active': self.time_active  # 活跃时间 (s)
        }
        return self.last_telemetry

    def _v2i_handshake(self, target_node_id, global_time):
        target_infra = self.infra.get(target_node_id)
        if not target_infra: return True

        if target_node_id in self.map.nodes:
            target_pos_2d = np.array(self.map.nodes[target_node_id].pos)
            dist = np.linalg.norm(self.pos_2d - target_pos_2d)
        else:
            dist = 100.0

        tx_power = 0.001 * (1 + (dist / 100.0) ** 2)
        self.energy.comm_joules += tx_power * 0.05

        success_pack = target_infra.handle_access_request({
            'vid': self.id, 'timestamp': self.timestamp, 'priority': 10,
            'duration': 20.0, 'global_time': global_time, 'action': 'LOCK', 'direction': 'NORMAL'
        })

        status = success_pack.get('status', 'FAIL')
        if status in ['GRANTED', 'GRANTED_PREEMPT']:
            self.current_lock = target_node_id
            return True
        else:
            return False

    def _plan_mission(self):
        if "Hauler" in self.id:
            self.path_queue = deque(["N_0_0", "N_0_1", "Stop_H_0_1"])
        else:
            self.path_queue = deque(["N_2_0", "N_2_1", "Stop_H_2_1"])

    def _dist_to(self, node_id):
        if node_id not in self.map.nodes: return 0.0
        target_pos = np.array(self.map.nodes[node_id].pos)
        return np.linalg.norm(self.pos_2d - target_pos)

    def _update_kinematics(self, vel_1d, dt):
        if not self.path_queue: return
        target_id = self.path_queue[0]
        if target_id not in self.map.nodes: return
        target_pos = np.array(self.map.nodes[target_id].pos)
        vec = target_pos - self.pos_2d
        dist = np.linalg.norm(vec)
        if dist > 0.1:
            direction = vec / dist
            self.pos_2d += direction * vel_1d * dt


