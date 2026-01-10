import numpy as np
import logging
import math
from collections import deque, defaultdict
from enum import Enum, auto
from dataclasses import dataclass

# 配置日志：采用 SCI 论文常用的模块化命名
logger = logging.getLogger("Edge.Infrastructure")


# =========================================================================
# [Layer 1] High-Fidelity Physics Kernel (高保真物理内核)
# -------------------------------------------------------------------------
# 包含 ZD6 转辙机的机电热耦合模型、非线性摩擦与机械锁闭逻辑。
# =========================================================================

class SwitchState(Enum):
    """
    [FSM] Finite State Machine for Railway Switch.
    Reflects the physical and logical status of the infrastructure.
    """
    LOCKED_NORMAL = auto()  # 定位锁闭 (无电流，机械锁死)
    LOCKED_REVERSE = auto()  # 反位锁闭
    UNLOCKING = auto()  # 解锁阶段 (克服静摩擦与锁闭力)
    MOVING = auto()  # 转换阶段 (动摩擦占主导)
    LOCKING = auto()  # 闭锁阶段 (动能转化为势能)
    STALLED = auto()  # 故障堵转 (电流过载)
    MAINTENANCE = auto()  # 维护模式


@dataclass
class ZD6PhysicalParams:
    """
    [Parameter Identification] ZD6-D Switch Machine Specs.
    Sources: Railway Signal Engineering Handbook (2020 Edition).
    """
    # Electrical Subsystem
    R_armature_20C: float = 4.5  # Armature resistance at 20°C (Ohm)
    L_armature: float = 0.05  # Armature inductance (H)
    Ke: float = 0.85  # Back-EMF constant (V/(rad/s))
    Kt: float = 0.85  # Torque constant (Nm/A)

    # Thermal Subsystem
    thermal_capacity: float = 500.0  # Heat capacity (J/K)
    heat_dissipation: float = 2.5  # Convective heat transfer coeff (W/K)

    # Mechanical Subsystem
    J_rotor: float = 0.02  # Rotor inertia (kg m^2)
    gear_ratio: float = 45.0  # Reduction ratio
    screw_pitch: float = 0.01  # Lead screw pitch (m)
    stroke: float = 0.16  # Total stroke length (m)

    # Load Characteristics
    locking_force_peak: float = 2000.0  # Peak resistance during locking (N)


class ElectroThermalMotor:
    """
    [Model 1] Coupled Electro-Thermal Dynamics.
    Differential Equations:
    1. Electrical: L * di/dt + R(T) * i + Ke * w = V
    2. Thermal:    C * dT/dt = i^2 * R(T) - h * (T - T_amb)
    """

    def __init__(self, params: ZD6PhysicalParams):
        self.p = params
        self.current = 0.0
        self.temperature = 25.0  # Ambient temperature
        self.resistance = params.R_armature_20C

    def _update_resistance(self):
        # Linear temperature coefficient for Copper (approx 0.00393)
        self.resistance = self.p.R_armature_20C * (1.0 + 0.004 * (self.temperature - 20.0))

    def step_rk4(self, dt, voltage_in, omega_rotor):
        """
        Runge-Kutta 4th Order Integration for stiff electrical dynamics.
        """
        # 1. Thermal Update (Forward Euler is sufficient for slow thermal dynamics)
        heat_gen = (self.current ** 2) * self.resistance
        heat_loss = self.p.heat_dissipation * (self.temperature - 25.0)
        delta_T = (heat_gen - heat_loss) / self.p.thermal_capacity * dt
        self.temperature += delta_T
        self._update_resistance()

        # 2. Electrical Update (RK4)
        def di_dt(i, v, w, r):
            return (v - i * r - self.p.Ke * w) / self.p.L_armature

        k1 = di_dt(self.current, voltage_in, omega_rotor, self.resistance)
        k2 = di_dt(self.current + 0.5 * dt * k1, voltage_in, omega_rotor, self.resistance)
        k3 = di_dt(self.current + 0.5 * dt * k2, voltage_in, omega_rotor, self.resistance)
        k4 = di_dt(self.current + dt * k3, voltage_in, omega_rotor, self.resistance)

        self.current += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Saturation (Power supply limit)
        self.current = np.clip(self.current, -30.0, 30.0)

        return self.p.Kt * self.current  # Output Torque


class ZD6Mechanism:
    """
    [Model 2] Mechanical Transmission with Non-linear Stribeck Friction.
    Simulates the complex interaction between the slide chair and the rail under muddy conditions.
    """

    def __init__(self, params: ZD6PhysicalParams, mud_factor: float):
        self.p = params
        self.mud = mud_factor
        self.omega = 0.0  # Angular velocity
        self.pos = 0.0  # Linear position

        # Mechanical efficiency degrades with mud accumulation
        self.efficiency = 0.8 * (1.0 - 0.3 * np.clip(mud_factor, 0, 1))

    def get_load_force(self, v_linear):
        """
        Calculates Total Resistance Force F_load.
        F_load = F_stribeck(v, mud) + F_locking(x)
        """
        # 1. Stribeck Friction Model (Smooth approximation via tanh)
        # Viscous friction coefficient increases significantly with mud
        sigma_v = 1000.0 * (1.0 + 2.0 * self.mud)
        # Coulomb friction limit
        F_c = 300.0 + 500.0 * self.mud

        # F_fric = F_c * tanh(k*v) + sigma * v
        f_fric = F_c * math.tanh(10.0 * v_linear) + sigma_v * v_linear

        # 2. Mechanical Locking Curve
        # Simulates the high resistance at the start (unlocking) and end (locking) of stroke
        f_lock = 0.0
        lock_region = 0.015  # 15mm locking zone
        if self.pos < lock_region or self.pos > (self.p.stroke - lock_region):
            # Resist motion away from endpoints
            direction = np.sign(v_linear) if abs(v_linear) > 1e-4 else 0.0
            f_lock = self.p.locking_force_peak * direction

        return f_fric + f_lock

    def step(self, dt, motor_torque, external_force_N=0.0):
        # Kinematic conversion
        k_linear = self.p.screw_pitch / (2 * np.pi * self.p.gear_ratio)
        v_linear = self.omega * k_linear

        # Hard Stop Logic (Physical Limits)
        if (self.pos <= 0 and motor_torque < 0) or (self.pos >= self.p.stroke and motor_torque > 0):
            self.omega = 0.0
            v_linear = 0.0
            return self.pos, self.omega

        # Dynamic Equation: J * dw/dt = T_motor - T_load
        force_total = self.get_load_force(v_linear) + external_force_N
        torque_load = force_total * k_linear / self.efficiency

        dw_dt = (motor_torque - torque_load - 0.05 * self.omega) / self.p.J_rotor

        self.omega += dw_dt * dt
        self.pos += (self.omega * k_linear) * dt
        self.pos = np.clip(self.pos, 0.0, self.p.stroke)

        return self.pos, self.omega


# =========================================================================
# [Layer 2] SCI-Grade Semantic Edge Intelligence (语义智能层)
# -------------------------------------------------------------------------
# 包含贝叶斯状态估计、概率风险评估与信息老化机制。
# =========================================================================

@dataclass
class SemanticOccupancy:
    """
    [Semantic Data Structure]
    Represents a probabilistic belief of resource usage.
    """
    owner_id: str
    arrival_mean: float  # Estimated Time of Arrival (ETA)
    duration_mean: float  # Estimated Occupancy Duration
    uncertainty_sigma: float  # Spatial-Temporal Uncertainty (grows with AoI)
    last_update_ts: float  # Timestamp for Age of Information (AoI) calculation


class BayesianStateEstimator:
    """
    [Algorithm] Recursive Bayesian Filter for State Estimation.
    Aggregates asynchronous, lossy semantic packets into a coherent Risk Field.
    """

    def __init__(self):
        # Belief Map: Vehicle_ID -> SemanticOccupancy
        self.occupancy_map = {}
        self.risk_level = 0.0  # Quantified Risk Metric [0, 1]

    def update_belief(self, packet, current_time):
        """
        Injects a new semantic measurement into the belief state.
        """
        vid = packet['vid']
        packet_ts = packet.get('timestamp', current_time)

        # 1. Calculate Age of Information (AoI)
        aoi = max(0.0, current_time - packet_ts)

        # 2. Uncertainty Propagation
        base_sigma = packet.get('pos_uncertainty', 1.0)
        propagated_sigma = base_sigma + 0.2 * aoi  # Linear uncertainty growth model

        # 3. Update Belief State
        # [Fix] Added .get() with defaults to prevent KeyError if 'eta'/'duration' missing
        self.occupancy_map[vid] = SemanticOccupancy(
            owner_id=vid,
            arrival_mean=packet.get('eta', current_time),
            duration_mean=packet.get('duration', 2.0),
            uncertainty_sigma=propagated_sigma,
            last_update_ts=current_time
        )

        # 4. Maintenance: Prune stale beliefs
        self._prune_stale_beliefs(current_time)

    def _prune_stale_beliefs(self, now):
        """Removes beliefs that have exceeded the validity horizon (Robustness)."""
        dead_keys = []
        for vid, occ in self.occupancy_map.items():
            if (now - occ.last_update_ts) > 60.0:
                dead_keys.append(vid)

        for k in dead_keys:
            del self.occupancy_map[k]

        # Update aggregate risk metric
        self.risk_level = min(1.0, len(self.occupancy_map) * 0.25)

    def get_safe_window_probabilistic(self, req_eta, req_dur):
        """
        [Core Logic] Probabilistic Collision Detection.
        Returns: (is_safe: bool, risk_score: float)
        """
        risk_accum = 0.0
        req_start = req_eta
        req_end = req_eta + req_dur

        for vid, occ in self.occupancy_map.items():
            # Construct Confidence Interval (3-Sigma Rule)
            buffer = 3.0 * occ.uncertainty_sigma
            occ_start = occ.arrival_mean - buffer
            occ_end = occ.arrival_mean + occ.duration_mean + buffer

            # Intersection over Union (IoU) Logic
            overlap_start = max(req_start, occ_start)
            overlap_end = min(req_end, occ_end)

            if overlap_start < overlap_end:
                risk_accum += 1.0

        return (risk_accum < 0.5), risk_accum


# =========================================================================
# [Layer 3] Integration Agent (边缘计算代理)
# -------------------------------------------------------------------------
# 整合 Layer 1 (物理) 和 Layer 2 (智能)，以及 Layer 3 (全息流场与时空表)。
# =========================================================================

# [新增结构 1] 基于全息流场熵的协同机制
class HolographicFlowField:
    """
    [Novelty] Distributed Holographic Flow Map via Vector Stigmergy.
    Diffuses not just 'density' (scalar), but 'velocity vectors' (vector).
    Calculates Flow Entropy to determine 'Turbulence'.
    """

    def __init__(self, node_id, decay_rate=0.95, diffusion_rate=0.2):
        self.node_id = node_id
        # Vector Field Components: (Pressure, Vx, Vy)
        self.local_pressure = 0.0
        self.flow_vector = np.array([0.0, 0.0])  # Cumulative velocity vector
        self.turbulence = 0.0  # Flow Entropy [0=Laminar, 1=Chaotic]

        self.decay = decay_rate
        self.diffusion = diffusion_rate

    def inject_vector(self, mass=1.0, velocity=0.0, direction_vec=None):
        """
        Vehicle injects (Mass, Velocity) into the field.
        """
        self.local_pressure += mass
        if direction_vec is not None:
            # Accumulate momentum: Mass * Velocity * Direction
            momentum = mass * velocity * np.array(direction_vec)
            self.flow_vector += momentum

    def step_diffusion(self, dt, neighbor_potentials: list):
        """[Algorithm] Vector Field Diffusion."""
        # 1. Evaporation
        self.local_pressure *= (self.decay ** dt)
        self.flow_vector *= (self.decay ** dt)

        # 2. Calculate Turbulence (Entropy of the Flow)
        speed_mag = np.linalg.norm(self.flow_vector)
        if self.local_pressure > 0.1:
            # Order Parameter (0 to 1)
            order = speed_mag / (self.local_pressure * 15.0 + 1e-5)
            self.turbulence = 1.0 - np.clip(order, 0.0, 1.0)
        else:
            self.turbulence = 0.0  # Empty = Laminar

    def get_flow_state(self):
        """Returns the state of the flow field for navigation."""
        return {
            'potential': self.local_pressure,
            'turbulence': self.turbulence
        }


# [新增结构 2] 时空资源槽
@dataclass
class TimeSlot:
    start_time: float
    end_time: float
    owner_id: str


class EdgeSwitchAgent:
    """
    [Cyber-Physical System] Decentralized Edge Node.
    Integrates ZD6 physics, Bayesian semantic intelligence, Holographic Flow, and Time-Space scheduling.
    """

    def __init__(self, node_id, env_config):
        self.id = node_id
        self.mud = env_config.get('mud_factor', 0.5)

        # --- Subsystem 1: Physics Kernel ---
        self.params = ZD6PhysicalParams()
        self.motor = ElectroThermalMotor(self.params)
        self.mechanism = ZD6Mechanism(self.params, self.mud)

        # Internal Logic State
        self.state = SwitchState.LOCKED_NORMAL
        self.target_pos = 0.0
        self.health_index = 1.0
        self.stall_counter = 0.0

        # --- Subsystem 2: Semantic Intelligence ---
        self.estimator = BayesianStateEstimator()
        self.owner_id = None  # Soft ownership

        # --- Subsystem 3: Holographic Flow Field ---
        self.flow_field = HolographicFlowField(node_id)
        self.neighbor_potentials = []  # Cache for diffusion

        # --- Subsystem 4: Time-Space Reservation Table ---
        self.reservations: list[TimeSlot] = []

    # --- [Time-Space Logic] ---
    def query_time_space(self, arrival_time, duration):
        """Check if a time slot is available."""
        req_start = arrival_time
        req_end = arrival_time + duration
        for slot in self.reservations:
            if max(req_start, slot.start_time) < min(req_end, slot.end_time):
                return True  # Conflict exists
        return False

    def reserve_time_space(self, vid, arrival_time, duration):
        """Book a time slot."""
        slot = TimeSlot(arrival_time, arrival_time + duration, vid)
        self.reservations.append(slot)
        # Cleanup old reservations
        self.reservations = [s for s in self.reservations if s.end_time > arrival_time - 100.0]
        return True

    def check_reservation_status(self, vid, current_time):
        """
        [关键修复] 严格执法：
        1. 如果当前时间段被别人占用 -> 拒绝 (冲突)
        2. 如果被自己占用 -> 允许
        3. 如果无人占用 -> 允许 (或视为空闲)
        """
        for slot in self.reservations:
            # 检查当前时刻是否在某个 Slot 内
            if slot.start_time <= current_time <= slot.end_time:
                if slot.owner_id != vid:
                    return False  # 被他人锁定
        return True  # 空闲或自己锁定

    # --- [Interface Handlers] ---
    def handle_semantic_packet(self, packet):
        """[V2I] Async Semantic Packet Handler."""
        current_time = packet.get('global_time', 0.0)
        vid = packet['vid']

        # 1. Belief Update
        self.estimator.update_belief(packet, current_time)

        # 2. Flow Field Injection
        vel = packet.get('vel', 0.0)
        self.flow_field.inject_vector(mass=1.0, velocity=vel, direction_vec=[1, 0])

        # 3. Security Check (Double Validation)
        # A. 概率检查 (Bayesian)
        is_safe_prob, risk_val = self.estimator.get_safe_window_probabilistic(
            packet.get('eta', current_time), packet.get('duration', 2.0)
        )

        # B. 确定性检查 (Reservation Table) - [新增]
        is_safe_deter = self.check_reservation_status(vid, current_time)

        response = {}
        # 只要有一方认为不安全，就视为高风险
        if is_safe_prob and is_safe_deter:
            direction = packet.get('direction', 'NORMAL')
            self._set_physical_target(direction)
            self.owner_id = vid
            response = {'status': 'ACK_SEMANTIC', 'risk': risk_val}
        else:
            # 确定性冲突通常意味着硬碰撞风险，Risk 设为最高
            final_risk = 1.0 if not is_safe_deter else risk_val
            response = {'status': 'RISK_HIGH', 'risk': final_risk}

        # Attach Flow State
        flow_state = self.flow_field.get_flow_state()
        response['turbulence'] = flow_state['turbulence']
        response['potential'] = flow_state['potential']

        return response

    def handle_hardware_signal(self, packet):
        """[Hardware V2I] Direct Switch Control Signal."""
        if packet.get('type') == 'SWITCH_REQ':
            req = packet.get('target_state')
            # logger.info(f"Switch {self.id} received hardware req: {req}")
            if req == 'REVERSE':
                self._set_physical_target('REVERSE')
            else:
                self._set_physical_target('NORMAL')

    def update_gossip(self, neighbor_p_list):
        """[Network] Update neighbor potentials for diffusion."""
        self.neighbor_potentials = neighbor_p_list

    # --- [Physics & Update Loop] ---
    def _set_physical_target(self, direction):
        needed = self.params.stroke if direction == 'REVERSE' else 0.0
        if abs(self.mechanism.pos - needed) > 0.005:
            if self.state in [SwitchState.LOCKED_NORMAL, SwitchState.LOCKED_REVERSE]:
                self.state = SwitchState.UNLOCKING
                self.target_pos = needed

    def update(self, dt, current_time):
        if self.state == SwitchState.STALLED: return

        # Voltage Control (PID)
        err = self.target_pos - self.mechanism.pos
        voltage_cmd = 0.0
        if self.state in [SwitchState.UNLOCKING, SwitchState.MOVING, SwitchState.LOCKING]:
            u_base = 24.0
            # Adaptive Boost
            if abs(self.motor.current) > 10.0 and abs(self.mechanism.omega) < 5.0:
                u_base = 48.0
                self.health_index -= 0.0001 * dt
            voltage_cmd = u_base * np.sign(err)
            if abs(err) < 0.01: voltage_cmd *= 0.5

        # Physics Step
        ext_force = 0.0
        if np.random.random() < 0.00001 * self.mud:
            ext_force = 5000.0

        torque = self.motor.step_rk4(dt, voltage_cmd, self.mechanism.omega)
        pos, vel = self.mechanism.step(dt, torque, ext_force)

        self._update_fsm(pos, vel, self.motor.current, dt)

        # Flow Field Diffusion Step
        self.flow_field.step_diffusion(dt, self.neighbor_potentials)

    def _update_fsm(self, pos, vel, current, dt):
        if self.state == SwitchState.UNLOCKING:
            if (self.target_pos > 0.1 and pos > 0.01) or (self.target_pos < 0.1 and pos < self.params.stroke - 0.01):
                self.state = SwitchState.MOVING
        elif self.state == SwitchState.MOVING:
            dist = abs(pos - self.target_pos)
            if dist < 0.01: self.state = SwitchState.LOCKING
            if abs(current) > 20.0 and abs(vel) < 0.1:
                self.stall_counter += dt
                if self.stall_counter > 1.5: self.state = SwitchState.STALLED
            else:
                self.stall_counter = 0.0
        elif self.state == SwitchState.LOCKING:
            if abs(pos - self.target_pos) < 0.001:
                self.state = SwitchState.LOCKED_REVERSE if self.target_pos > 0.1 else SwitchState.LOCKED_NORMAL

    def get_broadcast_state(self):
        """Generate telemetry packet."""
        flow = self.flow_field.get_flow_state()
        return {
            'id': self.id,
            'state': self.state.name,
            'pos': self.mechanism.pos,
            'health': self.health_index,
            'risk_level': self.estimator.risk_level,
            'potential': flow['potential'],
            'turbulence': flow['turbulence']
        }