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
# 对应 DeepSnake 版本的核心优势：物理真实性与 PHM 基础。
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
# 对应 RobustSnake 版本的核心优势：弱网适配性与分布鲁棒性。
# =========================================================================

@dataclass
class SemanticOccupancy:
    """
    [Semantic Data Structure]
    Represents a probabilistic belief of resource usage.
    Unlike a deterministic lock, this includes uncertainty metrics.
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

        Args:
            packet: {vid, eta, duration, pos_uncertainty, timestamp}
            current_time: Global simulation time
        """
        vid = packet['vid']
        packet_ts = packet.get('timestamp', current_time)

        # 1. Calculate Age of Information (AoI)
        # AoI represents the freshness of the data. High AoI -> High Uncertainty.
        aoi = max(0.0, current_time - packet_ts)

        # 2. Uncertainty Propagation
        # Sigma(t) = Sigma_reported + alpha * AoI
        base_sigma = packet.get('pos_uncertainty', 1.0)
        propagated_sigma = base_sigma + 0.2 * aoi  # Linear uncertainty growth model

        # 3. Update Belief State
        self.occupancy_map[vid] = SemanticOccupancy(
            owner_id=vid,
            arrival_mean=packet['eta'],
            duration_mean=packet['duration'],
            uncertainty_sigma=propagated_sigma,
            last_update_ts=current_time
        )

        # 4. Maintenance: Prune stale beliefs to prevent ghost blockages
        self._prune_stale_beliefs(current_time)

    def _prune_stale_beliefs(self, now):
        """
        Removes beliefs that have exceeded the validity horizon (Robustness).
        Threshold: 60s (Assumed max coherence time in weak net).
        """
        dead_keys = []
        for vid, occ in self.occupancy_map.items():
            if (now - occ.last_update_ts) > 60.0:
                dead_keys.append(vid)

        for k in dead_keys:
            del self.occupancy_map[k]

        # Update aggregate risk metric (Entropy proxy)
        # Simple heuristic: more occupants = higher risk
        self.risk_level = min(1.0, len(self.occupancy_map) * 0.25)

    def get_safe_window_probabilistic(self, req_eta, req_dur):
        """
        [Core Logic] Probabilistic Collision Detection.
        Instead of checking deterministic overlap, we check Gaussian overlap probability.

        Returns:
            (is_safe: bool, risk_score: float)
        """
        risk_accum = 0.0
        req_start = req_eta
        req_end = req_eta + req_dur

        for vid, occ in self.occupancy_map.items():
            # Construct Confidence Interval (3-Sigma Rule -> 99.7%)
            buffer = 3.0 * occ.uncertainty_sigma

            occ_start = occ.arrival_mean - buffer
            occ_end = occ.arrival_mean + occ.duration_mean + buffer

            # Intersection over Union (IoU) Logic for 1D time intervals
            overlap_start = max(req_start, occ_start)
            overlap_end = min(req_end, occ_end)

            if overlap_start < overlap_end:
                # Collision detected in belief space
                # Contribution to risk depends on uncertainty (wider uncertainty = wider blocking)
                risk_accum += 1.0

        # Thresholding: Risk < 0.5 means <50% chance of substantial conflict
        return (risk_accum < 0.5), risk_accum


# =========================================================================
# [Layer 3] Integration Agent (边缘计算代理)
# -------------------------------------------------------------------------
# 整合 Layer 1 (物理) 和 Layer 2 (智能)，对外提供 V2I 接口。
# =========================================================================

class EdgeSwitchAgent:
    """
    [Cyber-Physical System] Decentralized Edge Node.
    Integrates ZD6 physics with Bayesian semantic intelligence.

    Key Features:
    1. No-ACK Semantic Communication (Fire-and-Forget).
    2. Probabilistic Reservation Logic.
    3. Adaptive Voltage Control based on Physical Load.
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
        self.owner_id = None  # Soft ownership based on semantic agreement

    def handle_semantic_packet(self, packet):
        """
        [V2I Interface] Handles asynchronous semantic packets.
        This method is 'Fire-and-Forget' compatible (No ACK required for protocol correctness).

        Args:
            packet: {vid, eta, duration, pos_uncertainty, direction, global_time}
        Returns:
            dict: {status, risk} (Immediate feedback, though transmission may be lossy)
        """
        current_time = packet.get('global_time', 0.0)

        # 1. Belief Update (Bayesian Filtering)
        self.estimator.update_belief(packet, current_time)

        # 2. Risk Assessment (Chance-Constrained Check)
        is_safe, risk = self.estimator.get_safe_window_probabilistic(
            packet['eta'], packet['duration']
        )

        if is_safe:
            # Low Risk -> Actuate Physics
            direction = packet.get('direction', 'NORMAL')
            self._set_physical_target(direction)
            self.owner_id = packet['vid']
            return {'status': 'ACK_SEMANTIC', 'risk': risk}
        else:
            # High Risk -> Reject (Vehicle must brake)
            return {'status': 'RISK_HIGH', 'risk': risk}

    def get_broadcast_state(self):
        """
        [Digital Pheromone] Generates a compressed state packet for broadcast.
        Used by vehicles to build their local potential field map.
        """
        return {
            'id': self.id,
            'risk_level': self.estimator.risk_level,  # Semantic Congestion
            'pos': self.mechanism.pos,  # Physical State
            'health': self.health_index,  # PHM State
            'state': self.state.name
        }

    def _set_physical_target(self, direction):
        """Triggers the state machine to start moving."""
        needed = self.params.stroke if direction == 'REVERSE' else 0.0
        # Only trigger if position mismatch exists
        if abs(self.mechanism.pos - needed) > 0.005:
            # State transition: LOCKED -> UNLOCKING
            if self.state in [SwitchState.LOCKED_NORMAL, SwitchState.LOCKED_REVERSE]:
                self.state = SwitchState.UNLOCKING
                self.target_pos = needed

    def update(self, dt, current_time):
        """
        [Main Loop] Executed every simulation step.
        Couples Physics Simulation with Logic Updates.
        """
        # 1. Fault Handling
        if self.state == SwitchState.STALLED:
            return  # Failure State

        # 2. Semantic Maintenance (Prune old beliefs periodically)
        # Done implicitly in handle_semantic_packet, but can be forced here
        # self.estimator._prune_stale_beliefs(current_time)

        # 3. Adaptive Control Law (Voltage Regulation)
        voltage_cmd = 0.0

        if self.state in [SwitchState.UNLOCKING, SwitchState.MOVING, SwitchState.LOCKING]:
            # P-Control for position
            err = self.target_pos - self.mechanism.pos

            # Base Voltage
            u_base = 24.0

            # Adaptive Boosting: If stuck in mud (High Current, Low Velocity)
            if abs(self.motor.current) > 10.0 and abs(self.mechanism.omega) < 5.0:
                u_base = 48.0  # Boost to overcome stiction
                # Boosting degrades health
                self.health_index -= 0.0001 * dt

            voltage_cmd = u_base * np.sign(err)

            # Soft Landing (prevent mechanical shock)
            if abs(err) < 0.01:
                voltage_cmd *= 0.5

        # 4. Physical Step (Coupled Simulation)
        # External disturbance injection (Random stone jam probability)
        ext_force = 0.0
        if np.random.random() < 0.00001 * self.mud:
            ext_force = 5000.0  # Jamming force

        torque = self.motor.step_rk4(dt, voltage_cmd, self.mechanism.omega)
        pos, vel = self.mechanism.step(dt, torque, ext_force)

        # 5. State Machine Transition Logic
        self._update_fsm(pos, vel, self.motor.current, dt)

    def _update_fsm(self, pos, vel, current, dt):
        """Finite State Machine logic for phase transitions."""
        if self.state == SwitchState.UNLOCKING:
            # Transition to MOVING after overcoming initial locking zone
            if (self.target_pos > 0.1 and pos > 0.01) or \
                    (self.target_pos < 0.1 and pos < self.params.stroke - 0.01):
                self.state = SwitchState.MOVING

        elif self.state == SwitchState.MOVING:
            # Transition to LOCKING when near target
            dist = abs(pos - self.target_pos)
            if dist < 0.01:
                self.state = SwitchState.LOCKING

            # Stall Detection (PHM)
            if abs(current) > 20.0 and abs(vel) < 0.1:
                self.stall_counter += dt
                if self.stall_counter > 1.5:  # 1.5s stall limit
                    self.state = SwitchState.STALLED
                    logger.critical(f"Switch {self.id} STALLED due to overload!")
            else:
                self.stall_counter = 0.0

        elif self.state == SwitchState.LOCKING:
            # Transition to LOCKED when fully seated
            if abs(pos - self.target_pos) < 0.001:
                self.state = SwitchState.LOCKED_REVERSE if self.target_pos > 0.1 else SwitchState.LOCKED_NORMAL

    def get_telemetry(self):
        """Full system telemetry for analytics."""
        return {
            'id': self.id,
            'state': self.state.name,
            'risk': self.estimator.risk_level,  # Semantic metric
            'pos': self.mechanism.pos,  # Physical metric
            'current': self.motor.current,  # Electrical metric
            'temp': self.motor.temperature,  # Thermal metric
            'health': self.health_index
        }