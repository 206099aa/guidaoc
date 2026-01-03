import numpy as np
import logging
import random
import math
from enum import Enum, auto
from collections import deque
from dataclasses import dataclass

# 引入高保真物理内核
from physics import RailVehicleMBDSystem

# 配置日志
logger = logging.getLogger("Edge.Vehicle")


# =========================================================================
# [Layer 1] State & Mode Definitions (状态与模式定义)
# -------------------------------------------------------------------------
# 定义智能体的生命周期状态与基于网络质量的控制模式。
# =========================================================================

class ControlMode(Enum):
    """
    [Adaptive Control] Operational modes based on Network Quality (AoI).
    """
    PERFORMANCE = auto()  # Strong Net: High speed, tight spacing (MPC enabled)
    ROBUST = auto()  # Weak Net: Reduced speed, larger gaps (Conservative P-Control)
    EMERGENCY = auto()  # No Net: Crawl or Stop (Safety Critical)


class VehicleState(Enum):
    """
    [FSM] Lifecycle states of the Snake Agent.
    """
    IDLE = auto()  # Resting at Depot
    SEARCHING = auto()  # Outbound: High precision, Low speed
    LOADING = auto()  # At Food Source
    RETURNING = auto()  # Inbound: High speed cruising
    UNLOADING = auto()  # At Depot
    FAULT_RECOVERY = auto()  # PHM Triggered


@dataclass
class EnergyAudit:
    """[SCI Metric] Fine-grained Energy Consumption Tracking."""
    traction_joules: float = 0.0  # Mechanical work
    compute_joules: float = 0.0  # Edge computing cost
    comm_joules: float = 0.0  # RF transmission cost

    @property
    def total_energy(self):
        return self.traction_joules + self.compute_joules + self.comm_joules


# =========================================================================
# [Layer 2] Cyber-Physical Agent (物理-信息融合智能体)
# -------------------------------------------------------------------------
# 核心类：集成物理引擎、事件触发通信与鲁棒控制律。
# =========================================================================

class VehicleAgent:
    """
    [Agent Implementation]
    A Cyber-Physical System (CPS) agent that adapts its control strategy
    based on environmental uncertainty (Mud) and network reliability (AoI).
    """

    def __init__(self, agent_id, vehicle_type_cfg, env_config, start_node, map_graph, infra_agents):
        self.id = agent_id
        self.cfg = vehicle_type_cfg
        self.env = env_config
        self.map = map_graph
        self.infra = infra_agents  # Direct access to local infrastructure (V2I)

        # --- 1. High-Fidelity Physics Integration (DeepSnake Advantage) ---
        # Instantiates the MBD system (Mass-Spring-Damper + Motor Dynamics)
        self.physics = RailVehicleMBDSystem(self.cfg, self.env)
        self.length = float(self.cfg.get('length', 12.0))
        # Sync physics engine parameters
        self.physics.length = self.length

        # Kinematic State (2D Map Projection)
        if start_node in self.map.nodes:
            self.pos_2d = np.array(self.map.nodes[start_node].pos, dtype=float)
            self.physics._init_position(spacing=2.0)
        else:
            self.pos_2d = np.array([0.0, 0.0])

        # [新增功能 Start] 3D 状态与 PHM 指标初始化
        self.pos_3d = np.array([self.pos_2d[0], self.pos_2d[1], 0.0])  # x, y, z
        self.orientation = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw
        self.health_status = 1.0
        self.vibration_level = 0.0
        # [新增功能 End]

        # --- 2. Cognitive State Management ---
        self.state = VehicleState.IDLE
        self.mode = ControlMode.PERFORMANCE
        self.home_node = start_node
        self.target_node = self._hash_food_target()

        # Navigation Stack
        self.current_node_id = start_node
        self.next_node_id = None
        self.path_queue = deque()

        # --- 3. Robust Networking (RobustSnake Advantage) ---
        # AoI & Uncertainty State
        self.network_uncertainty = 0.0  # Sigma (Position variance)
        self.last_comm_ts = -100.0  # Last successful handshake

        # Event-Triggered Comms State
        self.last_broadcast_ts = -100.0
        self.last_broadcast_pos = self.pos_2d.copy()

        # --- 4. Telemetry & Auditing ---
        self.energy = EnergyAudit()
        self.current_speed = 0.0
        self.dist_accumulated = 0.0
        self.time_active = 0.0
        self.wait_timer = 0.0
        self.last_telemetry = {}

    def step(self, dt, global_time):
        """
        [Main Control Loop] Frequency: 1/dt Hz
        Execution Order: Sense -> Assess -> Plan -> Control -> Actuate -> Communicate
        """
        # 1. Network Assessment (AoI Logic)
        # Determine reliability of the environment
        self._assess_network_condition(global_time)

        # 2. State Machine & High-Level Planning
        # Determine v_limit and logical transitions
        v_limit_ref = self._update_fsm(dt, global_time)

        # 3. Robust Tracking Control (MPC + Semantic Reservation)
        # Calculate voltage command for physics engine
        u_cmd = 0.0
        if self.state in [VehicleState.SEARCHING, VehicleState.RETURNING]:
            if not self.next_node_id:
                # [新增功能 Start] 优先尝试势能场梯度导航 (Shared Location)
                self._resolve_next_hop_gradient()
                # [新增功能 End]

                # [修改说明] 如果势能场不可用（例如无信号），回退到分布式表
                if not self.next_node_id:
                    self._resolve_next_hop_distributed()

            if self.next_node_id:
                u_cmd = self._robust_tracking_control(dt, v_limit_ref, global_time)

        # 4. Physical Actuation (RK4 Integration)
        # Apply voltage, simulate motor & mechanics
        dynamics = self.physics.step_rk4(dt, u_cmd)

        # 5. State Synchronization (Physics -> Kinematics)
        # Correct 2D map position using high-fidelity physics output
        self.current_speed = dynamics['loco_vel']
        self._sync_kinematics(dt)

        # [新增功能 Start] 同步 3D 状态 (PHM & 3D Sim)
        self._update_kinematics_3d(dt)
        # [新增功能 End]

        # 6. Event-Triggered Communication
        # Broadcast only if necessary
        self._try_broadcast_semantic(global_time)

        # [新增功能 Start] 位置共享广播 (Holographic Location)
        self._share_location_holographic(global_time)
        # [新增功能 End]

        # 7. Energy Auditing
        p_inst = abs(u_cmd * dynamics['motor_current'])  # Electrical Power
        self.energy.traction_joules += p_inst * dt
        if abs(self.current_speed) > 0.01:
            self.dist_accumulated += abs(self.current_speed * dt)
            self.time_active += dt

        # [新增功能 Start] 模拟 PHM 振动数据
        self.vibration_level = abs(self.current_speed) * self.env['mud_factor'] * np.random.normal(1, 0.1)
        # [新增功能 End]

        # 8. Telemetry Packaging
        self.last_telemetry = {
            'id': self.id,
            'state': self.state.name,
            'mode': self.mode.name,
            'pos': self.pos_2d,
            'vel': self.current_speed,
            'force': dynamics['coupler_force_1'],
            'current': dynamics['motor_current'],
            'energy': self.energy.total_energy,
            'uncert': self.network_uncertainty,
            'target': self.target_node,
            # [新增功能] 3D & PHM 数据
            'z': self.pos_3d[2],
            'vib': self.vibration_level,
            'potential': 0.0  # 占位，实际可从 infra 获取
        }
        return self.last_telemetry

    # =========================================================================
    # [Module 1] Network Awareness & Adaptation (网络感知与自适应)
    # -------------------------------------------------------------------------
    # 实现 "可取之处 1": 信息新鲜度感知与控制模式动态切换
    # =========================================================================

    def _assess_network_condition(self, now):
        """
        Assess local infrastructure availability to estimate AoI.
        Adjusts 'network_uncertainty' and switches 'ControlMode'.
        """
        # Check connectivity to current node's edge agent
        curr_infra = self.infra.get(self.current_node_id)

        if curr_infra:
            # Simulate AoI: Low if infra is present, High if dead zone
            # In a real system, this would check 'last_seen' timestamp of beacons
            aoi = 0.1
        else:
            aoi = 5.0  # Weak signal area assumption

        # Uncertainty grows linearly with AoI
        # Sigma = Base_Error + Drift_Rate * AoI
        self.network_uncertainty = 0.5 + 0.2 * aoi

        # Mode Switching Logic
        if aoi < 1.0:
            self.mode = ControlMode.PERFORMANCE
        elif aoi < 10.0:
            self.mode = ControlMode.ROBUST
        else:
            self.mode = ControlMode.EMERGENCY

    def _get_speed_limit(self):
        """
        Dynamic speed limit based on Physics (Length) and Control Mode.
        """
        # 1. Physical Constraint: V ~ 1/L (Snake Physics)
        # Longer vehicles must move slower to clear junctions safely
        base_v = 300.0 / self.length

        # 2. Task Constraint
        if self.state == VehicleState.SEARCHING:
            base_v *= 0.4  # Precision mode

        # 3. Network Constraint (Mode degradation)
        if self.mode == ControlMode.ROBUST:
            return base_v * 0.6  # Conservative speed
        elif self.mode == ControlMode.EMERGENCY:
            return base_v * 0.1  # Crawl speed

        return base_v

    # =========================================================================
    # [Module 2] Event-Triggered Communication (事件触发通信)
    # -------------------------------------------------------------------------
    # 实现 "可取之处 2": 非周期性广播，仅在误差超限时发送语义包
    # =========================================================================

    def _try_broadcast_semantic(self, now):
        """
        Check if state deviation exceeds threshold. If so, broadcast update.
        Reduces bandwidth usage in weak networks.
        """
        # Calculate deviation from last broadcasted belief
        pos_error = np.linalg.norm(self.pos_2d - self.last_broadcast_pos[:2])  # [修改] 适配 pos_3d

        # Adaptive Threshold: Allow larger error when uncertainty is already high
        threshold = 2.0 * max(1.0, self.network_uncertainty)

        # Time-out trigger (Heartbeat): Ensure liveliness at least every 5s
        time_since_last = now - self.last_broadcast_ts

        if pos_error > threshold or time_since_last > 5.0:
            # Broadcast Packet (Simulated)
            # In a real impl, this would call comms.send(packet)

            # Here we update the "Digital Twin" state locally to simulate successful TX
            # Note: The 'infrastructure' actually receives this via async calls if we wired it up.
            curr_infra = self.infra.get(self.current_node_id)
            if curr_infra:
                packet = {
                    'vid': self.id,
                    'pos': (self.pos_2d[0], self.pos_2d[1]),
                    'vel': self.current_speed,
                    'timestamp': now,
                    'pos_uncertainty': self.network_uncertainty
                }
                # Simulate "Fire-and-Forget" UDP transmission
                # The infrastructure's Bayesian Estimator will process this
                # curr_infra.handle_async_update(packet) # Requires method in infra

            # Update internal state
            self.last_broadcast_pos = self.pos_3d.copy()  # [修改] 适配 pos_3d
            self.last_broadcast_ts = now

            # Energy Cost
            self.energy.comm_joules += 0.01

    # [新增功能 Start] 基于势能场的全息位置共享广播
    def _share_location_holographic(self, now):
        """
        [Shared Location Method]
        Uploads presence to the local node to contribute to the global Potential Field.
        Efficient: Only sends when moving significantly (Event-Triggered).
        """
        err = np.linalg.norm(self.pos_3d - self.last_broadcast_pos)
        if err > 5.0 or (now - self.last_broadcast_ts) > 2.0:
            target_infra = self.infra.get(self.current_node_id)
            if target_infra:
                # Contribute mass to the field
                packet = {
                    'vid': self.id, 'eta': now, 'duration': 5.0,
                    'pos_uncertainty': self.network_uncertainty,
                    'timestamp': now
                }
                # "Fire-and-forget" update
                target_infra.handle_semantic_packet(packet)

            self.last_broadcast_pos = self.pos_3d.copy()
            self.last_broadcast_ts = now

    # [新增功能 End]

    # =========================================================================
    # [Module 3] Robust Control & Semantic Reservation (鲁棒控制与预约)
    # -------------------------------------------------------------------------
    # 实现 "可取之处 4": 分布式语义预约，结合 MPC 思想的轨迹跟踪
    # =========================================================================

    def _robust_tracking_control(self, dt, v_limit, now):
        """
        Calculates control input (Voltage) considering semantic reservations.
        """
        if not self.path_queue and not self.next_node_id:
            return 0.0

        target_id = self.next_node_id if self.next_node_id else self.path_queue[0]
        dist = self._dist_to(target_id)

        # 1. Semantic Reservation (Safety Barrier)
        # Before entering the intersection zone, check for probabilistic conflicts
        safety_gap = self.length + 15.0

        if dist < safety_gap:
            target_infra = self.infra.get(target_id)
            if target_infra:
                # Construct Semantic Packet
                # ETA = now + dist / current_speed
                # Duration = length / speed + buffer
                avg_v = max(1.0, self.current_speed)
                packet = {
                    'vid': self.id,
                    'eta': now + dist / avg_v,
                    'duration': (self.length / avg_v) + 3.0,
                    'pos_uncertainty': self.network_uncertainty,
                    'direction': 'NORMAL',  # Simplified
                    'global_time': now
                }

                # V2I Query (Non-blocking)
                response = target_infra.handle_semantic_packet(packet)

                # React to Risk
                if response['status'] == 'RISK_HIGH':
                    # High Collision Probability -> Emergency Brake
                    # Override Speed Limit
                    return -48.0  # Max Braking Voltage

        # 2. Tracking Controller (Simplified MPC/P-Control)
        # Target: Stop exactly at node if it's a waypoint, or pass through

        # Arrival check
        if dist < 1.0:
            self.current_node_id = target_id
            self.next_node_id = None  # Clear next hop
            if self.path_queue: self.path_queue.popleft()
            return 0.0

        # Velocity Error
        err_v = v_limit - self.current_speed

        # P-Control mapping to Voltage (-48V to +48V)
        # Kp = 200.0 based on ZD6 motor characteristics
        u_cmd = 200.0 * err_v
        return np.clip(u_cmd, -48.0, 48.0)

    # =========================================================================
    # [Module 4] Navigation & FSM (导航与状态机)
    # -------------------------------------------------------------------------
    # 状态流转与分布式下一跳解析
    # =========================================================================

    def _update_fsm(self, dt, now):
        """Update FSM and return target velocity limit."""
        v_target = 0.0

        if self.state == VehicleState.IDLE:
            self.wait_timer -= dt
            if self.wait_timer <= 0:
                self.state = VehicleState.SEARCHING
                self._plan_local_path()  # Initialize path

        elif self.state == VehicleState.SEARCHING:
            v_target = self._get_speed_limit()
            if self.current_node_id == self.target_node:
                self.state = VehicleState.LOADING
                self.wait_timer = 2.0

        elif self.state == VehicleState.LOADING:
            self.wait_timer -= dt
            if self.wait_timer <= 0:
                self.state = VehicleState.RETURNING
                self.target_node = self.home_node  # Return trip
                self.next_node_id = None  # Reset nav
                self._plan_local_path()

        elif self.state == VehicleState.RETURNING:
            v_target = self._get_speed_limit()
            if self.current_node_id == self.target_node:
                self.state = VehicleState.UNLOADING
                self.wait_timer = 3.0

        elif self.state == VehicleState.UNLOADING:
            self.wait_timer -= dt
            if self.wait_timer <= 0:
                self.state = VehicleState.IDLE
                self.target_node = self._hash_food_target()  # New mission
                self.wait_timer = 5.0

        return v_target

    def _resolve_next_hop_distributed(self):
        """
        Query Local/Distributed Router for Next Hop.
        Simulates accessing the distributed routing table on the edge node.
        """
        # Mock interaction with Router Protocol
        # In a full system, this would query 'self.router.get_local_guidance'
        # Fallback: Greedy Euclidean
        neighbors = list(self.map.graph.neighbors(self.current_node_id))
        if not neighbors: return

        p_t = np.array(self.map.nodes[self.target_node].pos)

        # Select neighbor minimizing distance to target
        best_n = min(neighbors, key=lambda n: np.linalg.norm(np.array(self.map.nodes[n].pos) - p_t))
        self.next_node_id = best_n

    # [新增功能 Start] 基于势能场梯度的下一跳解析
    def _resolve_next_hop_gradient(self):
        """
        [Navigation] Gradient Descent on Traffic Potential.
        Vehicles naturally flow away from high-potential (crowded) nodes.
        """
        neighbors = list(self.map.graph.neighbors(self.current_node_id))
        if not neighbors: return

        p_t = np.array(self.map.nodes[self.target_node].pos)

        best_n = None
        min_cost = float('inf')

        for n in neighbors:
            # 1. Distance Cost
            p_n = np.array(self.map.nodes[n].pos)
            dist_cost = np.linalg.norm(p_n[:2] - p_t[:2])

            # 2. Potential Cost (Shared Location Data)
            # Query the infra agent for its potential level
            potential_cost = 0.0
            n_infra = self.infra.get(n)
            if n_infra:
                # Access the public broadcast state
                state = n_infra.get_broadcast_state()
                potential_cost = state.get('potential', 0.0) * 100.0  # Weighting

            total_cost = dist_cost + potential_cost

            if total_cost < min_cost:
                min_cost = total_cost
                best_n = n

        self.next_node_id = best_n

    # [新增功能 End]

    # =========================================================================
    # [Module 5] Physics Sync & Utilities (物理同步与辅助)
    # -------------------------------------------------------------------------
    # 实现 "可取之处 3": 从物理引擎同步状态
    # =========================================================================

    def _sync_kinematics(self, dt):
        """
        Update 2D position based on Physics 1D velocity.
        Constrains movement to the track graph to prevent drift.
        """
        target_id = self.next_node_id if self.next_node_id else (self.path_queue[0] if self.path_queue else None)
        if not target_id: return

        target_pos = np.array(self.map.nodes[target_id].pos)
        vec = target_pos - self.pos_2d
        dist = np.linalg.norm(vec)

        if dist > 1e-4:
            # Project physics velocity onto 2D vector
            step = self.current_speed * dt

            # Anti-overshoot
            if step > dist:
                self.pos_2d = target_pos
            else:
                self.pos_2d += (vec / dist) * step

    # [新增功能 Start] 3D 状态更新
    def _update_kinematics_3d(self, dt):
        """Updates 3D position based on 1D track velocity."""
        if not self.next_node_id: return

        # Get target vector in 2D plane (Z is handled by terrain map later)
        t_pos_2d = np.array(self.map.nodes[self.next_node_id].pos)
        curr_2d = self.pos_3d[:2]
        vec = t_pos_2d - curr_2d
        dist = np.linalg.norm(vec)

        if dist > 1e-4:
            step = self.current_speed * dt
            # Simple Euler integration for pos
            move_vec = (vec / dist) * step
            self.pos_3d[0] += move_vec[0]
            self.pos_3d[1] += move_vec[1]
            # self.pos_3d[2] += 0.0 # Future: Add elevation change

    # [新增功能 End]

    def _plan_local_path(self):
        # Initial dummy path to start movement
        self.path_queue = deque()  # Cleared, will use _resolve_next_hop_distributed

    def _hash_food_target(self):
        # Deterministic random target
        h = hash(self.id)
        rows, cols = self.map.rows, self.map.cols
        return f"N_{h % rows}_{(cols // 2) + (h % (cols // 2))}"

    def _dist_to(self, node_id):
        if node_id not in self.map.nodes: return 0.0
        return np.linalg.norm(self.pos_2d - np.array(self.map.nodes[node_id].pos))