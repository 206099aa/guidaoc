import numpy as np
import logging
import random
import math
import networkx as nx  # [新增] 引入图算法库，用于本地路径规划兜底
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
    [Adaptive Control] Operational modes based on Network Quality & Flow Entropy.
    """
    # [修改] 替代原 PERFORMANCE。弱网下基于流场有序度的高效协同模式
    HOLO_COOP = auto()  # Low Turbulence (Ordered Flow) -> High speed, tight spacing

    ROBUST = auto()  # High Turbulence (Chaotic Flow) -> Reduced speed, larger gaps
    EMERGENCY = auto()  # No Signal -> Crawl or Stop (Safety Critical)


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
    TRACTION_CONTROL = auto()  # [修复] 兼容旧任务逻辑的状态
    BRAKING_NORMAL = auto()  # [修复] 兼容旧任务逻辑的状态


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
        self.mode = ControlMode.HOLO_COOP  # [修改] 默认全息协同
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

        # [新增] 局部流场湍流度感知
        self.local_turbulence = 0.0

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
        # 1. Network Assessment (Holographic Flow Logic)
        # Determine reliability using Flow Entropy, not just AoI
        self._assess_network_condition(global_time)

        # 2. State Machine & High-Level Planning
        # Determine v_limit and logical transitions
        v_limit_ref = self._update_fsm(dt, global_time)

        # 3. Robust Tracking Control (MPC + Semantic Reservation)
        # Calculate voltage command for physics engine
        u_cmd = 0.0

        # [关键修复] 将 TRACTION_CONTROL 加入活跃控制状态列表
        active_moving_states = [
            VehicleState.SEARCHING,
            VehicleState.RETURNING,
            VehicleState.TRACTION_CONTROL
        ]

        if self.state in active_moving_states:
            if not self.next_node_id:
                # [新增功能 Start] 优先尝试势能场梯度导航 (Shared Location)
                self._resolve_next_hop_gradient()
                # [新增功能 End]

                # [修改说明] 如果势能场不可用（例如无信号），回退到分布式表
                if not self.next_node_id:
                    self._resolve_next_hop_distributed()

            if self.next_node_id or self.path_queue:
                u_cmd = self._robust_tracking_control(dt, v_limit_ref, global_time)

        elif self.state == VehicleState.BRAKING_NORMAL:
            # 简单的停车阻尼控制
            if abs(self.current_speed) > 0.1:
                u_cmd = -48.0 if self.current_speed > 0 else 48.0
            else:
                u_cmd = 0.0

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
            # [关键修复] 添加质量数据，修复表格显示
            'mass_total': self.physics.mass_total,
            # [新增功能] 3D & PHM 数据
            'z': self.pos_3d[2],
            'vib': self.vibration_level,
            'potential': 0.0  # 占位，实际可从 infra 获取
        }
        return self.last_telemetry

    # =========================================================================
    # [Module 1] Network Awareness & Adaptation (网络感知与自适应)
    # -------------------------------------------------------------------------
    # 实现 "可取之处 1": 基于流场熵的模态切换 (替代原有的强网判断)
    # =========================================================================

    def _assess_network_condition(self, now):
        """
        [Novelty] Holographic Flow Assessment.
        Decides mode based on 'Turbulence' (Flow Entropy), not just AoI.
        """
        # [优化] 如果在 Start 节点 (车库)，认为是该区域有线连接，信号满格
        if "Start" in str(self.current_node_id):
            aoi = 0.0
            turbulence = 0.0
        else:
            # Check connectivity to current node's edge agent
            curr_infra = self.infra.get(self.current_node_id)
            if curr_infra:
                state = curr_infra.get_broadcast_state()
                aoi = 0.1
                # [新增] 获取流场湍流度
                turbulence = state.get('turbulence', 0.0)
            else:
                aoi = 5.0  # Weak signal area assumption
                turbulence = 1.0  # Unknown -> Assume Chaotic

        # Uncertainty grows linearly with AoI
        self.network_uncertainty = 0.5 + 0.2 * aoi
        self.local_turbulence = turbulence

        # [核心创新点] 模式切换逻辑重构
        # 原逻辑：AoI 低 -> Performance
        # 新逻辑：流场有序 (低湍流) -> Holo Coop (即使是弱网)

        if aoi > 10.0:  # 彻底无信号
            self.mode = ControlMode.EMERGENCY
        elif turbulence < 0.3:  # 流场有序，可以高效协同
            self.mode = ControlMode.HOLO_COOP
        else:  # 流场混乱，降级为鲁棒模式
            self.mode = ControlMode.ROBUST

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
                    'timestamp': now,
                    'vel': self.current_speed  # [新增] 上传速度用于计算流场
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
                    return -48.0  # Max Braking Voltage

        # 2. Tracking Controller (Simplified MPC/P-Control)
        # Target: Stop exactly at node if it's a waypoint, or pass through

        # Arrival check
        if dist < 1.0:
            self.current_node_id = target_id
            self.next_node_id = None  # Clear next hop
            if self.path_queue and self.path_queue[0] == target_id:
                self.path_queue.popleft()
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

        # [修复] 增加对 TRACTION_CONTROL 的支持 (用于固定任务)
        elif self.state == VehicleState.TRACTION_CONTROL:
            if not self.path_queue:
                self.state = VehicleState.BRAKING_NORMAL
            v_target = self._get_speed_limit()

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
        """
        [关键修复] 使用 NetworkX 进行初始路径规划 (Onboard Planning)。
        解决 SEARCHING 初始阶段 path_queue 为空导致不动的 Bug。
        """
        try:
            # 计算从当前点到目标点的最短路
            path = nx.shortest_path(self.map.graph, self.current_node_id, self.target_node)
            # 移除起始点（即当前点）
            if path and path[0] == self.current_node_id:
                path.pop(0)

            self.path_queue = deque(path)
            logger.info(f"Vehicle {self.id} initialized path: {list(self.path_queue)}")
        except Exception as e:
            # 规划失败（如目标不可达），清空队列，依赖分布式导航
            logger.warning(f"Local planning failed for {self.id}: {e}")
            self.path_queue = deque()

    def _plan_mission(self):
        # 简单的硬编码任务，用于测试
        # [修改] 使用正确的距离逻辑，避免直接寻路到远端导致的计算误差
        if "Hauler" in self.id:
            # 假设 Start_1 连接到 N_0_3 (根据地图生成逻辑估算)
            self.path_queue = deque(["N_0_3", "N_0_2", "Stop_H_0_2"])
        else:
            self.path_queue = deque(["N_2_0", "N_2_1", "Stop_H_2_1"])

    def _hash_food_target(self):
        # Deterministic random target
        h = hash(self.id)
        rows, cols = self.map.rows, self.map.cols
        return f"N_{h % rows}_{(cols // 2) + (h % (cols // 2))}"

    def _dist_to(self, node_id):
        if node_id not in self.map.nodes: return 0.0
        return np.linalg.norm(self.pos_2d - np.array(self.map.nodes[node_id].pos))