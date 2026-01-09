import numpy as np
import logging
import random
import math
import networkx as nx
from enum import Enum, auto
from collections import deque
from dataclasses import dataclass

# 引入高保真物理内核
from physics import RailVehicleMBDSystem

# 配置日志
logger = logging.getLogger("Edge.Vehicle")


# =========================================================================
# [Layer 1] State & Mode Definitions
# =========================================================================

class ControlMode(Enum):
    """
    [Control Mode]
    PERFORMANCE:  High-speed, centralized time-space optimization (4D Planning).
    ROBUST:       Conservative speed, potential field navigation (Weak Net).
    HARDWARE_V2X: Distributed dynamic game theory (No Net / Ad-hoc).
    EMERGENCY:    Mechanical fallback / Safety stop.
    """
    PERFORMANCE = auto()
    ROBUST = auto()
    HARDWARE_V2X = auto()
    EMERGENCY = auto()


class VehicleState(Enum):
    """
    [FSM] Lifecycle states.
    """
    IDLE = auto()
    SEARCHING = auto()
    LOADING = auto()
    RETURNING = auto()
    UNLOADING = auto()
    FAULT_RECOVERY = auto()  # 故障自愈
    TRACTION_CONTROL = auto()  # 牵引控制
    BRAKING_NORMAL = auto()  # 常规制动
    WAITING_SWITCH = auto()  # 等待道岔


@dataclass
class EnergyAudit:
    traction_joules: float = 0.0
    compute_joules: float = 0.0
    comm_joules: float = 0.0

    @property
    def total_energy(self):
        return self.traction_joules + self.compute_joules + self.comm_joules


# =========================================================================
# [Layer 2] Cyber-Physical Agent Implementation
# =========================================================================

class VehicleAgent:
    def __init__(self, agent_id, vehicle_type_cfg, env_config, start_node, map_graph, infra_agents):
        self.id = agent_id
        self.cfg = vehicle_type_cfg
        self.env = env_config
        self.map = map_graph
        self.infra = infra_agents
        self.all_vehicles = []  # Global reference for V2V simulation

        # --- 1. Physics Engine Integration ---
        self.physics = RailVehicleMBDSystem(self.cfg, self.env)
        self.length = float(self.cfg.get('length', 12.0))
        self.physics.length = self.length

        # --- 2. Kinematics (2D & 3D) ---
        if start_node in self.map.nodes:
            self.pos_2d = np.array(self.map.nodes[start_node].pos, dtype=float)
            self.physics._init_position(spacing=2.0)
        else:
            self.pos_2d = np.array([0.0, 0.0])

        self.pos_3d = np.array([self.pos_2d[0], self.pos_2d[1], 0.0])
        self.orientation = np.array([0.0, 0.0, 0.0])
        self.health_status = 1.0
        self.vibration_level = 0.0
        self.recovery_timer = 0.0

        # --- 3. Cognitive State & Navigation ---
        self.state = VehicleState.IDLE
        self.mode = ControlMode.HARDWARE_V2X  # Default to decentralized
        self.home_node = start_node
        self.target_node = self._hash_food_target()

        self.current_node_id = start_node
        self.next_node_id = None
        self.path_queue = deque()
        self.switch_triggered = False  # Interlocking flag

        # --- 4. Perception & Networking ---
        self.network_uncertainty = 0.0
        self.last_broadcast_pos = self.pos_2d.copy()
        self.last_broadcast_ts = -100.0

        # V2X Lists
        self.v2v_neighbors = []  # All vehicles in comms range (Raw Hardware Data)
        self.rail_obstacles = []  # Vehicles physically blocking my track (Topology Logic)
        self.cached_node_potential = {}
        self.current_rssi = -120.0  # 默认底噪

        # --- 5. Control Internal State ---
        self.mpc_prev_u = 0.0
        self.game_weight = 0.0  # Priority weight for dynamic game

        # --- 6. Telemetry ---
        self.energy = EnergyAudit()
        self.current_speed = 0.0
        self.dist_accumulated = 0.0
        self.time_active = 0.0
        self.wait_timer = 0.0
        self.last_telemetry = {}

    def step(self, dt, global_time):
        """
        [Main Loop] Frequency: 1/dt Hz
        Phase 1: Perception & Assessment
        Phase 2: Planning (Time-Space or Game Theory)
        Phase 3: Control & Actuation
        """
        # -----------------------------------------------------------------
        # 1. Perception Layer
        # -----------------------------------------------------------------
        # A. 硬件扫描 (模拟雷达/DSRC 物理接收)
        self._perform_hardware_v2x_scan(global_time)

        # B. 轨道拓扑过滤 (从硬件邻居中筛选出轨道上的障碍)
        self._perform_rail_topology_scan()

        # C. 网络环境评估
        self._assess_network_condition(global_time)

        # -----------------------------------------------------------------
        # 2. Planning Layer (Decision Making)
        # -----------------------------------------------------------------
        # 随机故障注入 (Reliability Testing)
        if self.state != VehicleState.FAULT_RECOVERY and np.random.random() < 0.00001:
            logger.warning(f"Vehicle {self.id} experienced POWERTRAIN_FAULT!")
            self.state = VehicleState.FAULT_RECOVERY
            self.recovery_timer = 5.0

        # 获取基于物理环境的动态限速 (EAVP)
        v_limit_ref = self._get_speed_limit()

        # 任务逻辑流转
        self._update_mission_logic(dt)

        # -----------------------------------------------------------------
        # 3. Control Layer (Execution)
        # -----------------------------------------------------------------
        u_cmd = 0.0

        # === Case A: Fault Recovery ===
        if self.state == VehicleState.FAULT_RECOVERY:
            u_cmd = 0.0
            self.recovery_timer -= dt
            if self.recovery_timer <= 0:
                logger.info(f"Vehicle {self.id} recovered from fault.")
                self.state = VehicleState.SEARCHING
                self.health_status = 0.9

        # === Case B: Moving State ===
        elif self.state in [VehicleState.SEARCHING, VehicleState.RETURNING, VehicleState.TRACTION_CONTROL]:
            # Step 1: 路径规划 (Path Planning)
            # 如果没有路径，根据网络模式选择规划策略
            if not self.path_queue and not self.next_node_id:
                if self.mode == ControlMode.PERFORMANCE:
                    # 有网: 时空 A* 规划 (4D Trajectory)
                    self._plan_spacetime_optimal(global_time)
                else:
                    # 无网: 静态拓扑规划 (离线地图兜底)
                    self._plan_local_path_static()

            # 填充下一跳
            if self.path_queue and not self.next_node_id:
                self.next_node_id = self.path_queue[0]

            # Step 2: 速度协商 (Speed Negotiation)
            target_v = v_limit_ref

            if self.mode == ControlMode.HARDWARE_V2X:
                # 无网: 分布式动态博弈 (V2V Negotiation)
                target_v = self._negotiate_v2v_game(v_limit_ref)
            elif self.mode == ControlMode.PERFORMANCE:
                # 有网: 动态冲突响应 (Dynamic Conflict Response)
                target_v = self._handle_dynamic_conflict(v_limit_ref)

            # Step 3: 道岔与运动控制 (Switch & Motion)
            if self.next_node_id:
                # 触发道岔动作 (Interlocking)
                self._handle_switch_control(global_time)

                # 执行 MPC 鲁棒追踪
                u_cmd = self._mpc_control(dt, target_v)

        # === Case C: Waiting for Switch ===
        elif self.state == VehicleState.WAITING_SWITCH:
            u_cmd = self._mpc_control(dt, 0.0)  # 保持停车
            if self._check_switch_status():
                self.state = VehicleState.SEARCHING  # 道岔到位，恢复行驶

        # === Case D: Normal Braking ===
        elif self.state == VehicleState.BRAKING_NORMAL:
            if abs(self.current_speed) > 0.1:
                u_cmd = -48.0 if self.current_speed > 0 else 48.0
            else:
                u_cmd = 0.0

        # -----------------------------------------------------------------
        # 4. Actuation & Feedback Layer
        # -----------------------------------------------------------------
        # 物理引擎解算
        dynamics = self.physics.step_rk4(dt, u_cmd)
        self.current_speed = dynamics['loco_vel']

        # 状态同步
        self._sync_kinematics(dt)
        self._update_kinematics_3d(dt)

        # 通信与能耗
        self._try_broadcast_semantic(global_time)

        p_inst = abs(u_cmd * dynamics['motor_current'])
        self.energy.traction_joules += p_inst * dt
        if abs(self.current_speed) > 0.01:
            self.dist_accumulated += abs(self.current_speed * dt)
            self.time_active += dt

        self.vibration_level = abs(self.current_speed) * 0.5 * np.random.normal(1, 0.1)

        # 遥测打包
        self.last_telemetry = {
            'id': self.id, 'state': self.state.name, 'mode': self.mode.name,
            'pos': self.pos_2d, 'vel': self.current_speed,
            'force': dynamics['coupler_force_1'], 'current': dynamics['motor_current'],
            'mu': dynamics.get('mu_effective', 0.0),
            'energy': self.energy.total_energy, 'mass_total': self.physics.mass_total,
            'z': self.pos_3d[2], 'vib': self.vibration_level,
            'potential': 0.0
        }
        return self.last_telemetry

    # =========================================================================
    # [Module 1] Perception: Hardware V2X & Rail Topology
    # =========================================================================

    def _perform_hardware_v2x_scan(self, now):
        """
        [Hardware Layer] Simulate DSRC/C-V2X radio receiving beacons.
        Populates self.v2v_neighbors with all 'audible' vehicles within range.
        """
        self.v2v_neighbors = []
        scan_radius = 200.0  # DSRC typical range
        max_rssi = -120.0 + np.random.normal(0, 1.0)

        for v in self.all_vehicles:
            if v.id == self.id: continue

            dist = np.linalg.norm(self.pos_2d - v.pos_2d)
            if dist < scan_radius:
                # 剔除严重故障车辆 (不可信节点)
                if v.health_status < 0.3: continue
                self.v2v_neighbors.append((dist, v))

        # 同时读取路侧单元 (RFID/Balise) 缓存势能
        curr_infra = self.infra.get(self.current_node_id)
        if curr_infra:
            state = curr_infra.get_broadcast_state()
            self.cached_node_potential[self.current_node_id] = (state.get('potential', 0.0), now)

    def _perform_rail_topology_scan(self):
        """
        [Topology Layer] Filter neighbors to find those actually blocking my rail path.
        Replaces simple radius scan with logic-based obstacle detection.
        """
        self.rail_obstacles = []
        if not self.next_node_id: return

        my_pos = self.pos_2d
        target_pos = np.array(self.map.nodes[self.next_node_id].pos)

        # 计算当前轨道段的方向矢量
        path_vec = target_pos - my_pos
        path_len = np.linalg.norm(path_vec)
        if path_len < 0.1: return
        path_dir = path_vec / path_len

        for dist, v in self.v2v_neighbors:
            rel_vec = v.pos_2d - my_pos

            # 1. 投影检查：是否在前方？
            proj = np.dot(rel_vec, path_dir)
            if proj > 0 and proj < path_len + 10.0:
                # 2. 垂直距离检查：是否在轨道宽度内？
                perp_dist = np.linalg.norm(rel_vec - proj * path_dir)
                if perp_dist < 2.5:  # 轨道走廊宽度
                    self.rail_obstacles.append((dist, v))

    # =========================================================================
    # [Module 2] Planning: Time-Space & Static
    # =========================================================================

    def _plan_spacetime_optimal(self, start_time):
        """
        [Advanced] Time-Space A* Planning.
        Finds a path minimizing cost in (Node, Time) space to avoid conflicts.
        """
        try:
            # 简化的 TS-A* 实现
            # 实际部署应查询 infrastructure.query_time_space
            path = nx.shortest_path(self.map.graph, self.current_node_id, self.target_node)

            # 尝试预约沿途资源
            curr_t = start_time
            valid_path = True

            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge_len = 300.0  # Approximate
                duration = edge_len / 10.0  # Est speed 10m/s

                infra = self.infra.get(v)
                if infra:
                    # [关键修复] 如果时间窗冲突，尝试推迟进入时间 (Wait logic)
                    wait_time = 0.0
                    max_wait = 60.0

                    # 循环检测：直到找到空闲窗口
                    while infra.query_time_space(curr_t + wait_time, duration):
                        wait_time += 2.0  # 每次推迟2秒
                        if wait_time > max_wait:
                            break  # 超时，只能硬着头皮上了(依靠MPC避障)

                    final_start_t = curr_t + wait_time
                    infra.reserve_time_space(self.id, final_start_t, duration)

                    # 更新当前规划时间
                    curr_t = final_start_t + duration
                else:
                    curr_t += duration  # 无基站，直接累加时间

            if path and path[0] == self.current_node_id: path.pop(0)
            self.path_queue = deque(path)
            logger.info(f"[{self.id}] TS-A* Path Planned: {len(path)} hops")

        except Exception as e:
            logger.warning(f"TS-A* Failed: {e}, fallback to static.")
            self._plan_local_path_static()

    def _plan_local_path_static(self):
        """[Fallback] Static Dijkstra/BFS Planning."""
        try:
            path = nx.shortest_path(self.map.graph, self.current_node_id, self.target_node)
            if path and path[0] == self.current_node_id: path.pop(0)
            self.path_queue = deque(path)
        except:
            self.path_queue = deque()

    # =========================================================================
    # [Module 3] Decision: Dynamic Game & Conflict Handling
    # =========================================================================

    def _negotiate_v2v_game(self, desired_v):
        """
        [No-Net Innovation] Distributed Dynamic Game for Right-of-Way.
        Calculates priority weight and negotiates speed with neighbors.
        """
        if not self.next_node_id: return desired_v

        # 1. 计算自身博弈权重 (Weight Function)
        # W = alpha*Speed + beta*Mass + gamma/Distance
        dist_to_next = self._dist_to(self.next_node_id)
        mass = self.physics.mass_total
        self.game_weight = 0.5 * abs(self.current_speed) + 0.001 * mass + 100.0 / (dist_to_next + 1.0)

        safe_v = desired_v

        # 2. 与邻居博弈
        for d, neighbor in self.v2v_neighbors:
            # 仅与争夺同一目标节点的邻居博弈
            if neighbor.next_node_id == self.next_node_id:
                # 获取对方权重 (模拟 V2V 数据包解析)
                n_weight = getattr(neighbor, 'game_weight', 0.0)

                if n_weight > self.game_weight:
                    # 我输了 (Yield)
                    # 计算精确减速曲线，在路口前 safety_gap 处将速度降至微速
                    gap = 20.0
                    if dist_to_next > gap:
                        # 平滑减速，保持流动性 (Rolling Stop)
                        safe_v = min(safe_v, 2.0)  # 降级为蠕行
                    else:
                        safe_v = 0.0  # 必须停车让行

        return safe_v

    def _handle_dynamic_conflict(self, v_ref):
        """
        [Networked] React to dynamic uncertainties (e.g., front car slowing down).
        """
        # 检查轨道前车距离
        if self.rail_obstacles:
            nearest_d, nearest_v = min(self.rail_obstacles, key=lambda x: x[0])

            # ACC 跟驰逻辑
            safe_gap = 25.0
            if nearest_d < safe_gap * 2:
                # P-Control maintain gap
                err = nearest_d - safe_gap
                target_v = max(0.0, nearest_v.current_speed + 0.5 * err)
                return min(v_ref, target_v)

        return v_ref

    # =========================================================================
    # [Module 4] Control: MPC & Switching
    # =========================================================================

    def _mpc_control(self, dt, v_limit_ref):
        """
        [Control Core] Physics-Aware Explicit MPC.
        Minimizes J = (v - v_ref)^2 + lambda * du^2
        Includes Potential Well Braking for obstacles.
        """
        # 1. System Identification (First-order Inertia)
        mass = max(100.0, self.physics.mass_total)
        B = (200.0 / mass) * dt

        # 2. Refined Target Calculation (Potential Well)
        v_ref_final = v_limit_ref
        target_id = self.next_node_id if self.next_node_id else (self.path_queue[0] if self.path_queue else None)

        if target_id:
            dist = self._dist_to(target_id)

            # 障碍物距离
            obs_dist = float('inf')
            if self.rail_obstacles:
                obs_dist, _ = min(self.rail_obstacles, key=lambda x: x[0])

            # 道岔状态
            switch_ready = True
            if dist < 40.0 and not self._check_switch_status():
                switch_ready = False

            # 速度规划融合
            if not switch_ready:
                # 道岔未好，目标设为路口前 5m 停车
                v_ref_final = self._calc_smooth_approach_v(dist - 5.0, v_limit_ref)
            elif obs_dist < 60.0:
                # 前车避让
                v_ref_final = self._calc_smooth_approach_v(obs_dist - 20.0, v_limit_ref)
            elif dist < 20.0:
                # 进站自然减速
                v_ref_final = self._calc_smooth_approach_v(dist, v_limit_ref)

            # 终点吸附
            if dist < 1.0:
                self.current_node_id = target_id
                self.next_node_id = None
                self.switch_triggered = False
                if self.path_queue and self.path_queue[0] == target_id:
                    self.path_queue.popleft()
                v_ref_final = 0.0

        # 3. Optimization (Analytical)
        lam = 0.1  # Smoothing factor
        u_opt = (B * (v_ref_final - self.current_speed) + lam * self.mpc_prev_u) / (B ** 2 + lam)

        # 4. Saturation
        u_opt = np.clip(u_opt, -48.0, 48.0)
        self.mpc_prev_u = u_opt
        return u_opt

    def _calc_smooth_approach_v(self, dist, v_max):
        """Physics-based braking curve: v = sqrt(2*a*d)."""
        if dist <= 0: return 0.0

        # 估算当前环境下的最大可用减速度
        mud = self.env.get('mud_factor', 0.5)
        mu_est = 0.4 * (1.0 - 0.5 * mud)
        a_brake = mu_est * 9.81 * 0.8  # 留 20% 安全余量

        v_brake = math.sqrt(2 * a_brake * dist)
        return min(v_max, v_brake)

    def _handle_switch_control(self, now):
        """[Interlocking] Active Switch Triggering based on Vector."""
        target_id = self.next_node_id
        dist = self._dist_to(target_id)
        TRIGGER_DIST = 50.0

        if dist < TRIGGER_DIST and not self.switch_triggered:
            infra = self.infra.get(target_id)
            if infra:
                # 计算入站矢量，决定道岔方向
                curr_p = self.map.nodes[self.current_node_id].pos
                next_p = self.map.nodes[target_id].pos
                vec_in = np.array(next_p) - np.array(curr_p)

                # 网格逻辑：水平直行(NORMAL)，垂直转弯(REVERSE)
                req_state = "NORMAL"
                if abs(vec_in[1]) > abs(vec_in[0]):
                    req_state = "REVERSE"

                # 发送硬件指令
                infra.handle_hardware_signal({
                    'vid': self.id,
                    'type': 'SWITCH_REQ',
                    'target_state': req_state
                })
                self.switch_triggered = True

        if dist > TRIGGER_DIST + 10.0:
            self.switch_triggered = False

    def _check_switch_status(self):
        """Verify if switch is locked in position."""
        infra = self.infra.get(self.next_node_id)
        if not infra: return True
        state = infra.get_broadcast_state()
        # 只要不在移动或解锁中，即视为安全锁定
        return state['state'] not in ["MOVING", "UNLOCKING"]

    def _get_speed_limit(self):
        """[EAVP] Environment-Adaptive Velocity Profiling."""
        base_v = 300.0 / self.length

        # 环境衰减
        mud = self.env.get('mud_factor', 0.5)
        env_factor = 1.0 / (1.0 + 1.5 * mud)
        uncert_factor = 1.0 / (1.0 + 0.2 * self.network_uncertainty)

        v_physics = base_v * env_factor * uncert_factor

        # 模式约束
        if self.state == VehicleState.SEARCHING:
            return v_physics * 0.5

        if self.mode == ControlMode.ROBUST:
            return v_physics * 0.8
        elif self.mode == ControlMode.HARDWARE_V2X:
            return v_physics * 0.7
        elif self.mode == ControlMode.EMERGENCY:
            return v_physics * 0.1

        return v_physics

    # =========================================================================
    # [Module 5] Support Functions
    # =========================================================================

    def _assess_network_condition(self, now):
        if "Start" in str(self.current_node_id):
            aoi = 0.0
        else:
            curr_infra = self.infra.get(self.current_node_id)
            aoi = 0.1 if curr_infra else 15.0  # Large AoI if no infra

        self.network_uncertainty = 0.5 + 0.2 * aoi

        # Mode Switching Logic
        if aoi < 1.0:
            self.mode = ControlMode.PERFORMANCE
        elif aoi < 10.0:
            self.mode = ControlMode.ROBUST
        else:
            self.mode = ControlMode.HARDWARE_V2X

    def _update_mission_logic(self, dt):
        """
        [Mission Loop]
        Revised Logic: IDLE(Home) -> LOADING(Home) -> SEARCHING(To Target) -> UNLOADING(Target) -> RETURNING(Home)
        Includes dynamic loading times based on wagon count.
        """
        # 获取车厢数量 (如果 Physics 中未定义，默认 1)
        wagons = getattr(self.physics, 'num_wagons', 1)
        # 动态装卸时间：基础 3秒 + 每车厢 2秒
        op_time = 3.0 + wagons * 2.0

        if self.state == VehicleState.IDLE:
            self.wait_timer -= dt
            if self.wait_timer <= 0:
                # 休息结束，开始装货 (Loading at Home)
                self.state = VehicleState.LOADING
                self.wait_timer = op_time
                logger.info(f"[{self.id}] State: IDLE -> LOADING (Duration: {op_time:.1f}s)")

        elif self.state == VehicleState.LOADING:
            self.wait_timer -= dt
            if self.wait_timer <= 0:
                # 装货完成，出发送货 (Search/Deliver)
                self.state = VehicleState.SEARCHING
                # 确保有目标
                if not self.target_node or self.target_node == self.home_node:
                    self.target_node = self._hash_food_target()

                # 触发寻路
                if self.mode == ControlMode.PERFORMANCE:
                    self._plan_spacetime_optimal(0)
                else:
                    self._plan_local_path_static()

                logger.info(f"[{self.id}] State: LOADING -> SEARCHING (Target: {self.target_node})")

        elif self.state == VehicleState.SEARCHING:
            if self.current_node_id == self.target_node:
                # 到达终点，开始卸货
                self.state = VehicleState.UNLOADING
                self.wait_timer = op_time
                logger.info(f"[{self.id}] State: SEARCHING -> UNLOADING")

        elif self.state == VehicleState.UNLOADING:
            self.wait_timer -= dt
            if self.wait_timer <= 0:
                # 卸货完成，返程回家
                self.state = VehicleState.RETURNING
                self.target_node = self.home_node
                self.next_node_id = None
                self.path_queue.clear()

                if self.mode == ControlMode.PERFORMANCE:
                    self._plan_spacetime_optimal(0)
                else:
                    self._plan_local_path_static()

                logger.info(f"[{self.id}] State: UNLOADING -> RETURNING (Home: {self.home_node})")

        elif self.state == VehicleState.RETURNING:
            if self.current_node_id == self.target_node:
                # 到家，休息
                self.state = VehicleState.IDLE
                self.wait_timer = 5.0  # Rest time
                # 计算下一轮的目标
                self.target_node = self._hash_food_target()
                logger.info(f"[{self.id}] State: RETURNING -> IDLE")

    def _sync_kinematics(self, dt):
        tid = self.next_node_id if self.next_node_id else (self.path_queue[0] if self.path_queue else None)
        if tid:
            t_pos = np.array(self.map.nodes[tid].pos)
            vec = t_pos - self.pos_2d
            dist = np.linalg.norm(vec)
            if dist > 1e-4:
                step = self.current_speed * dt
                # 防止超调
                if step > dist:
                    self.pos_2d = t_pos
                else:
                    self.pos_2d += (vec / dist) * step

    def _update_kinematics_3d(self, dt):
        self.pos_3d[:2] = self.pos_2d
        if hasattr(self.map, 'get_terrain_height'):
            self.pos_3d[2] = self.map.get_terrain_height(self.pos_3d[0], self.pos_3d[1])

    def _try_broadcast_semantic(self, now):
        # 简单的事件触发广播
        err = np.linalg.norm(self.pos_2d - self.last_broadcast_pos[:2])
        if err > 1.0 or (now - self.last_broadcast_ts) > 5.0:
            curr_infra = self.infra.get(self.current_node_id)
            if curr_infra:
                # [Fix] Added 'eta' and 'duration' to payload to prevent KeyError in Infrastructure
                packet = {
                    'vid': self.id,
                    'pos': self.pos_2d,
                    'vel': self.current_speed,
                    'timestamp': now,
                    'pos_uncertainty': self.network_uncertainty,
                    'eta': now,
                    'duration': 2.0,
                    'direction': 'NORMAL'
                }
                curr_infra.handle_semantic_packet(packet)
            self.last_broadcast_pos = self.pos_3d.copy()
            self.last_broadcast_ts = now
            self.energy.comm_joules += 0.01

    def _hash_food_target(self):
        h = hash(self.id)
        rows, cols = self.map.rows, self.map.cols
        return f"N_{h % rows}_{(cols // 2) + (h % (cols // 2))}"

    def _dist_to(self, node_id):
        if node_id not in self.map.nodes: return 0.0
        return np.linalg.norm(self.pos_2d - np.array(self.map.nodes[node_id].pos))

    # 占位函数 (兼容性)
    def _resolve_next_hop_gradient(self):
        pass

    def _resolve_next_hop_distributed(self):
        pass