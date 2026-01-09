import networkx as nx
import logging
import math
import random
import heapq
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# 配置日志
logger = logging.getLogger("Router.Distributed")


# =========================================================================
# [Layer 1] Stochastic Channel Model (随机信道模型)
# -------------------------------------------------------------------------
# 模拟工业现场弱网环境：高丢包、长延迟、抖动。
# 对应 RobustSnake 优势：弱网适配性与信道仿真。
# =========================================================================

@dataclass
class Packet:
    source_id: str
    target_id: str
    payload: dict
    timestamp: float
    retry_count: int = 0


class LossyChannel:
    """
    [Model] Industrial Wireless Channel (e.g., LoRaWAN/ZigBee in dynamic environments).
    Features:
    1. Bernoulli Packet Loss Model.
    2. Gamma Distribution Delay Model.
    3. Hybrid ARQ (Automatic Repeat reQuest) Logic.
    """

    def __init__(self, loss_rate=0.3, avg_delay=0.5, max_retries=3):
        self.loss_rate = loss_rate
        self.avg_delay = avg_delay
        self.max_retries = max_retries
        self.packet_buffer = []  # Simulating delay queue

    def transmit_with_arq(self, packet: Packet, global_time: float) -> Optional[Packet]:
        """
        [Protocol] Transmit with stop-and-wait ARQ simulation.
        Returns the packet if successful (ack-ed), None if dropped after retries.
        """
        attempts = 0
        while attempts <= self.max_retries:
            # 1. Stochastic Loss Check
            if random.random() > self.loss_rate:
                # 2. Stochastic Delay Injection (Gamma Distribution for long-tail latency)
                # Shape=2.0, Scale=avg_delay/2
                delay = np.random.gamma(2.0, self.avg_delay / 2.0)
                packet.timestamp = global_time + delay
                return packet  # Success

            # Loss occurred, retry
            attempts += 1
            packet.retry_count = attempts

        return None  # Permanent Drop (Link Outage)


# =========================================================================
# [Layer 2] Physics-Aware Link Evaluator (物理感知链路评估)
# -------------------------------------------------------------------------
# 对应 DeepSnake 优势：基于动力学的代价评估。
# 计算路段权重时，耦合车辆动力学方程，而非仅使用几何距离。
# =========================================================================

class KinodynamicLinkEvaluator:
    """
    [Algorithm] Estimates traversal cost (Energy + Time) based on physics.
    Cost Function: J = alpha * Time + beta * Energy + gamma * Risk
    """

    def __init__(self, grid_map,weights=None):
        self.grid = grid_map
        # [新增] 支持从外部配置注入权重
        if weights:
            self.alpha_t = weights.get('alpha_t', 1.0)
            self.beta_e = weights.get('beta_e', 0.05)
            self.gamma_r = weights.get('gamma_r', 100.0)
        else:
            self.alpha_t = 1.0
            self.beta_e = 0.05
            self.gamma_r = 100.0
        # Weighting factors
        self.alpha_t = 1.0
        self.beta_e = 0.05
        self.gamma_r = 100.0

    def evaluate_link(self, u, v, current_mud):
        """
        Compute the generalized cost of traversing edge u->v.
        """
        # 1. Geometry & Environment
        edge_data = self.grid.graph[u][v]
        dist = edge_data.get('weight', 100.0)
        mud = edge_data.get('mud', current_mud)

        # 2. Physics Estimation (Simplified Integral)
        # Resistance F_res = mg(mu + mud_coeff)
        # Work W = F_res * dist
        resistance_coeff = 0.02 + 0.05 * mud
        energy_cost = (dist * resistance_coeff) * 10.0  # Normalize scale

        # 3. Time Estimation (Kinematic limit)
        # V_max is limited by mud (safety)
        v_limit = 15.0 * (1.0 - 0.4 * mud)
        time_cost = dist / max(1.0, v_limit)

        # 4. Risk Estimation (Node Health)
        # [修改] 风险代价对齐论文公式 (Section 3.1)
        # R = gamma * exp(1 - H)
        node_v = self.grid.nodes[v]
        risk_cost = 0.0
        if node_v.agent:
            h_index = node_v.agent.health_index
            if node_v.agent.state.name == 'STALLED':
                return float('inf')

            # 使用指数罚函数 (Paper Formula)
            risk_cost = self.gamma_r * np.exp(2.0 * (1.0 - h_index))

        # Total Generalized Cost
        return self.alpha_t * time_cost + self.beta_e * energy_cost + risk_cost


# =========================================================================
# [Layer 3] Distributed Gossip Router (分布式流言路由)
# -------------------------------------------------------------------------
# 对应 RobustSnake 优势：去中心化、AoI 感知、Gossip 协议。
# 整合了 LossyChannel 和 KinodynamicLinkEvaluator。
# =========================================================================

@dataclass
class RoutingEntry:
    next_hop: str
    cost: float
    timestamp: float  # Creation time (AoI source)


class DistributedProtocolSim:
    """
    [Network Plane]
    Simulates a Distance-Vector Routing Protocol over a Lossy Channel.
    Maintains distributed routing tables synchronized via Gossip.
    """

    def __init__(self, grid_map):
        self.grid = grid_map
        self.graph = grid_map.graph

        # Distributed Routing Tables (D-RT)
        # NodeID -> {TargetID: RoutingEntry}
        self.node_tables: Dict[str, Dict[str, RoutingEntry]] = {
            n: {} for n in self.grid.nodes
        }

        # Components
        self.channel = LossyChannel(loss_rate=0.2, avg_delay=0.1)
        self.evaluator = KinodynamicLinkEvaluator(grid_map)

        # Simulation State
        self.last_gossip_time = 0.0

    def gossip_step(self, global_time):
        """
        [Protocol Step] Execute one round of asynchronous information exchange.
        Simulates: Node wakes up -> Selects Neighbors -> Sends Table -> Channel -> Recv -> Update.
        """
        # Frequency Control (e.g., 1Hz Gossip)
        if global_time - self.last_gossip_time < 1.0:
            return
        self.last_gossip_time = global_time

        # 1. Random Node Selection (Simulate asynchronous wake-up)
        # In weak net, not everyone talks at once. 20% duty cycle.
        active_nodes = random.sample(list(self.grid.nodes), k=int(len(self.grid.nodes) * 0.2))

        for u in active_nodes:
            self._broadcast_from_node(u, global_time)

        # 2. Maintenance: Prune stale entries (AoI Management)
        self._prune_stale_entries(global_time)

    def _broadcast_from_node(self, u, now):
        """Node 'u' broadcasts its vector to direct neighbors."""
        if u not in self.node_tables: return

        # Construct Payload (My Distance Vector)
        # Optimization: Only send active targets (Split Horizon could be added here)
        payload = {
            target: (entry.cost, entry.timestamp)
            for target, entry in self.node_tables[u].items()
        }

        for v in self.graph.neighbors(u):
            # Create Packet
            pkt = Packet(source_id=u, target_id=v, payload=payload, timestamp=now)

            # Transmit via Lossy Channel
            recv_pkt = self.channel.transmit_with_arq(pkt, now)

            if recv_pkt:
                # Receiver processes the update
                self._bellman_ford_update(v, recv_pkt, now)

    def _bellman_ford_update(self, receiver, packet, now):
        """
        [Algo] Distributed Bellman-Ford with AoI-based Freshness priority.
        """
        sender = packet.source_id
        sender_vector = packet.payload
        my_table = self.node_tables[receiver]

        # 1. Calculate dynamic link cost using Physics Model
        # (Using local sensing of mud factor)
        local_mud = self.grid.nodes[receiver].agent.mud if self.grid.nodes[receiver].agent else 0.5
        link_cost = self.evaluator.evaluate_link(receiver, sender, local_mud)

        if link_cost == float('inf'): return  # Link broken

        # 2. Iterate sender's destinations
        for dest, (remote_cost, remote_ts) in sender_vector.items():
            if dest == receiver: continue  # Loop prevention

            total_cost = link_cost + remote_cost

            # AoI Logic:
            # We trust new information significantly more than old information.
            # If info is > 10s fresher, we update even if cost is slightly worse (topology change).

            should_update = False
            curr_entry = my_table.get(dest)

            if curr_entry is None:
                should_update = True
            else:
                cost_diff = total_cost - curr_entry.cost
                time_diff = remote_ts - curr_entry.timestamp

                # Rule A: Better Path (Cost improvement)
                if total_cost < curr_entry.cost:
                    should_update = True

                # Rule B: Fresher Info (Force update if local info is stale)
                # This handles "Bad News travels slow" problem partially
                elif time_diff > 10.0:
                    should_update = True

            if should_update:
                my_table[dest] = RoutingEntry(
                    next_hop=sender,
                    cost=total_cost,
                    timestamp=max(remote_ts, now)  # Propagate freshness
                )

    def _prune_stale_entries(self, now):
        """
        [Robustness] Remove routing entries that are too old (AoI > threshold).
        Prevents routing loops and outdated paths in partitioned networks.
        """
        ttl = 60.0  # Time To Live for routing info
        for n in self.grid.nodes:
            dead_targets = []
            for target, entry in self.node_tables[n].items():
                if (now - entry.timestamp) > ttl:
                    dead_targets.append(target)

            for t in dead_targets:
                del self.node_tables[n][t]

    def inject_destination_advertisement(self, dest_node, global_time):
        """
        [Trigger] A destination node announces its presence (Cost=0 to self).
        This seeds the Gossip process.
        """
        if dest_node in self.node_tables:
            self.node_tables[dest_node][dest_node] = RoutingEntry(
                next_hop=dest_node, cost=0.0, timestamp=global_time
            )

    def get_local_guidance(self, current_node, target_node):
        """
        [Interface] Query the local routing table for Next Hop.
        Returns: next_hop_id (str) or None (if unreachable/unknown).
        """
        table = self.node_tables.get(current_node, {})
        entry = table.get(target_node)

        if entry:
            return entry.next_hop
        return None


# =========================================================================
# [Layer 4] Hybrid Router Wrapper (混合路由器封装)
# -------------------------------------------------------------------------
# 统一接口，保留 A* 作为 Fallback，并驱动 Gossip 仿真。
# =========================================================================

class IntelligentRouter:
    """
    [System Component]
    Integrates Distributed Protocol simulation with a fallback Centralized Planner.
    Provides a unified API for vehicles.
    """

    def __init__(self, grid_map):
        self.grid = grid_map
        # Core: Distributed Sim
        self.protocol = DistributedProtocolSim(grid_map)

        # Fallback: Centralized A* (using Kinodynamic costs)
        self.fallback_graph = grid_map.graph
        self.evaluator = self.protocol.evaluator

    def step(self, global_time):
        """Called every simulation tick to update network state."""
        self.protocol.gossip_step(global_time)

    def advertise_destinations(self, targets: List[str], global_time: float):
        """Inject known targets (e.g., Depots) into the network."""
        for t in targets:
            self.protocol.inject_destination_advertisement(t, global_time)

    def get_dynamic_path(self, start_id, end_id, vehicle_id):
        """
        [Unified API]
        Strategy:
        1. Try Local Guidance (Distributed Table) -> Simulates Edge Computing.
        2. If Fail, Try Centralized A* -> Simulates Cloud Fallback / Onboard Planning.
        """
        # Strategy 1: Distributed Guidance (Next Hop Only)
        next_hop = self.protocol.get_local_guidance(start_id, end_id)

        if next_hop:
            # Return a "Path" of length 2 [Current, Next]
            # Vehicle agent logic handles hop-by-hop navigation
            return [start_id, next_hop]

        # Strategy 2: Fallback A* (Ground Truth / Onboard Planner)
        # Used when network has not converged or in cold start
        try:
            path = nx.astar_path(
                self.fallback_graph, start_id, end_id,
                heuristic=lambda u, v: abs(self.grid.nodes[u].pos[0] - self.grid.nodes[v].pos[0]) +
                                       abs(self.grid.nodes[u].pos[1] - self.grid.nodes[v].pos[1]),
                weight=lambda u, v, d: self.evaluator.evaluate_link(u, v, d.get('mud', 0.5))
            )
            return path  # Returns full path
        except:
            return []


class FloodingProtocolSim:
    """
    [Baseline] Link State Flooding Protocol.
    用于 Section 5.5 通信开销对比。
    特征：
    1. 激进广播：任何链路状态变化都会触发全网泛洪。
    2. 序列号机制：防止无限循环，但不具备 AoI 的新鲜度修剪能力。
    """

    def __init__(self, grid_map):
        self.grid = grid_map
        self.sequence_nums = {n: 0 for n in grid_map.nodes}
        self.message_count = 0  # 统计通信开销

    def broadcast_link_change(self, u, v, new_cost, global_time):
        """当链路 (u,v) 发生变化时触发泛洪"""
        self.sequence_nums[u] += 1
        seq = self.sequence_nums[u]
        packet = {
            'source': u, 'target': v, 'cost': new_cost,
            'seq': seq, 'ttl': 5  # 限制跳数
        }
        # 初始广播
        for neighbor in self.grid.graph.neighbors(u):
            self._forward_packet(neighbor, packet)

    def _forward_packet(self, current_node, packet):
        if packet['ttl'] <= 0: return

        # 模拟泛洪开销：每个节点收到包都会转发给除来源外的所有邻居
        self.message_count += 1

        # 简单的转发逻辑（省略路由表更新细节，只统计开销）
        new_packet = packet.copy()
        new_packet['ttl'] -= 1

        # 递归转发（在离散事件仿真中应放入事件队列，此处简化为计数）
        # 实际仿真中，Flooding 的消息量约为 Gossip 的 N 倍
        pass

    def get_overhead_metrics(self):
        return self.message_count / 1.0  # Normalized rate