from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Dict, Any, List

class LockStatus(Enum):
    GRANTED = 1
    WAITING = 2
    REJECTED = 3
    PREEMPTED = 4
    JAMMED = 5

class NodeType(Enum):
    T_JUNCTION = "T_JUNCTION"
    CROSS = "CROSS"
    STATION = "STATION"
    DEPOT = "DEPOT"

class VehicleState(Enum):
    IDLE = 0
    PLANNING = 1
    REQUESTING_ACCESS = 2
    MOVING = 3
    WAITING_SIGNAL = 4
    LOADING = 5
    UNLOADING = 6
    RETURNING = 7
    JAMMED = 8
    EMERGENCY_STOP = 9
    CRASHED = 10

class IDiagnosable(ABC):
    @abstractmethod
    def get_diagnostics(self) -> Dict[str, Any]:
        pass

class IPhysicalComponent(IDiagnosable):
    @abstractmethod
    def step(self, dt: float, global_time: float):
        pass
    @abstractmethod
    def get_energy_consumption(self) -> float:
        pass

class ILinkLayer(IDiagnosable):
    @abstractmethod
    def transmit(self, dist: float, payload_size: int, velocity: float = 0.0) -> Tuple[bool, float, float, float, Dict]:
        pass

class IScheduler(ABC):
    """边缘调度器接口"""
    @abstractmethod
    def request_node(self, vid: str, node_id: Any, priority: int, duration: float) -> Tuple[LockStatus, str]:
        pass
    @abstractmethod
    def request_segment(self, vid: str, u: Any, v: Any, priority: int, duration: float) -> Tuple[LockStatus, str]:
        pass
    @abstractmethod
    def release_node(self, vid: str, node_id: Any) -> bool:
        pass
    @abstractmethod
    def release_segment(self, vid: str, u: Any, v: Any) -> bool:
        pass
    @abstractmethod
    def get_queue_length(self, resource_id: Any) -> int:
        pass
    @abstractmethod
    def register_agent(self, vid: str, timestamp: float):
        pass