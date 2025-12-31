import networkx as nx
import logging
from collections import defaultdict
from typing import List, Tuple, Dict
from interfaces import IScheduler, LockStatus

logger = logging.getLogger("TimeSpaceScheduler")


class TimeWindow:
    def __init__(self, start, end, vid, priority):
        self.start, self.end, self.vid, self.priority = start, end, vid, priority

    def overlaps(self, other):
        return max(self.start, other.start) < min(self.end, other.end)


class TimeSpaceScheduler(IScheduler):
    def __init__(self, grid_map, env_config):
        self.reservations = defaultdict(list)
        self.waiting_for = {}
        self.held_by = {}
        self.agent_timestamps = {}

    def register_agent(self, vid, timestamp):
        self.agent_timestamps[vid] = timestamp

    def get_segment_key(self, u, v):
        return tuple(sorted((u, v)))

    def request_segment(self, vid, u, v, priority, duration):
        seg_key = self.get_segment_key(u, v)
        return self._request_generic(vid, seg_key, priority, duration)

    def request_node(self, vid, node_id, priority, duration):
        return self._request_generic(vid, node_id, priority, duration)

    def _request_generic(self, vid, resource_id, priority, duration):
        if resource_id in self.held_by:
            if self.held_by[resource_id] == vid:
                return LockStatus.GRANTED, "RENEWED"

            holder = self.held_by[resource_id]
            my_ts = self.agent_timestamps.get(vid, float('inf'))
            holder_ts = self.agent_timestamps.get(holder, float('inf'))

            if priority > 10 or my_ts < holder_ts:  # Wound-Wait
                self._force_preempt(resource_id, vid)
                return LockStatus.GRANTED, "PREEMPTED"
            else:
                self.waiting_for[vid] = resource_id
                return LockStatus.WAITING, "QUEUED"

        self.held_by[resource_id] = vid
        if vid in self.waiting_for: del self.waiting_for[vid]
        return LockStatus.GRANTED, "OK"

    def _force_preempt(self, res, new_vid):
        self.held_by[res] = new_vid

    def release_segment(self, vid, u, v):
        return self._release(vid, self.get_segment_key(u, v))

    def release_node(self, vid, n):
        return self._release(vid, n)

    def _release(self, vid, res):
        if self.held_by.get(res) == vid:
            del self.held_by[res]
            return True
        return False

    def get_queue_length(self, res):
        return list(self.waiting_for.values()).count(res)

    def watchdog_check(self):
        # 死锁检测
        wfg = nx.DiGraph()
        for waiter, res in self.waiting_for.items():
            if res in self.held_by: wfg.add_edge(waiter, self.held_by[res])
        try:
            cycles = list(nx.simple_cycles(wfg))
            if cycles:
                victim = cycles[0][0]
                logger.warning(f"Deadlock! Kicking {victim}")
                to_del = [r for r, h in self.held_by.items() if h == victim]
                for r in to_del: del self.held_by[r]
        except:
            pass