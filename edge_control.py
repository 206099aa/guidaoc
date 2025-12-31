from interfaces import IScheduler, LockStatus
class EdgeController(IScheduler):
    def __init__(self, nodes, cfg):
        self.locks = {}
    def register_agent(self, vid, ts): pass
    def get_segment_key(self, u, v): return tuple(sorted((u, v)))
    def request_segment(self, vid, u, v, prio, dur):
        k = self.get_segment_key(u, v)
        if k not in self.locks or self.locks[k] == vid:
            self.locks[k] = vid
            return LockStatus.GRANTED, "OK"
        return LockStatus.REJECTED, "BUSY"
    def request_node(self, vid, n, p, d):
        if n not in self.locks or self.locks[n] == vid:
            self.locks[n] = vid
            return LockStatus.GRANTED, "OK"
        return LockStatus.REJECTED, "BUSY"
    def release_segment(self, vid, u, v):
        k = self.get_segment_key(u, v)
        if self.locks.get(k) == vid: del self.locks[k]; return True
        return False
    def release_node(self, vid, n):
        if self.locks.get(n) == vid: del self.locks[n]; return True
        return False
    def get_queue_length(self, r): return 0
    def watchdog_check(self): pass