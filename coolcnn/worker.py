from __future__ import annotations
from multiprocessing.pool import Pool
from multiprocessing import cpu_count


class Worker:
    __instance = None

    def __init__(self):
        if Worker.__instance is not None:
            raise Exception('Only allowed 1 instance')
        else:
            Worker.__instance = self
            self.pool = Pool(4)

    def __del__(self):
        self.pool.close()

    @staticmethod
    def get_instance() -> Worker:
        if Worker.__instance is None:
            Worker()
        return Worker.__instance
