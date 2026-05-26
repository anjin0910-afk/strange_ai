from queue import Empty, Full, Queue


class LatestFrameQueue:
    def __init__(self, max_size=2):
        self._queue = Queue(maxsize=max(1, max_size))

    def put_latest(self, frame):
        while self._queue.full():
            try:
                self._queue.get_nowait()
            except Empty:
                break
        try:
            self._queue.put_nowait(frame)
        except Full:
            pass

    def get(self, timeout=None):
        return self._queue.get(timeout=timeout)

    def empty(self):
        return self._queue.empty()
