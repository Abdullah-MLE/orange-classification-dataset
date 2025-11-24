from queue import Queue


class QualityQueue:
    def __init__(self):
        self.queue = Queue()
        self.total_fresh = 0
        self.total_rotten = 0
    
    def add(self, value):
        self.queue.put(value)
        
        if value == 1:
            self.total_fresh += 1
            print(f"  ✓ FRESH orange added to queue")
        else:
            self.total_rotten += 1
            print(f"  ✗ ROTTEN orange added to queue")
        
        print(f"  Queue size: {self.queue.qsize()}")
    
    def get_all(self):
        items = []
        while not self.queue.empty():
            items.append(self.queue.get())
        return items
    
    def size(self):
        return self.queue.qsize()
    
    def get_stats(self):
        return {
            'queue_size': self.queue.qsize(),
            'total_fresh': self.total_fresh,
            'total_rotten': self.total_rotten,
            'total': self.total_fresh + self.total_rotten
        }