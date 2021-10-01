import time
from tabulate import tabulate
class Biddy:
    """ Intuitively staightforward profiler 
    """
    starts = {}
    total_spent = {}
    counts = {}

    def start(self, s=""):
        self.starts[s] = time.time()
        if s in self.counts:
            self.counts[s] = self.counts[s] + 1
        else:
            self.counts[s] = 1
        
    def end(self, s=""):
        time_spent = time.time() - self.starts[s]
        if s in self.total_spent:
            self.total_spent[s] = self.total_spent[s] + time_spent
        else:    
            self.total_spent[s] = time_spent
        return time_spent

    def print_results(self):
        print(tabulate([("Mark", "Total time spent in seconds")] + \
              sorted(self.total_spent.items()), tablefmt='grid'))

        print(tabulate([("Mark", "Mean time per mark")] + \
              sorted((x, y / self.counts[x] ) for (x, y) in self.total_spent.items() ), tablefmt='grid'))
