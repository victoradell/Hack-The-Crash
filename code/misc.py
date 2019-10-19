import argparse
import datetime
import time


class Stopwatch():
    
    def __init__(self):
        self.t0 = time.time()
        self.t = time.time()

    def mark(self):
        dt = (time.time()-self.t)
        self.t = time.time()
        return str(datetime.timedelta(seconds=dt))

    def total(self):
        dt = (time.time() - self.t0)
        return str(datetime.timedelta(seconds=dt))

class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def partgen(size, split):
    q = int(size/split)
    for i in range(q): yield split
    if size > q*split: yield size - q*split

def progress(i, n, p):
    b = int(n*p)
    if i % b == b - 1:
        return int((i+1)/b)*p
    else:
        return False
