from ..base import Tensor
import numpy as np
import time

class AverageMeter:
    """Computes and stores the average and current value of a metric."""
    def __init__(self, name='Metric', fmt=':.4f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # 统一转为 Python float，避免 numpy/cupy 标量流入 sum/avg，
        # 导致后续 matplotlib 绘图时因无法隐式转换 cupy array 而崩溃。
        if hasattr(val, 'item'):
            val = val.item()   # numpy / cupy 0-d array → Python scalar
        else:
            val = float(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class TimeMeter:
    """Measures the average time per batch."""
    def __init__(self):
        self.reset()
        self.start = time.time()

    def reset(self):
        self.elapsed = 0.0
        self.batch_time = 0.0
        self.data_time = 0.0
        self.start = time.time()

    def update(self, batch_time, data_time):
        self.elapsed += batch_time
        self.batch_time = batch_time
        self.data_time = data_time

    def avg(self):
        return self.elapsed / (self.count + 1e-8)
