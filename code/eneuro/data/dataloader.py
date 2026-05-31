from . import Dataset
import random
from typing import Tuple, List
from ..base import Tensor
import multiprocessing as mp
from multiprocessing import Queue, Process
import numpy as np


def _worker_loop(dataset, indices, batch_size, drop_last, output_queue, indices_queue, num_batches):
    """
    子进程中的数据加载循环函数
    
    """
    cursor = 0
    batch_data_list = []
    batch_target_list = []
    
    while cursor < len(indices):
        remaining = len(indices) - cursor
        current_batch_size = min(batch_size, remaining)
        
        if current_batch_size < batch_size and drop_last:
            break
        
        batch_indices = indices[cursor:cursor + current_batch_size]
        cursor += current_batch_size
        
        batch_data = []
        batch_target = []
        
        for idx in batch_indices:
            data, target = dataset[idx]
            batch_data.append(data)
            batch_target.append(target)
        
        batch_data_tensor = Tensor.stack(batch_data)
        batch_target_tensor = Tensor.stack(batch_target)
        
        output_queue.put((batch_data_tensor, batch_target_tensor))
    
    output_queue.put(None)


class AsyncDataLoader:
    """
    异步数据加载器，支持多进程并行加载数据
    
    通过num_workers参数控制是否启用异步加载:
    - num_workers=0: 同步加载，在主进程中顺序加载数据
    - num_workers>0: 异步加载，使用多个子进程并行加载数据
    
    """
    
    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False, 
                 drop_last: bool = False, num_workers: int = 0, prefetch_factor: int = 2) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        
        self.indices = list(range(len(self.dataset)))
        if shuffle:
            random.shuffle(self.indices)
        
        self._num_batches = (len(self.indices) + batch_size - 1) // batch_size if not drop_last else len(self.indices) // batch_size
    
    def __len__(self) -> int:
        """返回DataLoader的总批次数"""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        """
        根据num_workers自动选择同步或异步迭代器
        
        当num_workers=0时使用同步迭代器，与原有DataLoader行为一致
        当num_workers>0时使用异步迭代器，开启多进程并行加载
        """
        if self.num_workers == 0:
            return _SyncLoaderIter(self)
        return _AsyncLoaderIter(self)


class _SyncLoaderIter:
    """
    同步加载迭代器，当num_workers=0时使用
    
    在主进程中顺序加载数据，不涉及多进程开销
    """
    
    def __init__(self, loader: AsyncDataLoader):
        self.loader = loader
        self.indices = loader.indices.copy()
        if loader.shuffle:
            random.shuffle(self.indices)
        self.cursor = 0
        self.num_samples = len(self.indices)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """返回下一个batch数据"""
        if self.cursor >= self.num_samples:
            raise StopIteration
        
        remaining = self.num_samples - self.cursor
        if remaining < self.loader.batch_size and self.loader.drop_last:
            raise StopIteration
        
        current_batch_size = min(self.loader.batch_size, remaining)
        batch_indices = self.indices[self.cursor:self.cursor + current_batch_size]
        self.cursor += current_batch_size
        
        batch_data, batch_target = self._load_batch(batch_indices)
        return batch_data, batch_target
    
    def _load_batch(self, indices):
        """根据索引列表加载一个batch的数据"""
        batch_data = []
        batch_target = []
        for idx in indices:
            data, target = self.loader.dataset[idx]
            batch_data.append(data)
            batch_target.append(target)
        return Tensor.stack(batch_data), Tensor.stack(batch_target)


class _AsyncLoaderIter:
    """
    异步加载迭代器,当num_workers>0时使用
    
    使用多个子进程并行加载数据,通过Queue进行进程间通信
    """
    
    def __init__(self, loader: AsyncDataLoader):
        self.loader = loader
        self.workers = []
        self.output_queues = []
        self.indices_queue = Queue()
        self.num_workers = loader.num_workers
        self.prefetch_factor = loader.prefetch_factor
        self._shutdown = False
        
        total_batches = len(self)
        batches_per_worker = (total_batches + self.num_workers - 1) // self.num_workers
        
        for w in range(self.num_workers):
            output_queue = Queue(maxsize=self.prefetch_factor)
            start_batch = w * batches_per_worker
            end_batch = min(start_batch + batches_per_worker, total_batches)
            worker_indices = self._get_batch_indices(start_batch, end_batch)
            
            p = Process(target=_worker_loop, args=(
                self.loader.dataset, 
                worker_indices,
                self.loader.batch_size,
                self.loader.drop_last,
                output_queue,
                self.indices_queue,
                end_batch - start_batch
            ))
            p.start()
            self.workers.append(p)
            self.output_queues.append(output_queue)
        
        self._batches_yielded = 0
        self._total_batches = total_batches
    
    def _get_batch_indices(self, start_batch: int, end_batch: int) -> List[int]:
        """
        根据batch范围获取对应的样本索引列表
        
        参数:
            start_batch: 起始batch索引
            end_batch: 结束batch索引
        
        返回:
            包含该batch范围内所有样本索引的列表
        """
        indices = []
        for batch_idx in range(start_batch, end_batch):
            batch_start = batch_idx * self.loader.batch_size
            batch_end = min(batch_start + self.loader.batch_size, len(self.loader.indices))
            indices.extend(self.loader.indices[batch_start:batch_end])
        return indices
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """
        获取下一个可用的batch数据
        
        优先从已有数据的队列中取，若都为空则阻塞等待
        """
        if self._batches_yielded >= self._total_batches:
            self._cleanup()
            raise StopIteration
        
        for output_queue in self.output_queues:
            if not output_queue.empty():
                result = output_queue.get()
                if result is None:
                    continue
                self._batches_yielded += 1
                return result
        
        result = self.output_queues[self._batches_yielded % len(self.output_queues)].get()
        if result is None:
            self._cleanup()
            raise StopIteration
        self._batches_yielded += 1
        return result
    
    def _cleanup(self):
        """
        清理所有子进程，释放资源
        
        依次发送终止信号、等待退出，若超时则强制杀死进程
        """
        if self._shutdown:
            return
        self._shutdown = True
        for p in self.workers:
            p.terminate()
            p.join(timeout=1)
            if p.is_alive():
                p.kill()
    
    def __del__(self):
        """析构时确保进程被正确清理"""
        self._cleanup()

class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False, drop_last: bool = False) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self) -> '_LoaderIter':
        return _LoaderIter(self)   # 每次都新建一个迭代器
    
    def __len__(self) -> int:
        """返回DataLoader的批次数"""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

class _LoaderIter:
    def __init__(self, loader: DataLoader):
        self.dataset = loader.dataset
        self.batch_size = loader.batch_size
        self.drop_last = loader.drop_last
        
        # 生成索引
        self.indices = list(range(len(self.dataset)))
        if loader.shuffle:
            random.shuffle(self.indices)
        
        self.cursor = 0
        self.num_samples = len(self.indices)

    def __iter__(self) -> '_LoaderIter':
        return self

    def __next__(self) -> Tuple[Tensor, Tensor]:
        """返回下一个批量数据"""
        
        # 检查是否已经遍历完所有样本
        if self.cursor >= self.num_samples:
            raise StopIteration
        
        # 检查是否剩余样本不足一个batch且设置了drop_last
        remaining = self.num_samples - self.cursor
        if remaining < self.batch_size and self.drop_last:
            raise StopIteration
        
        # 获取当前batch的实际大小
        current_batch_size = min(self.batch_size, remaining)
        
        # 获取当前批次的索引
        batch_indices = self.indices[self.cursor:self.cursor + current_batch_size]
        self.cursor += current_batch_size
        
        # 批量加载数据
        batch_data, batch_target = self._load_batch(batch_indices)
        
        return batch_data, batch_target
    
    def _load_batch(self, indices: List[int]) -> Tuple[Tensor, Tensor]:
        """加载指定索引的批量数据"""
        batch_data = []
        batch_target = []
        
        for idx in indices:
            # 从数据集中获取单个样本
            data, target = self.dataset[idx]
            batch_data.append(data)
            batch_target.append(target)
        
        # 将列表中的样本堆叠成批量Tensor
        batch_data_tensor = Tensor.stack(batch_data)
        batch_target_tensor = Tensor.stack(batch_target)
        
        return batch_data_tensor, batch_target_tensor
