from __future__ import annotations
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
import numpy as np
import weakref

from .graph import Graph, Node, NodeType

from ..base.core import Tensor, Function, Config
from ..nn.optim import Optimizer
from ..base.functions import to_xp, get_array_module
from ..base import functions as f

class AutoCastManager:
    @staticmethod
    def apply_cast(graph: Graph, dtype: str='float16') -> Graph:
        # 初始化
        for node in graph.nodes.values():
            node.casted = False
            node.decasted = False

        # 按拓扑序遍历所有 Function 节点
        for node in graph.topological_order():
            if node.type != NodeType.FUNCTION: # 排除tensor
                continue

            func_cls = node.true_obj.__class__
            # 已有Cast/Decast，则对后继节点（Tensor）进行标记
            if func_cls in (f.Cast, f.DeCast):
                suc_nodes = graph.get_successors(node)
                for suc_node in suc_nodes:
                    if func_cls == f.Cast:
                        suc_node.casted = True
                        suc_node.decasted = False
                    else:
                        suc_node.decasted = True
                        suc_node.casted = False
                    
            # 其他Function节点
            else:
                # 获取前继节点状态
                pre_nodes = graph.get_predecessors(node)
                pre_casted = True
                for pre in pre_nodes:
                    if pre.casted == False:
                        pre_casted = False
                        break

                pre_decasted = True
                for pre in pre_nodes:
                    if pre.decasted == False:
                        pre_decasted = False
                        break
                
                # 后继节点
                suc_nodes = graph.get_successors(node)

                # 可以低精度的Function
                if func_cls in f.CastRigistry.cancast:
                    # 若输入不是低精度，则添加Cast
                    if not pre_casted:
                        '''
                        pre(Tensor) -> node
                        变为
                        pre(Tensor) -> Cast -> Tensor -> node 
                        '''
                        # 去除原来的边
                        graph._remove_edges_to_node(node, keep_set=set(pre_nodes))
                        for pre in pre_nodes:
                            # 添加节点
                            cast_func = f.Cast(dtype=dtype)
                            tensor = cast_func(pre.true_obj)
                            cast_node = graph.add_node(cast_func)
                            tensor_node = graph.add_node(weakref.ref(tensor))
                            # 添加边
                            graph.add_edge(pre, cast_node)
                            graph.add_edge(cast_node, tensor_node)
                            graph.add_edge(tensor_node, node)
                    
                    # 标记后继节点
                    for suc in suc_nodes:
                        suc.casted = True
                        suc.decasted = False

                # 需要高精度的Function
                elif func_cls in f.CastRigistry.resist_cast:
                    # 若输入是低精度，则添加Decast
                    if not pre_decasted:
                        '''
                        pre(Tensor) -> node
                        变为
                        pre(Tensor) -> DeCast -> Tensor -> node 
                        '''
                        # 去除原来的边
                        graph._remove_edges_to_node(node, keep_set=set(pre_nodes))
                        for pre in pre_nodes:
                            # 添加节点
                            decast_func = f.DeCast(dtype=dtype)
                            tensor = decast_func(pre.true_obj)
                            decast_node = graph.add_node(decast_func)
                            tensor_node = graph.add_node(weakref.ref(tensor))
                            # 添加边
                            graph.add_edge(pre, decast_node)
                            graph.add_edge(decast_node, tensor_node)
                            graph.add_edge(tensor_node, node)
                    
                    # 标记后继节点
                    for suc in suc_nodes:
                        suc.casted = False
                        suc.decasted = True
                        
                # 没有要求的Function
                else:
                    # 传递标记
                    for suc in suc_nodes:
                        suc.casted = pre_casted
                        suc.decasted = pre_decasted

        return graph

class GradScaler:
    '''
    动态损失缩放器，使梯度落入FP16的可表示范围内。
    用例：
    scaler = GradScaler()  # 损失缩放器
    for input, target in data:
        with autocast(dtype='float16'):   # 前向：低精度
            output = model(input)
            loss = loss_fn(output, target)

        scaler.scale(loss).backward()         # 反向：缩放后的低精度梯度计算
        scaler.step(optimizer)               # 反缩放梯度并更新参数
    '''
    def __init__(self, init_scale=32768.0, growth_factor=2.0, backoff_factor=0.5, 
                 growth_interval=2000, min_scale=1.0) -> None:
        self.scale_factor = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.min_scale = min_scale

        self.step_count = 0

    # 缩放增大梯度，防止下溢
    def scale(self, loss: Tensor) -> Tensor:
        return self.scale_factor * loss

    def step(self, optimizer: Optimizer) -> None:
        # 检查溢出
        has_overflow = False
        for param in optimizer.params:
            if param.grad is None:
                continue

            xp = get_array_module(param.grad.data)
            if (xp.isinf(param.grad.data) | xp.isnan(param.grad.data)).any():
                has_overflow = True
                break

        # 溢出则本次梯度无效
        if has_overflow:
            optimizer.zero_grad() # 清除梯度

            self.scale_factor *= self.backoff_factor # 缩小缩放倍数
            self.scale_factor = max(self.scale_factor, self.min_scale) # 不小于min_scale

            self.step_count = 0

        # 未溢出则梯度下降
        else:
            # 还原梯度大小，正常更新
            for param in optimizer.params:
                if param.grad is None:
                    continue
                xp = get_array_module(param.grad.data)
                param.grad.data /= xp.asarray(self.scale_factor)
            optimizer.step()

            # 连续未溢出则放大缩放倍数
            self.step_count += 1
            if self.step_count >= self.growth_interval:
                self.scale_factor *= self.growth_factor
                self.step_count = 0
        
