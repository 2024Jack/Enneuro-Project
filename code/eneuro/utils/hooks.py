"""
钩子系统 - 用于捕获前向传播特征图和反向传播梯度
"""
import weakref
from typing import Callable, Dict, Any, Optional, List


class HookHandle:
    """钩子句柄，用于管理钩子的生命周期"""
    
    def __init__(self, hook_id: int, remove_fn: Callable[[int], None]):
        self.hook_id = hook_id
        self._remove_fn = remove_fn
        self._removed = False
        
    def remove(self):
        """移除钩子"""
        if not self._removed:
            self._remove_fn(self.hook_id)
            self._removed = True
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()


class HookManager:
    """钩子管理器 - 管理前向和反向传播钩子"""
    
    def __init__(self):
        self._forward_hooks: Dict[int, Callable] = {} # Callable是可调用对象，即函数、方法或任何实现了__call__方法的对象，用于存储钩子函数
        self._backward_hooks: Dict[int, Callable] = {}
        self._hook_counter = 0
        
    def register_forward_hook(self, hook_fn: Callable) -> HookHandle:
        """注册前向传播钩子"""
        hook_id = self._hook_counter
        self._hook_counter += 1
        self._forward_hooks[hook_id] = hook_fn
        
        def remove_func(id_to_remove: int):
            if id_to_remove in self._forward_hooks:
                del self._forward_hooks[id_to_remove]
                
        return HookHandle(hook_id, remove_func)
    
    def register_backward_hook(self, hook_fn: Callable) -> HookHandle:
        """注册反向传播钩子"""
        hook_id = self._hook_counter
        self._hook_counter += 1
        self._backward_hooks[hook_id] = hook_fn
        
        def remove_func(id_to_remove: int):
            if id_to_remove in self._backward_hooks:
                del self._backward_hooks[id_to_remove]
                
        return HookHandle(hook_id, remove_func)
    
    def trigger_forward_hooks(self, *args, **kwargs):
        """触发所有前向钩子"""
        for hook in self._forward_hooks.values():
            hook(*args, **kwargs)
            
    def trigger_backward_hooks(self, *args, **kwargs):
        """触发所有反向钩子"""
        for hook in self._backward_hooks.values():
            hook(*args, **kwargs)
            
    def clear_all_hooks(self):
        """清除所有钩子"""
        self._forward_hooks.clear()
        self._backward_hooks.clear()


def add_hooks_to_module():
    """向现有的 Layer 和 Module 类添加钩子功能"""
    from ..nn.module import Layer
    from ..base.core import Function
    
    # 保存原始方法
    original_layer_call = Layer.__call__
    original_function_call = Function.__call__
    
    # 为 Layer 类添加钩子管理器
    def layer_call_with_hooks(self, *inputs):
        # 初始化钩子管理器（如果还没有）
        if not hasattr(self, '_hook_manager'):
            self._hook_manager = HookManager()
            self._saved_inputs = None
            self._saved_outputs = None
            
        # 保存输入（用于反向钩子）
        self._saved_inputs = [weakref.ref(x) for x in inputs]
        
        # 调用原始方法
        outputs = original_layer_call(self, *inputs)
        
        # 保存输出
        if not isinstance(outputs, tuple):
            output_tuple = (outputs,)
        else:
            output_tuple = outputs
        self._saved_outputs = [weakref.ref(y) for y in output_tuple]
        
        # 触发前向钩子
        self._hook_manager.trigger_forward_hooks(self, inputs, outputs)
        
        return outputs
    
    # 替换 Layer.__call__
    Layer.__call__ = layer_call_with_hooks
    
    # 为 Layer 添加注册钩子的方法
    def register_forward_hook(self, hook_fn: Callable) -> HookHandle:
        """注册前向钩子"""
        if not hasattr(self, '_hook_manager'):
            self._hook_manager = HookManager()
        return self._hook_manager.register_forward_hook(hook_fn)
    
    def register_backward_hook(self, hook_fn: Callable) -> HookHandle:
        """注册反向钩子（通过 Function 层传递）"""
        if not hasattr(self, '_hook_manager'):
            self._hook_manager = HookManager()
        return self._hook_manager.register_backward_hook(hook_fn)
    
    Layer.register_forward_hook = register_forward_hook
    Layer.register_backward_hook = register_backward_hook
    
    # 为 Function 类添加钩子支持
    def function_call_with_hooks(self, *inputs):
        # 初始化钩子管理器
        if not hasattr(self, '_hook_manager'):
            self._hook_manager = HookManager()
            
        # 触发前置前向钩子
        self._hook_manager.trigger_forward_hooks(self, inputs, None, phase='pre')
        
        # 调用原始方法
        outputs = original_function_call(self, *inputs)
        
        # 触发后置前向钩子
        self._hook_manager.trigger_forward_hooks(self, inputs, outputs, phase='post')
        
        return outputs
    
    # 暂存 original backward（如果有的话）
    # 但我们需要一个更好的方式来捕获反向传播
    
    # 替换 Function.__call__
    Function.__call__ = function_call_with_hooks
    
    # 为 Function 添加注册钩子的方法
    Function.register_forward_hook = register_forward_hook
    Function.register_backward_hook = register_backward_hook


def capture_features(layer):
    """
    创建一个用于捕获特征图的前向钩子
    
    Args:
        layer: 要捕获的层
        
    Returns:
        tuple: (hook_handle, storage_dict)
    """
    storage = {'input': None, 'output': None}
    
    def hook(module, inputs, outputs):
        # 存储输入输出
        storage['input'] = [x.data.copy() if hasattr(x, 'data') else x for x in inputs]
        storage['output'] = outputs.data.copy() if hasattr(outputs, 'data') else outputs
        
    handle = layer.register_forward_hook(hook)
    return handle, storage


def capture_gradients(layer):
    """
    创建一个用于捕获梯度的钩子
    
    注意：这个函数需要配合反向传播使用
    
    Args:
        layer: 要捕获的层
        
    Returns:
        tuple: (hook_handle, storage_dict)
    """
    storage = {'input_grad': None, 'output_grad': None}
    
    # 注意：由于我们的反向传播是通过 Function 类处理的，
    # 我们需要一种方式来捕获梯度。这里提供一个基础框架。
    # 后续在 GradCAM 中我们会更深入地实现。
    
    def hook(module, inputs, outputs):
        # 这个占位符会在反向传播时补充
        pass
        
    handle = layer.register_backward_hook(hook)
    return handle, storage


# 在导入时自动初始化钩子系统
add_hooks_to_module()
