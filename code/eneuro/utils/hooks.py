"""
带调试信息的钩子系统
"""
import weakref
from typing import Callable, Dict, Any, Optional, List
import numpy as np


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
        self._forward_hooks: Dict[int, Callable] = {}
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

    def trigger_backward_hooks(self, grad_inputs, grad_outputs):
        """触发所有反向钩子"""
        for hook in self._backward_hooks.values():
            hook(grad_inputs, grad_outputs)

    def clear_all_hooks(self):
        """清除所有钩子"""
        self._forward_hooks.clear()
        self._backward_hooks.clear()


class HookRegistry:
    """全局钩子注册表"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._function_to_layer = weakref.WeakKeyDictionary()
            cls._instance._current_layer = None
        return cls._instance

    def register_function_layer_mapping(self, function, layer):
        self._function_to_layer[function] = weakref.ref(layer)

    def get_layer_for_function(self, function):
        if function in self._function_to_layer:
            return self._function_to_layer[function]()
        return None

    def set_current_layer(self, layer):
        self._current_layer = layer

    def get_current_layer(self):
        return self._current_layer


def add_hooks_to_module():
    """向现有的 Layer 和 Module 类添加钩子功能"""
    from ..nn.module import Layer
    from ..base.core import Function

    original_layer_call = Layer.__call__
    original_function_call = Function.__call__
    original_function_backward = Function.backward

    hook_registry = HookRegistry()

    def layer_call_with_hooks(self, *inputs):
        if not hasattr(self, '_hook_manager'):
            self._hook_manager = HookManager()
            self._saved_inputs = None
            self._saved_outputs = None

        self._saved_inputs = [weakref.ref(x) for x in inputs]

        hook_registry.set_current_layer(self)

        outputs = original_layer_call(self, *inputs)

        hook_registry.set_current_layer(None)

        if not isinstance(outputs, tuple):
            output_tuple = (outputs,)
        else:
            output_tuple = outputs
        self._saved_outputs = [weakref.ref(y) for y in output_tuple]

        self._hook_manager.trigger_forward_hooks(self, inputs, outputs)

        return outputs

    Layer.__call__ = layer_call_with_hooks

    def register_forward_hook(self, hook_fn: Callable) -> HookHandle:
        if not hasattr(self, '_hook_manager'):
            self._hook_manager = HookManager()
        return self._hook_manager.register_forward_hook(hook_fn)

    def register_backward_hook(self, hook_fn: Callable) -> HookHandle:
        if not hasattr(self, '_hook_manager'):
            self._hook_manager = HookManager()
        return self._hook_manager.register_backward_hook(hook_fn)

    Layer.register_forward_hook = register_forward_hook
    Layer.register_backward_hook = register_backward_hook

    def function_call_with_hooks(self, *inputs):
        if not hasattr(self, '_hook_manager'):
            self._hook_manager = HookManager()

        current_layer = hook_registry.get_current_layer()
        if current_layer is not None:
            hook_registry.register_function_layer_mapping(self, current_layer)

        self._hook_manager.trigger_forward_hooks(self, inputs, None, phase='pre')

        outputs = original_function_call(self, *inputs)

        self._hook_manager.trigger_forward_hooks(self, inputs, outputs, phase='post')

        # 将每个函数实例的 backward 包装为实例属性。
        # 实例属性优先于类方法，所以无论子类如何覆盖 backward，
        # 这里的包装都能被正确拦截，而 type(self).backward 则
        # 精确调用该子类自身的真实 backward 实现。
        real_backward = type(self).backward

        def instance_backward_with_hooks(gy):
            grad_inputs = real_backward(self, gy)

            if not isinstance(grad_inputs, tuple):
                grad_inputs_tuple = (grad_inputs,)
            else:
                grad_inputs_tuple = grad_inputs

            grad_outputs = (gy,) if not isinstance(gy, tuple) else gy

            if self._hook_manager._backward_hooks:
                self._hook_manager.trigger_backward_hooks(grad_inputs_tuple, grad_outputs)

            layer = hook_registry.get_layer_for_function(self)
            if layer is not None and hasattr(layer, '_hook_manager') and layer._hook_manager._backward_hooks:
                layer._hook_manager.trigger_backward_hooks(grad_inputs_tuple, grad_outputs)

            return grad_inputs

        self.backward = instance_backward_with_hooks

        return outputs

    Function.__call__ = function_call_with_hooks

    Function.register_forward_hook = register_forward_hook
    Function.register_backward_hook = register_backward_hook


def capture_features(layer):
    """捕获特征图"""
    storage = {'input': None, 'output': None}

    def hook(module, inputs, outputs):
        storage['input'] = [x.data.copy() if hasattr(x, 'data') else x for x in inputs]
        storage['output'] = outputs.data.copy() if hasattr(outputs, 'data') else outputs

    handle = layer.register_forward_hook(hook)
    return handle, storage


def capture_gradients(layer):
    """捕获梯度"""
    storage = {'grad_output': None, 'grad_input': None}

    def hook(grad_inputs, grad_outputs):
        if grad_outputs and len(grad_outputs) > 0 and grad_outputs[0] is not None:
            grad = grad_outputs[0]
            storage['grad_output'] = grad.data.copy() if hasattr(grad, 'data') else np.array(grad)

        if grad_inputs:
            storage['grad_input'] = [
                g.data.copy() if g is not None and hasattr(g, 'data') else None
                for g in grad_inputs
            ]

    handle = layer.register_backward_hook(hook)
    return handle, storage


def backward_hook(layer):
    """装饰器：为层添加反向钩子"""
    def decorator(hook_fn: Callable):
        handle = layer.register_backward_hook(hook_fn)
        return handle
    return decorator


add_hooks_to_module()
