"""
Guided Backpropagation 实现

参考论文: "Striving for Simplicity: The All Convolutional Net"
和 "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps"

核心原理：
- 在 ReLU 层的反向传播中，只保留正向梯度（> 0 的梯度）
- 这样可以得到更细粒度的特征可视化

公式：
对于 ReLU 层 y = max(0, x)
标准反向传播: dx = dy * I(x > 0)
Guided Backpropagation: dx = dy * I(x > 0) * I(dy > 0)
"""

import numpy as np
from typing import Dict, Optional, List
from scipy.ndimage import zoom

from ..base import Tensor
from ..nn.module import Module, Layer
from ..base import functions as F


class GuidedBackpropagation:
    """
    Guided Backpropagation 实现类
    
    Guided Backpropagation 通过修改 ReLU 层的反向传播行为，
    只保留正向梯度，从而生成更细粒度的可视化结果。
    
    使用方法：
    >>> guided_bp = GuidedBackpropagation(model)
    >>> saliency_map = guided_bp.generate(input_tensor, class_idx=0)
    """

    def __init__(self, model: Module):
        """
        初始化 Guided Backpropagation
        
        Args:
            model: 神经网络模型
        """
        self.model = model
        
        # 存储拦截的层信息
        self._hook_handles = []
        self._saved_activations = {}  # 存储正向激活值
        self._input_gradients = None
        
        # 标记是否处于 guided backprop 模式
        self._enabled = False

    def _enable(self):
        """启用 Guided Backpropagation 模式"""
        if self._enabled:
            return
        
        self._enabled = True
        self._saved_activations = {}
        self._input_gradients = None
        
        # 遍历模型，找到所有相关层并注册钩子
        self._register_hooks()

    def _disable(self):
        """禁用 Guided Backpropagation 模式"""
        if not self._enabled:
            return
        
        # 移除所有钩子
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        
        self._enabled = False
        self._saved_activations = {}
        self._input_gradients = None

    def _register_hooks(self):
        """注册钩子来拦截 ReLU 层的正向和反向传播"""
        visited = set()
        
        def register_layer_hooks(module):
            if id(module) in visited:
                return
            visited.add(id(module))
            
            for name, child in module.__dict__.items():
                if name.startswith('_'):
                    continue
                
                if isinstance(child, Layer):
                    # 为所有层注册前向钩子保存激活值
                    handle = child.register_forward_hook(self._forward_hook)
                    self._hook_handles.append(handle)
                    
                    # 为所有层注册反向钩子实现 guided backprop
                    handle = child.register_backward_hook(self._backward_hook)
                    self._hook_handles.append(handle)
                elif hasattr(child, '__dict__') and hasattr(child, '__call__'):
                    register_layer_hooks(child)
        
        register_layer_hooks(self.model)

    def _forward_hook(self, module, inputs, outputs):
        """前向钩子：保存激活值供反向传播使用"""
        module_id = id(module)
        
        if isinstance(outputs, tuple):
            self._saved_activations[module_id] = [
                out.data.copy() if hasattr(out, 'data') else np.array(out)
                for out in outputs
            ]
        else:
            self._saved_activations[module_id] = [
                outputs.data.copy() if hasattr(outputs, 'data') else np.array(outputs)
            ]

    def _backward_hook(self, grad_inputs, grad_outputs):
        """
        反向钩子：实现 Guided Backpropagation
        
        Guided Backpropagation 的核心逻辑：
        在每一层的反向传播中，只保留正向梯度
        """
        if grad_inputs is None or grad_outputs is None:
            return
        
        # 获取当前层的激活值
        # 注意：这里我们无法直接获取当前层，因为钩子只接收梯度参数
        # 所以我们需要在反向传播时处理梯度
        
        # Guided Backpropagation 的关键：只保留正向梯度
        # dx = dy * I(x > 0) * I(dy > 0)
        
        # 处理 grad_inputs
        if grad_inputs and isinstance(grad_inputs, tuple):
            new_grad_inputs = []
            for grad in grad_inputs:
                if grad is not None:
                    grad_data = grad.data if hasattr(grad, 'data') else grad
                    # 只保留正值梯度
                    grad_data = np.maximum(0, grad_data)
                    if hasattr(grad, 'data'):
                        grad.data = grad_data
                    new_grad_inputs.append(grad)
                else:
                    new_grad_inputs.append(None)
            grad_inputs = tuple(new_grad_inputs)
        
        # 处理 grad_outputs（传入本层的梯度）
        if grad_outputs and isinstance(grad_outputs, tuple):
            new_grad_outputs = []
            for grad in grad_outputs:
                if grad is not None:
                    grad_data = grad.data if hasattr(grad, 'data') else grad
                    # 只保留正值梯度
                    grad_data = np.maximum(0, grad_data)
                    if hasattr(grad, 'data'):
                        grad.data = grad_data
                    new_grad_outputs.append(grad)
                else:
                    new_grad_outputs.append(None)
            grad_outputs = tuple(new_grad_outputs)

    def generate(
        self,
        input_tensor: Tensor,
        class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        生成 Guided Backpropagation 显著性图
        
        Args:
            input_tensor: 输入张量 (N, C, H, W)
            class_idx: 目标类别索引，为 None 时使用预测最高置信度类别
            
        Returns:
            显著性图 (C, H, W)，已归一化到 [0, 1]
        """
        try:
            # 启用 Guided Backpropagation
            self._enable()

            # 将输入移到与模型相同的设备。
            # 若设备不同，Layer.forward 内 inputs.to(device) 会创建新 Tensor，
            # 原 input_tensor 不进入计算图，反向传播后其 .grad 永远为 None。
            model_device = getattr(self.model, 'device', 'cpu')
            if input_tensor.device != model_device:
                input_on_device = input_tensor.to(model_device)
            else:
                input_on_device = input_tensor
            input_on_device.requires_grad = True

            # 前向传播
            output = self.model(input_on_device)

            # 确定目标类别
            if class_idx is None:
                class_idx = int(np.argmax(output.data, axis=1)[0])

            # 反向传播到输入
            target_logit = output[0, class_idx]
            target_logit.backward()

            # 获取输入的梯度作为显著性图
            if input_on_device.grad is not None:
                grad_data = input_on_device.grad.data
                # cupy array → numpy（GPU 情况）
                if hasattr(grad_data, 'get'):
                    grad_data = grad_data.get()
                saliency_map = grad_data.copy()
            else:
                raise RuntimeError("未能捕获输入梯度")
            
            # 处理多通道情况
            if saliency_map.ndim == 4:
                saliency_map = saliency_map[0]  # 取第一个样本
            
            # 只保留正值（Guided Backpropagation 的关键）
            saliency_map = np.maximum(0, saliency_map)
            
            # 归一化到 [0, 1]
            saliency_map = self._normalize(saliency_map)
            
            return saliency_map
            
        finally:
            # 禁用 Guided Backpropagation
            self._disable()
            # 清除梯度
            if hasattr(input_tensor, 'grad') and input_tensor.grad is not None:
                input_tensor.grad = None

    def _normalize(self, saliency_map: np.ndarray) -> np.ndarray:
        """
        归一化显著性图到 [0, 1]
        
        Args:
            saliency_map: 原始显著性图
            
        Returns:
            归一化后的显著性图
        """
        saliency_map = saliency_map.astype(np.float32)
        
        max_val = np.max(saliency_map)
        min_val = np.min(saliency_map)
        
        if max_val > 1e-8:
            saliency_map = (saliency_map - min_val) / (max_val - min_val)
        else:
            saliency_map = np.zeros_like(saliency_map)
        
        return saliency_map

    def generate_batch(
        self,
        input_tensors: List[Tensor],
        class_indices: Optional[List[int]] = None
    ) -> List[np.ndarray]:
        """
        批量生成显著性图
        
        Args:
            input_tensors: 输入张量列表
            class_indices: 目标类别索引列表
            
        Returns:
            显著性图列表
        """
        saliency_maps = []
        
        for i, input_tensor in enumerate(input_tensors):
            class_idx = None
            if class_indices is not None:
                class_idx = class_indices[i]
            
            saliency_map = self.generate(input_tensor, class_idx)
            saliency_maps.append(saliency_map)
        
        return saliency_maps

    def __call__(
        self,
        input_tensor: Tensor,
        class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        调用 generate 方法的便捷方式
        
        Args:
            input_tensor: 输入张量
            class_idx: 目标类别索引
            
        Returns:
            显著性图
        """
        return self.generate(input_tensor, class_idx)


class GuidedGradCAM:
    """
    Guided Grad-CAM 实现
    
    将 Grad-CAM 的粗粒度热力图与 Guided Backpropagation 的细粒度图像相乘，
    得到既保留空间细节又突出关键区域的可视化结果。
    
    公式：
    L_guided-gradcam = L_(grad-CAM)^c ⊙ L_guided-backprop
    
    其中 ⊙ 表示逐元素相乘
    """

    def __init__(self, model: Module, target_layer: Layer):
        """
        初始化 Guided Grad-CAM
        
        Args:
            model: 神经网络模型
            target_layer: Grad-CAM 的目标层（通常是最后一个卷积层）
        """
        from .gradcam import GradCAM
        
        self.gradcam = GradCAM(model, target_layer)
        self.guided_bp = GuidedBackpropagation(model)

    def generate(
        self,
        input_tensor: Tensor,
        class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        生成 Guided Grad-CAM 可视化结果
        
        Args:
            input_tensor: 输入张量 (N, C, H, W)
            class_idx: 目标类别索引，为 None 时使用预测最高置信度类别
            
        Returns:
            Guided Grad-CAM 结果 (C, H, W)，已归一化到 [0, 1]
        """
        # 生成 Grad-CAM 热力图
        heatmap = self.gradcam.generate(input_tensor, class_idx)
        
        # 生成 Guided Backpropagation 显著性图
        saliency_map = self.guided_bp.generate(input_tensor, class_idx)
        
        # 获取原始输入大小
        input_shape = input_tensor.shape
        if len(input_shape) == 4:
            _, _, input_h, input_w = input_shape
        else:
            _, input_h, input_w = input_shape
        
        # 将热力图上采样到输入大小
        zoom_factor = (input_h / heatmap.shape[0], input_w / heatmap.shape[1])
        heatmap_resized = zoom(heatmap, zoom_factor, order=1)
        
        # 逐元素相乘
        # 如果显著性图是多通道的，将热力图应用到所有通道
        if saliency_map.ndim == 3:
            guided_gradcam = saliency_map * heatmap_resized[np.newaxis, :, :]
        else:
            guided_gradcam = saliency_map * heatmap_resized
        
        # 归一化到 [0, 1]
        guided_gradcam = self._normalize(guided_gradcam)
        
        return guided_gradcam

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """归一化到 [0, 1]"""
        data = data.astype(np.float32)
        max_val = np.max(data)
        min_val = np.min(data)
        
        if max_val > 1e-8:
            data = (data - min_val) / (max_val - min_val)
        else:
            data = np.zeros_like(data)
        
        return data

    def __call__(
        self,
        input_tensor: Tensor,
        class_idx: Optional[int] = None
    ) -> np.ndarray:
        """便捷调用方式"""
        return self.generate(input_tensor, class_idx)


def create_guided_gradcam(
    model: Module,
    target_layer: Optional[Layer] = None
) -> GuidedGradCAM:
    """
    创建 Guided Grad-CAM 实例的便捷工厂函数
    
    Args:
        model: 神经网络模型
        target_layer: 目标层，如果为 None 则自动选择最后一个卷积层
        
    Returns:
        GuidedGradCAM 实例
    """
    from .gradcam import suggest_target_layer
    
    if target_layer is None:
        target_layer = suggest_target_layer(model)
    
    if target_layer is None:
        raise ValueError(
            "无法自动找到合适的卷积层。"
            "请手动指定 target_layer 参数。"
        )
    
    return GuidedGradCAM(model, target_layer)