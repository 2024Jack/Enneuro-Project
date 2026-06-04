"""
Grad-CAM (Gradient-weighted Class Activation Mapping) 实现

参考论文: "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
https://arxiv.org/abs/1610.02391

设计说明：
- 完全使用 eneuro.utils.hooks 模块来捕获特征图和梯度
- GradCAM 通过 capture_features 和 capture_gradients 使用钩子功能
"""

import numpy as np
from typing import Dict, Optional, List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eneuro.base import Tensor
from eneuro.nn.module import Module, Layer
from eneuro.base import functions as F
from eneuro.utils import capture_features, capture_gradients


class GradCAM:
    """
    Grad-CAM 类激活映射计算

    核心公式：
    α_k^c = (1/Z) Σ_i Σ_j ∂Y^c / ∂A_ij^k
    L_(grad-CAM)^c = ReLU(Σ_k α_k^c · A^k)

    其中：
    - A_ij^k: 第k个卷积核在位置(i,j)的激活值
    - Y^c: 目标类别c的logits得分
    - α_k^c: 类别c关于第k个特征图的神经元重要性权重
    """

    def __init__(self, model: Module, target_layer: Layer):
        """
        初始化GradCAM

        Args:
            model: 神经网络模型
            target_layer: 目标层（通常是最后一个卷积层）
        """
        self.model = model
        self.target_layer = target_layer

        self._feature_handle = None
        self._gradient_handle = None
        self._feature_storage = None
        self._gradient_storage = None

    def _register_hooks(self):
        """注册特征图和梯度钩子"""
        if self._feature_handle is not None:
            return

        self._feature_handle, self._feature_storage = capture_features(self.target_layer)
        self._gradient_handle, self._gradient_storage = capture_gradients(self.target_layer)

    def _remove_hooks(self):
        """移除所有钩子"""
        if self._feature_handle is not None:
            self._feature_handle.remove()
            self._feature_handle = None
            self._feature_storage = None

        if self._gradient_handle is not None:
            self._gradient_handle.remove()
            self._gradient_handle = None
            self._gradient_storage = None

    def generate(
        self,
        input_tensor: Tensor,
        class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        生成Grad-CAM热力图

        Args:
            input_tensor: 输入张量 (N, C, H, W)
            class_idx: 目标类别索引，如果为None则使用预测类别

        Returns:
            热力图 (H, W)，已归一化到[0,1]
        """
        self._register_hooks()

        try:
            output = self.model(input_tensor)

            if class_idx is None:
                class_idx = int(np.argmax(output.data, axis=1)[0])

            target_logit = output[0, class_idx]
            target_logit.backward()

            activations = self._feature_storage['output']
            gradients = self._gradient_storage['grad_output']

            if activations is None:
                raise RuntimeError(
                    "未能捕获特征图。"
                    "请确保目标层正确注册了钩子。"
                )

            if gradients is None:
                raise RuntimeError(
                    "未能捕获梯度。"
                    "请确保目标层支持梯度计算。"
                )

            activations = activations[0]
            gradients = gradients[0]

            if len(gradients.shape) == 4:
                gradients = np.mean(gradients, axis=(2, 3))
            elif len(gradients.shape) == 3:
                gradients = np.mean(gradients, axis=(1, 2))

            alpha_k = self._compute_weights(gradients)

            weights_sum = sum(
                alpha_k[k] * activations[k]
                for k in range(activations.shape[0])
            )

            heatmap = F.relu(Tensor(weights_sum)).data

            heatmap = self._normalize_heatmap(heatmap)

            return heatmap

        finally:
            self._remove_hooks()

    def _compute_weights(self, gradients: np.ndarray) -> Dict[int, float]:
        """
        计算神经元重要性权重α

        α_k^c = GlobalAveragePool(∂Y^c/∂A^k) = mean(gradients[k])

        Args:
            gradients: 梯度张量 (C,) 或 (C, H, W)

        Returns:
            每个通道的权重字典
        """
        weights = {}

        for k in range(gradients.shape[0]):
            weights[k] = float(np.mean(gradients[k]))
        return weights

    def _normalize_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        """
        归一化热力图到[0,1]

        Args:
            heatmap: 原始热力图

        Returns:
            归一化后的热力图
        """
        heatmap = heatmap.astype(np.float32)

        min_val = np.min(heatmap)
        max_val = np.max(heatmap)

        if max_val - min_val > 1e-5:
            heatmap = (heatmap - min_val) / (max_val - min_val)
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap

    def generate_batch(
        self,
        input_tensors: List[Tensor],
        class_indices: Optional[List[int]] = None
    ) -> List[np.ndarray]:
        """
        批量生成Grad-CAM热力图

        Args:
            input_tensors: 输入张量列表
            class_indices: 目标类别索引列表，如果为None则使用预测类别

        Returns:
            热力图列表
        """
        heatmaps = []

        for i, input_tensor in enumerate(input_tensors):
            class_idx = None
            if class_indices is not None:
                class_idx = class_indices[i]

            heatmap = self.generate(input_tensor, class_idx)
            heatmaps.append(heatmap)

        return heatmaps

    def __call__(
        self,
        input_tensor: Tensor,
        class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        调用generate方法的便捷方式

        Args:
            input_tensor: 输入张量
            class_idx: 目标类别索引

        Returns:
            热力图
        """
        return self.generate(input_tensor, class_idx)


def get_all_conv_layers(model: Module) -> List[tuple]:
    """
    获取模型中所有卷积层

    Args:
        model: 神经网络模型

    Returns:
        卷积层列表 [(name, layer), ...]
    """
    conv_layers = []
    visited = set()

    def search_modules(module, prefix='', depth=0):
        if depth > 20 or id(module) in visited:
            return

        visited.add(id(module))

        for name, child in module.__dict__.items():
            if name.startswith('_'):
                continue

            if isinstance(child, Layer):
                module_name = type(child).__name__
                if 'conv' in module_name.lower() or 'Conv' in module_name:
                    full_name = f"{prefix}.{name}" if prefix else name
                    conv_layers.append((full_name, child))
            elif hasattr(child, '__dict__') and not name.startswith('_'):
                new_prefix = f"{prefix}.{name}" if prefix else name
                search_modules(child, new_prefix, depth + 1)

    search_modules(model)
    return conv_layers


def suggest_target_layer(model: Module) -> Optional[Layer]:
    """
    自动建议最佳的Grad-CAM目标层

    Args:
        model: 神经网络模型

    Returns:
        建议的目标层，如果找不到则返回None
    """
    conv_layers = get_all_conv_layers(model)

    if not conv_layers:
        return None

    return conv_layers[-1][1]


def create_gradcam(
    model: Module,
    target_layer: Optional[Layer] = None
) -> GradCAM:
    """
    创建GradCAM实例的便捷工厂函数

    Args:
        model: 神经网络模型
        target_layer: 目标层，如果为None则自动选择

    Returns:
        GradCAM实例
    """
    if target_layer is None:
        target_layer = suggest_target_layer(model)

    if target_layer is None:
        raise ValueError(
            "无法自动找到合适的卷积层。"
            "请手动指定 target_layer 参数。"
        )

    return GradCAM(model, target_layer)
