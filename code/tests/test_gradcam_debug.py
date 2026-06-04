"""
简单调试：检查 Grad-CAM 的每个步骤
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from eneuro.base import Tensor
from eneuro.nn.module import Conv2d, Linear, Module, BatchNorm
from eneuro.base import functions as F
from eneuro.explainability import GradCAM


class SimpleCNN(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(out_channels=16, kernel_size=3, stride=1, pad=1, in_channels=3)
        self.bn1 = BatchNorm(16)
        self.conv2 = Conv2d(out_channels=32, kernel_size=3, stride=1, pad=1)
        self.bn2 = BatchNorm(32)
        self.conv3 = Conv2d(out_channels=64, kernel_size=3, stride=1, pad=1)
        self.bn3 = BatchNorm(64)
        self.fc1 = Linear(128)
        self.fc2 = Linear(10)
        self.F = F

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.F.relu(x)
        x = self.F.pooling(x, kernel_size=2, stride=2)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.F.relu(x)
        x = self.F.pooling(x, kernel_size=2, stride=2)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.F.relu(x)
        x = self.F.pooling(x, kernel_size=2, stride=2)
        
        x = self.F.flatten(x)
        x = self.F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def debug_gradcam():
    print("=" * 70)
    print("Grad-CAM 调试")
    print("=" * 70)

    model = SimpleCNN()

    input_data = np.random.randn(1, 3, 32, 32).astype(np.float32)
    input_data[:, :, 8:16, 8:16] += 1.0
    input_tensor = Tensor(input_data)

    print("\n1. 测试简单的前向 + 反向...")
    output = model(input_tensor)
    target = output[0, 0]
    target.backward()
    print(f"   ✓ 前向 + 反向完成")

    print(f"\n2. 检查 conv3 的输入是否有梯度...")
    if hasattr(model.conv3, 'inputs') and model.conv3.inputs:
        for inp_ref in model.conv3.inputs:
            inp = inp_ref()
            if inp is not None:
                print(f"   ✓ 找到输入")
                if hasattr(inp, 'grad') and inp.grad is not None:
                    print(f"   ✓ 找到梯度: {inp.grad.shape}")
                    print(f"   ✓ 梯度范围: [{inp.grad.min():.4f}, {inp.grad.max():.4f}]")
                    print(f"   ✓ 梯度平均绝对值: {np.mean(np.abs(inp.grad)):.6f}")

    print("\n3. 测试 Grad-CAM 类...")
    gradcam = GradCAM(model, model.conv3)

    gradcam._register_hooks()
    output = model(input_tensor)
    target = output[0, 0]
    target.backward()

    print(f"\n   检查捕获的数据:")
    print(f"   - activations: {gradcam._feature_storage['output'] is not None}")
    if gradcam._feature_storage['output'] is not None:
        print(f"   - activations shape: {gradcam._feature_storage['output'].shape}")
    print(f"   - gradients: {gradcam._gradient_storage['grad_output'] is not None}")
    if gradcam._gradient_storage['grad_output'] is not None:
        print(f"   - gradients shape: {gradcam._gradient_storage['grad_output'].shape}")
        print(f"   - gradients range: [{gradcam._gradient_storage['grad_output'].min():.4f}, {gradcam._gradient_storage['grad_output'].max():.4f}]")

    gradcam._remove_hooks()

    print("\n4. 调用 generate()...")
    heatmap = gradcam.generate(input_tensor, class_idx=0)
    print(f"   ✓ Heatmap shape: {heatmap.shape}")
    print(f"   ✓ Heatmap range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
    print(f"   ✓ Heatmap mean: {heatmap.mean():.6f}")


if __name__ == "__main__":
    debug_gradcam()
