"""
简单调试：测试反向钩子是否能捕获梯度
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from eneuro.base import Tensor
from eneuro.nn.module import Conv2d, Module
from eneuro.base import functions as F
from eneuro.utils import capture_features, capture_gradients


class SimpleModel(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(out_channels=8, kernel_size=3, stride=1, pad=1, in_channels=3)
        self.F = F

    def forward(self, x):
        x = self.conv1(x)
        x = self.F.relu(x)
        return x


def test_hooks():
    print("=" * 70)
    print("测试钩子捕获")
    print("=" * 70)

    model = SimpleModel()
    input_data = Tensor(np.random.randn(1, 3, 16, 16).astype(np.float32))

    print("\n1. 注册钩子...")
    feature_handle, feature_storage = capture_features(model.conv1)
    gradient_handle, gradient_storage = capture_gradients(model.conv1)
    print("   ✓ 钩子已注册")

    print("\n2. 执行前向传播...")
    output = model(input_data)
    print(f"   ✓ 输出形状: {output.shape}")

    print("\n3. 检查特征图...")
    if feature_storage['output'] is not None:
        print(f"   ✓ 特征图已捕获: {feature_storage['output'].shape}")
    else:
        print("   ✗ 特征图未捕获")

    print("\n4. 执行反向传播...")
    target = output[0, 0, 0, 0]
    target.backward()
    print("   ✓ 反向传播完成")

    print("\n5. 检查梯度...")
    print(f"   gradient_storage: {gradient_storage}")
    if gradient_storage['grad_output'] is not None:
        print(f"   ✓ 梯度已捕获: {gradient_storage['grad_output'].shape}")
    else:
        print("   ✗ 梯度未捕获")

    print("\n6. 清理...")
    feature_handle.remove()
    gradient_handle.remove()
    print("   ✓ 钩子已移除")


if __name__ == "__main__":
    test_hooks()
