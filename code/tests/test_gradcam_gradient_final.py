"""
反向钩子最终测试
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from eneuro.base import Tensor
from eneuro.nn.module import Conv2d, Linear, Module
from eneuro.base import functions as F
from eneuro.utils import capture_gradients


class SimpleCNN(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(out_channels=8, kernel_size=3)
        self.F = F

    def forward(self, x):
        x = self.conv1(x)
        x = self.F.relu(x)
        return x


def test_final_gradcam():
    """最终测试：使用 capture_gradients 捕获 Conv2d 的梯度"""
    print("=" * 70)
    print("最终测试：GradCAM 梯度捕获")
    print("=" * 70)

    model = SimpleCNN()
    input_data = Tensor(np.random.randn(1, 1, 10, 10).astype(np.float32), requires_grad=True)

    print("\n1. 注册梯度钩子到 conv1...")
    handle, storage = capture_gradients(model.conv1)
    print(f"   ✓ 钩子已注册")
    print(f"   storage: {storage}")

    print("\n2. 执行前向传播...")
    output = model(input_data)
    print(f"   ✓ 前向传播完成")
    print(f"   ✓ 输出形状: {output.shape}")

    print("\n3. 执行反向传播...")
    output.backward()
    print(f"   ✓ 反向传播完成")

    print("\n4. 检查捕获的梯度...")
    print(f"   grad_output: {storage['grad_output']}")
    print(f"   grad_input: {storage['grad_input']}")

    if storage['grad_output'] is not None:
        print("\n   ✓✓✓ 梯度捕获成功！")
        print(f"   梯度形状: {storage['grad_output'].shape}")
        return True
    else:
        print("\n   ✗ 梯度未捕获")
        return False


if __name__ == "__main__":
    result = test_final_gradcam()
    print("\n" + "=" * 70)
    if result:
        print("🎉 反向钩子工作正常！")
    else:
        print("❌ 反向钩子有问题")
    print("=" * 70)
