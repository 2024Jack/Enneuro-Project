"""
反向钩子路径调试脚本
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from eneuro.base import Tensor
from eneuro.nn.module import Conv2d, Linear, Module
from eneuro.base import functions as F


class SimpleCNN(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(out_channels=8, kernel_size=3)
        self.F = F

    def forward(self, x):
        x = self.conv1(x)
        x = self.F.relu(x)
        return x


def test_backward_path():
    """测试反向传播路径"""
    print("=" * 70)
    print("反向传播路径调试")
    print("=" * 70)

    model = SimpleCNN()
    input_data = Tensor(np.random.randn(1, 1, 10, 10).astype(np.float32))

    print("\n1. 执行前向传播...")
    output = model(input_data)
    print(f"   ✓ 前向传播完成")
    print(f"   ✓ 输出形状: {output.shape}")

    print("\n2. 检查计算图...")
    func = output.creator
    print(f"   ✓ 最终的 Function: {func}")
    print(f"   ✓ Function 类型: {type(func).__name__}")

    if hasattr(func, 'inputs'):
        print(f"   ✓ Function.inputs: {func.inputs}")
        print(f"   ✓ Function.outputs: {func.outputs}")

    print("\n3. 执行反向传播并调试...")
    backward_count = {'count': 0}

    original_backward = func.__class__.backward

    def debug_backward(self, gy):
        backward_count['count'] += 1
        print(f"\n   ✓✓✓ backward 被调用！次数: {backward_count['count']}")
        print(f"   self: {self}")
        print(f"   self 类型: {type(self).__name__}")
        print(f"   gy: {gy}")
        result = original_backward(self, gy)
        print(f"   ✓✓✓ backward 完成")
        return result

    func.__class__.backward = debug_backward

    print(f"\n   临时替换 Function.backward 为调试版本")
    output[0, 0].backward()

    func.__class__.backward = original_backward

    print(f"\n4. 反向传播调试结果:")
    print(f"   ✓ backward 被调用次数: {backward_count['count']}")


if __name__ == "__main__":
    test_backward_path()
