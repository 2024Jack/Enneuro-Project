"""
反向钩子诊断脚本
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from eneuro.base import Tensor
from eneuro.nn.module import Conv2d, Linear, Module
from eneuro.base import functions as F
from eneuro.utils import HookManager


class SimpleCNN(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(out_channels=8, kernel_size=3)
        self.F = F

    def forward(self, x):
        x = self.conv1(x)
        x = self.F.relu(x)
        return x


def test_backward_hook_trigger():
    """测试反向钩子是否被触发"""
    print("=" * 70)
    print("反向钩子触发诊断")
    print("=" * 70)

    model = SimpleCNN()
    input_data = Tensor(np.random.randn(1, 1, 10, 10).astype(np.float32))

    # 创建一个自定义的钩子管理器
    hook_manager = HookManager()
    backward_hook_called = {'count': 0}

    def backward_hook(grad_inputs, grad_outputs):
        backward_hook_called['count'] += 1
        print(f"\n反向钩子被调用！次数: {backward_hook_called['count']}")
        print(f"grad_outputs: {grad_outputs}")
        print(f"grad_inputs: {grad_inputs}")

    # 注册反向钩子
    handle = model.conv1.register_backward_hook(backward_hook)
    print("✓ 反向钩子已注册")

    print("\n执行前向传播...")
    output = model(input_data)
    print(f"✓ 前向传播完成，输出形状: {output.shape}")

    print("\n执行反向传播...")
    output[0, 0].backward()
    print(f"✓ 反向传播完成")

    print(f"\n反向钩子被调用的次数: {backward_hook_called['count']}")

    if backward_hook_called['count'] > 0:
        print("✓ 反向钩子工作正常！")
        return True
    else:
        print("✗ 反向钩子从未被触发！")
        print("\n问题分析:")
        print("  - 反向钩子注册成功，但从未被调用")
        print("  - 需要在反向传播过程中触发反向钩子")
        print("  - 这需要修改 EnNeuro 框架的反向传播机制")
        return False


if __name__ == "__main__":
    test_backward_hook_trigger()
