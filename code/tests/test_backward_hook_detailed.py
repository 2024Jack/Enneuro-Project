"""
反向钩子详细调试脚本
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


def test_detailed_backward_hook():
    """详细测试反向钩子的注册和触发"""
    print("=" * 70)
    print("反向钩子详细调试")
    print("=" * 70)

    model = SimpleCNN()
    input_data = Tensor(np.random.randn(1, 1, 10, 10).astype(np.float32))

    print("\n1. 检查 model.conv1 是否支持 register_backward_hook...")
    if hasattr(model.conv1, 'register_backward_hook'):
        print("   ✓ model.conv1 支持 register_backward_hook")
    else:
        print("   ✗ model.conv1 不支持 register_backward_hook")
        return

    print("\n2. 注册反向钩子...")
    backward_hook_called = {'count': 0, 'args': None}

    def backward_hook(grad_inputs, grad_outputs):
        backward_hook_called['count'] += 1
        backward_hook_called['args'] = (grad_inputs, grad_outputs)
        print(f"\n   ✓✓✓ 反向钩子被调用！")
        print(f"   调用次数: {backward_hook_called['count']}")
        print(f"   grad_inputs: {grad_inputs}")
        print(f"   grad_outputs: {grad_outputs}")

    handle = model.conv1.register_backward_hook(backward_hook)
    print("   ✓ 反向钩子已注册")

    print("\n3. 检查 _hook_manager...")
    if hasattr(model.conv1, '_hook_manager'):
        print(f"   ✓ _hook_manager 存在")
        print(f"   前向钩子数量: {len(model.conv1._hook_manager._forward_hooks)}")
        print(f"   反向钩子数量: {len(model.conv1._hook_manager._backward_hooks)}")
    else:
        print("   ✗ _hook_manager 不存在")

    print("\n4. 执行前向传播...")
    output = model(input_data)
    print(f"   ✓ 前向传播完成，输出形状: {output.shape}")

    print("\n5. 执行反向传播...")
    print(f"   output[0, 0].backward()")
    output[0, 0].backward()
    print(f"   ✓ 反向传播完成")

    print(f"\n6. 反向钩子调用结果:")
    print(f"   调用次数: {backward_hook_called['count']}")

    if backward_hook_called['count'] == 0:
        print("\n   ✗ 反向钩子从未被调用")
        print("\n   可能原因:")
        print("   1. Function.backward 方法没有被正确调用")
        print("   2. 反向钩子注册到了错误的对象")
        print("   3. hooks 模块的 monkey patch 没有生效")

        print("\n   调试信息:")
        from eneuro.base.core import Function
        print(f"   Function.backward: {Function.backward}")
        print(f"   Function 是否在导入时已修改: {'with_hooks' in str(Function.backward)}")
    else:
        print("\n   ✓ 反向钩子工作正常！")


if __name__ == "__main__":
    test_detailed_backward_hook()
