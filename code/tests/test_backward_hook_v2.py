"""
反向钩子详细调试脚本 v2
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from eneuro.base import Tensor
from eneuro.nn.module import Conv2d, Linear, Module
from eneuro.base import functions as F
from eneuro.utils.hooks import HookRegistry


class SimpleCNN(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(out_channels=8, kernel_size=3)
        self.F = F

    def forward(self, x):
        x = self.conv1(x)
        x = self.F.relu(x)
        return x


def test_backward_hook_with_registry():
    """测试反向钩子和 HookRegistry"""
    print("=" * 70)
    print("反向钩子 + HookRegistry 调试")
    print("=" * 70)

    model = SimpleCNN()
    input_data = Tensor(np.random.randn(1, 1, 10, 10).astype(np.float32))

    print("\n1. 获取 HookRegistry 实例...")
    hook_registry = HookRegistry()
    print(f"   ✓ HookRegistry 实例: {hook_registry}")
    print(f"   ✓ 映射数量: {len(hook_registry._function_to_layer)}")

    print("\n2. 注册反向钩子...")
    backward_hook_called = {'count': 0, 'grad_outputs': None}

    def backward_hook(grad_inputs, grad_outputs):
        backward_hook_called['count'] += 1
        backward_hook_called['grad_outputs'] = grad_outputs
        print(f"\n   ✓✓✓ 反向钩子被调用！次数: {backward_hook_called['count']}")
        print(f"   grad_outputs: {grad_outputs}")

    handle = model.conv1.register_backward_hook(backward_hook)
    print(f"   ✓ 反向钩子已注册")
    print(f"   ✓ model.conv1._hook_manager._backward_hooks 数量: {len(model.conv1._hook_manager._backward_hooks)}")

    print("\n3. 执行前向传播...")
    output = model(input_data)
    print(f"   ✓ 前向传播完成")
    print(f"   ✓ 输出形状: {output.shape}")

    print(f"\n4. 检查 HookRegistry 映射...")
    print(f"   ✓ 映射数量: {len(hook_registry._function_to_layer)}")
    if len(hook_registry._function_to_layer) > 0:
        print(f"   ✓ 映射已建立！")
        for func, layer_ref in list(hook_registry._function_to_layer.items())[:3]:
            print(f"     - Function: {func}, Layer: {layer_ref()}")
    else:
        print(f"   ✗ 映射未建立！")

    print("\n5. 执行反向传播...")
    print(f"   output[0, 0].backward()")
    output[0, 0].backward()
    print(f"   ✓ 反向传播完成")

    print(f"\n6. 反向钩子调用结果:")
    print(f"   调用次数: {backward_hook_called['count']}")

    if backward_hook_called['count'] > 0:
        print("\n   ✓ 反向钩子工作正常！")
        return True
    else:
        print("\n   ✗ 反向钩子未被调用")
        print("\n   调试建议:")
        print("   1. 检查 HookRegistry 映射是否正确建立")
        print("   2. 检查 Function.backward 是否被调用")
        print("   3. 检查反向传播的路径")
        return False


if __name__ == "__main__":
    test_backward_hook_with_registry()
