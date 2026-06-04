"""
反向钩子终极调试脚本
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


def test_debug_backward_hooks():
    """终极调试：反向钩子"""
    print("=" * 70)
    print("终极调试：反向钩子")
    print("=" * 70)

    model = SimpleCNN()
    input_data = Tensor(np.random.randn(1, 1, 10, 10).astype(np.float32))

    print("\n1. 注册反向钩子...")
    hook_called = {'count': 0}

    def my_backward_hook(grad_inputs, grad_outputs):
        hook_called['count'] += 1
        print(f"\n   ✓✓✓ 反向钩子被调用！次数: {hook_called['count']}")
        print(f"   grad_outputs 数量: {len(grad_outputs) if grad_outputs else 0}")
        if grad_outputs and len(grad_outputs) > 0:
            print(f"   grad_outputs[0] 类型: {type(grad_outputs[0])}")
            if hasattr(grad_outputs[0], 'data'):
                print(f"   grad_outputs[0].data 形状: {grad_outputs[0].data.shape}")
        print(f"   grad_inputs: {grad_inputs}")

    handle = model.conv1.register_backward_hook(my_backward_hook)
    print(f"   ✓ 反向钩子已注册到 model.conv1")
    print(f"   ✓ model.conv1._hook_manager._backward_hooks: {len(model.conv1._hook_manager._backward_hooks)}")

    print("\n2. 执行前向传播...")
    output = model(input_data)
    print(f"   ✓ 前向传播完成")

    print("\n3. 检查 HookRegistry...")
    hook_registry = HookRegistry()
    print(f"   ✓ HookRegistry 映射数量: {len(hook_registry._function_to_layer)}")
    for func, layer_ref in hook_registry._function_to_layer.items():
        print(f"     - Function: {type(func).__name__}, Layer: {type(layer_ref()).__name__}")
        if type(layer_ref()).__name__ == 'Conv2d':
            print(f"       ✓ 找到了 Conv2d Function!")

    print("\n4. 执行反向传播...")
    output.backward()
    print(f"   ✓ 反向传播完成")

    print(f"\n5. 结果:")
    print(f"   ✓ 反向钩子被调用次数: {hook_called['count']}")

    if hook_called['count'] > 0:
        print("\n   ✓✓✓ 反向钩子工作正常！")
        return True
    else:
        print("\n   ✗✗✗ 反向钩子未被调用！")
        print("\n   可能原因分析:")
        print("   1. 反向传播没有经过 Conv2d Function")
        print("   2. HookRegistry 映射错误")
        print("   3. Function.backward 方法没有被正确调用")
        return False


if __name__ == "__main__":
    test_debug_backward_hooks()
