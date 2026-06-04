"""
详细调试：追踪反向钩子的执行流程
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from eneuro.base import Tensor
from eneuro.nn.module import Conv2d, Module
from eneuro.base import functions as F
from eneuro.utils.hooks import HookRegistry


class SimpleModel(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(out_channels=8, kernel_size=3, stride=1, pad=1, in_channels=3)
        self.F = F

    def forward(self, x):
        x = self.conv1(x)
        x = self.F.relu(x)
        return x


def test_detailed():
    print("=" * 70)
    print("详细调试：反向钩子执行流程")
    print("=" * 70)

    model = SimpleModel()
    input_data = Tensor(np.random.randn(1, 3, 16, 16).astype(np.float32))

    print("\n1. 注册反向钩子...")
    hook_called = {'count': 0}

    def my_hook(grad_inputs, grad_outputs):
        hook_called['count'] += 1
        print(f"\n   ✓✓✓ 反向钩子被调用！次数: {hook_called['count']}")
        print(f"   grad_inputs: {grad_inputs}")
        print(f"   grad_outputs: {grad_outputs}")

    handle = model.conv1.register_backward_hook(my_hook)
    print(f"   ✓ 反向钩子已注册")
    print(f"   ✓ model.conv1._hook_manager._backward_hooks: {len(model.conv1._hook_manager._backward_hooks)}")

    print("\n2. 执行前向传播...")
    output = model(input_data)
    print(f"   ✓ 输出形状: {output.shape}")

    print("\n3. 检查 HookRegistry 映射...")
    hook_registry = HookRegistry()
    print(f"   ✓ 映射数量: {len(hook_registry._function_to_layer)}")
    for func, layer_ref in hook_registry._function_to_layer.items():
        layer = layer_ref()
        print(f"     - Function: {type(func).__name__}, Layer: {type(layer).__name__ if layer else 'None'}")

    print("\n4. 检查 output.creator...")
    print(f"   ✓ output.creator: {output.creator}")
    print(f"   ✓ output.creator 类型: {type(output.creator).__name__}")

    print("\n5. 执行反向传播...")
    target = output[0, 0, 0, 0]
    print(f"   ✓ target: {target}")
    print(f"   ✓ target.creator: {target.creator}")

    target.backward()
    print(f"   ✓ 反向传播完成")

    print(f"\n6. 结果...")
    print(f"   ✓ 反向钩子被调用次数: {hook_called['count']}")

    if hook_called['count'] == 0:
        print("\n   ✗ 反向钩子未被调用！")
        print("\n   问题分析:")
        print("   1. 检查 target.creator 是否在 HookRegistry 中")
        if target.creator in hook_registry._function_to_layer:
            print("      ✓ target.creator 在 HookRegistry 中")
        else:
            print("      ✗ target.creator 不在 HookRegistry 中")
            print("        这就是问题所在！反向传播经过的 Function 没有被映射到 Layer")


if __name__ == "__main__":
    test_detailed()
