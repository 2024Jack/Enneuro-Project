"""
HookHandle 作用演示
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from eneuro.base import Tensor
from eneuro.nn.module import Conv2d
from eneuro.utils import capture_features


def demonstrate_handle_usage():
    """演示 HookHandle 的核心作用"""
    print("=" * 70)
    print("HookHandle（钩子句柄）作用演示")
    print("=" * 70)
    
    # 创建卷积层
    conv = Conv2d(out_channels=3, kernel_size=3)
    input_data = Tensor(np.random.randn(1, 1, 10, 10).astype(np.float32))
    
    # -------------------------------------------------------------
    # 演示 1：没有 handle 的问题
    # -------------------------------------------------------------
    print("\n【演示 1】如果没有 handle...")
    print("-" * 70)
    
    # ❌ 假设我们只注册钩子，没有 handle
    # （这里为了演示，我们手动模拟这个过程）
    # 在实际代码中，capture_features 总是返回 handle
    handle1, storage1 = capture_features(conv)
    
    # 前向传播
    output1 = conv(input_data)
    print(f"第一次前向传播: 钩子工作中，特征图已捕获")
    
    # 没有 handle，我们无法移除钩子！
    # handle1.remove()  # ← 这行如果不写，钩子会一直存在
    
    output2 = conv(input_data)
    print(f"第二次前向传播: 钩子仍在工作！浪费计算资源！")
    
    # 移除钩子（手动调用）
    handle1.remove()
    print(f"✓ 手动移除钩子")
    
    # -------------------------------------------------------------
    # 演示 2：使用 handle 正确管理
    # -------------------------------------------------------------
    print("\n【演示 2】使用 handle 正确管理钩子生命周期")
    print("-" * 70)
    
    # 注册钩子并获取 handle
    handle2, storage2 = capture_features(conv)
    print("✓ 钩子已注册")
    
    # 使用钩子
    output3 = conv(input_data)
    print(f"✓ 前向传播，钩子捕获了特征图")
    print(f"  特征图形状: {storage2['output'].shape}")
    
    # ★ 使用 handle 移除钩子
    handle2.remove()
    print("✓ 钩子已移除")
    
    # 验证钩子是否真的被移除了
    # （这里我们检查层的钩子管理器）
    if hasattr(conv, '_hook_manager') and len(conv._hook_manager._forward_hooks) == 0:
        print("✓ 确认：钩子已完全移除")
    else:
        print("✗ 确认：钩子未正确移除")
    
    # -------------------------------------------------------------
    # 演示 3：使用上下文管理器（with 语句）
    # -------------------------------------------------------------
    print("\n【演示 3】使用 with 语句自动管理")
    print("-" * 70)
    
    handle3, storage3 = capture_features(conv)
    
    with handle3:
        print("✓ 在 with 块内：钩子工作中")
        output4 = conv(input_data)
        print(f"  特征图已捕获: {storage3['output'].shape}")
    
    print("✓ 退出 with 块：钩子已自动移除！")
    
    # 验证
    if hasattr(conv, '_hook_manager') and len(conv._hook_manager._forward_hooks) == 0:
        print("✓ 确认：钩子已自动移除")
    
    # -------------------------------------------------------------
    # 演示 4：多个钩子的情况
    # -------------------------------------------------------------
    print("\n【演示 4】管理多个钩子")
    print("-" * 70)
    
    # 注册多个钩子
    handle_a, storage_a = capture_features(conv)
    handle_b, storage_b = capture_features(conv)
    print(f"✓ 注册了 2 个钩子")
    
    # 可以单独移除其中一个
    handle_a.remove()
    print(f"✓ 移除了钩子 A，钩子 B 仍在工作")
    
    # 移除剩下的钩子
    handle_b.remove()
    print(f"✓ 移除了钩子 B")
    
    print("\n" + "=" * 70)
    print("总结：HookHandle 的核心作用")
    print("=" * 70)
    print("1. 🎯 唯一标识：每个钩子都有唯一 ID")
    print("2. 🔄 生命周期管理：可以精确控制何时移除钩子")
    print("3. 💾 资源节约：不需要时及时移除，避免浪费内存")
    print("4. 🧩 灵活性：支持手动移除和 with 语句自动管理")
    print("5. 🎲 独立性：多个钩子可以独立管理")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_handle_usage()
