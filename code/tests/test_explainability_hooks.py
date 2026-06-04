"""
测试钩子系统功能
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from eneuro.base import Tensor
from eneuro.nn.module import Conv2d, Linear, Sequential
from eneuro.utils import capture_features


def test_forward_hook_capture():
    """测试前向钩子捕获特征图功能"""
    print("=" * 50)
    print("测试 1: 前向钩子特征捕获")
    print("=" * 50)
    
    # 创建一个简单的模型
    conv1 = Conv2d(out_channels=3, kernel_size=3, stride=1, pad=1)
    conv2 = Conv2d(out_channels=5, kernel_size=3, stride=1, pad=1)
    model = Sequential(conv1, conv2)
    
    # 创建输入数据
    input_data = Tensor(np.random.randn(1, 1, 28, 28).astype(np.float32))
    
    # 为 conv1 注册钩子
    print("注册 conv1 的特征捕获钩子...")
    handle, storage = capture_features(conv1)
    
    # 前向传播
    print("执行前向传播...")
    output = model(input_data)
    
    # 检查捕获结果
    print(f"\n捕获结果:")
    print(f"  输入形状: {[x.shape for x in storage['input']]}")
    print(f"  输出形状: {storage['output'].shape}")
    
    # 移除钩子
    handle.remove()
    print("\n测试 1 完成 ✓")
    
    return True


def test_hook_handle():
    """测试钩子句柄功能"""
    print("\n" + "=" * 50)
    print("测试 2: 钩子句柄管理")
    print("=" * 50)
    
    conv = Conv2d(out_channels=2, kernel_size=3)
    
    # 测试 with 语句自动移除钩子
    print("测试 with 语句管理钩子...")
    with capture_features(conv)[0] as handle:
        input_data = Tensor(np.random.randn(1, 1, 10, 10).astype(np.float32))
        output = conv(input_data)
        print(f"  钩子在 with 语句内有效")
    
    print(f"  钩子已自动移除")
    
    print("\n测试 2 完成 ✓")
    return True


def test_multiple_hooks():
    """测试多个钩子同时工作"""
    print("\n" + "=" * 50)
    print("测试 3: 多个钩子同时工作")
    print("=" * 50)
    
    # 创建模型
    model = Sequential(
        Conv2d(out_channels=4, kernel_size=3, pad=1),
        Conv2d(out_channels=8, kernel_size=3, pad=1),
        Conv2d(out_channels=16, kernel_size=3, pad=1)
    )
    
    # 为所有层注册钩子
    storages = []
    handles = []
    
    print("为所有卷积层注册钩子...")
    for i, layer in enumerate(model.layers):
        handle, storage = capture_features(layer)
        storages.append(storage)
        handles.append(handle)
    
    # 前向传播
    input_data = Tensor(np.random.randn(1, 1, 32, 32).astype(np.float32))
    output = model(input_data)
    
    print("\n各层捕获结果:")
    for i, storage in enumerate(storages):
        print(f"  层 {i}: 输出形状 = {storage['output'].shape}")
    
    # 清理所有钩子
    for handle in handles:
        handle.remove()
    
    print("\n测试 3 完成 ✓")
    return True


if __name__ == "__main__":
    print("EnNeuro 钩子系统测试套件\n")
    
    test_results = []
    test_results.append(("前向钩子特征捕获", test_forward_hook_capture()))
    test_results.append(("钩子句柄管理", test_hook_handle()))
    test_results.append(("多个钩子同时工作", test_multiple_hooks()))
    
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    print("=" * 50)
    all_passed = True
    for name, passed in test_results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + ("所有测试通过！" if all_passed else "部分测试失败"))
    sys.exit(0 if all_passed else 1)
