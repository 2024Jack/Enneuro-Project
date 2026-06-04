"""
钩子系统功能验证测试
创建简单网络并测试钩子获取数据的能力
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from eneuro.base import Tensor
from eneuro.nn.module import Conv2d, Linear, Sequential, Module
from eneuro.base import functions as F
from eneuro.utils import capture_features


class SimpleCNN(Module):
    """一个简单的卷积神经网络"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 卷积层
        self.conv1 = Conv2d(out_channels=8, kernel_size=3, stride=1, pad=1)
        self.conv2 = Conv2d(out_channels=16, kernel_size=3, stride=1, pad=1)
        
        # 全连接层
        self.fc1 = Linear(128)
        self.fc2 = Linear(num_classes)
        
        # 保存函数引用
        self.F = F
    
    def forward(self, x):
        # 卷积块 1
        x = self.conv1(x)
        x = self.F.relu(x)
        x = self.F.pooling(x, kernel_size=2, stride=2)
        
        # 卷积块 2
        x = self.conv2(x)
        x = self.F.relu(x)
        x = self.F.pooling(x, kernel_size=2, stride=2)
        
        # 展平
        x = self.F.flatten(x)
        
        # 全连接层
        x = self.F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


def test_hook_data_capture():
    """测试钩子能否成功捕获网络各层的特征图"""
    print("=" * 60)
    print("测试：钩子数据捕获能力")
    print("=" * 60)
    
    # 创建模型
    model = SimpleCNN(num_classes=10)
    print("创建简单CNN模型成功")
    print(f"模型结构: conv1 → conv2 → fc1 → fc2")
    
    # 创建测试输入 (batch_size=2, channels=1, height=32, width=32)
    input_data = Tensor(np.random.randn(2, 1, 32, 32).astype(np.float32))
    print(f"\n输入数据形状: {input_data.shape}")
    
    # 为每个卷积层注册钩子
    hooks_info = []
    print("\n为各卷积层注册钩子...")
    
    # conv1 钩子
    handle1, storage1 = capture_features(model.conv1)
    hooks_info.append({
        'name': 'conv1',
        'handle': handle1,
        'storage': storage1
    })
    print(f"  ✓ conv1 钩子已注册")
    
    # conv2 钩子
    handle2, storage2 = capture_features(model.conv2)
    hooks_info.append({
        'name': 'conv2', 
        'handle': handle2,
        'storage': storage2
    })
    print(f"  ✓ conv2 钩子已注册")
    
    # 执行前向传播
    print("\n执行前向传播...")
    output = model(input_data)
    print(f"输出形状: {output.shape}")
    
    # 检查捕获的数据
    print("\n" + "=" * 60)
    print("钩子捕获结果检查")
    print("=" * 60)
    
    all_passed = True
    for info in hooks_info:
        name = info['name']
        storage = info['storage']
        
        print(f"\n--- {name} 层 ---")
        
        # 检查输入
        if storage['input'] is not None:
            input_shape = storage['input'][0].shape
            print(f"  输入数据: ✓ 已捕获")
            print(f"    形状: {input_shape}")
        else:
            print(f"  输入数据: ✗ 未捕获")
            all_passed = False
            
        # 检查输出（特征图）
        if storage['output'] is not None:
            output_shape = storage['output'].shape
            print(f"  输出特征图: ✓ 已捕获")
            print(f"    形状: {output_shape}")
            print(f"    通道数: {output_shape[1]}")
            print(f"    空间尺寸: {output_shape[2]}x{output_shape[3]}")
            
            # 检查数据是否有效
            if np.any(storage['output']):
                print(f"    数据值: ✓ 非零数据")
            else:
                print(f"    数据值: ⚠ 全零数据")
                
            # 显示部分统计信息
            print(f"    数据统计:")
            print(f"      最大值: {np.max(storage['output']):.4f}")
            print(f"      最小值: {np.min(storage['output']):.4f}")
            print(f"      平均值: {np.mean(storage['output']):.4f}")
        else:
            print(f"  输出特征图: ✗ 未捕获")
            all_passed = False
            
        # 移除钩子
        info['handle'].remove()
        print(f"  钩子已移除")
    
    # 验证输出
    print("\n" + "=" * 60)
    print("输出验证")
    print("=" * 60)
    print(f"模型输出形状: {output.shape}")
    print(f"输出类别数: {output.shape[1]}")
    
    # 总结
    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)
    if all_passed:
        print("✓ 所有测试通过！")
        print("\n钩子系统工作正常，可以成功捕获:")
        print("  - 各层的输入数据")
        print("  - 各层的输出特征图")
        print("  - 特征图的完整数值信息")
    else:
        print("✗ 部分测试失败")
        
    return all_passed


def test_multiple_inferences():
    """测试多次前向传播时钩子的行为"""
    print("\n\n" + "=" * 60)
    print("测试：多次前向传播")
    print("=" * 60)
    
    model = SimpleCNN(num_classes=10)
    
    # 注册钩子
    handle, storage = capture_features(model.conv1)
    
    # 第一次前向传播
    input1 = Tensor(np.random.randn(1, 1, 32, 32).astype(np.float32))
    output1 = model(input1)
    features1 = storage['output'].copy()
    print(f"第一次前向传播: 特征图形状 {features1.shape}")
    
    # 第二次前向传播（不同输入）
    input2 = Tensor(np.random.randn(1, 1, 32, 32).astype(np.float32))
    output2 = model(input2)
    features2 = storage['output'].copy()
    print(f"第二次前向传播: 特征图形状 {features2.shape}")
    
    # 验证两次特征图不同
    diff = np.abs(features1 - features2).sum()
    if diff > 1e-5:
        print(f"✓ 两次特征图不同（差异: {diff:.4f}）")
    else:
        print(f"✗ 两次特征图相同（差异: {diff:.4f}）")
    
    handle.remove()
    return diff > 1e-5


if __name__ == "__main__":
    print("EnNeuro 钩子系统数据捕获测试\n")
    
    # 运行测试
    test1_passed = test_hook_data_capture()
    test2_passed = test_multiple_inferences()
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    print(f"1. 钩子数据捕获: {'✓ 通过' if test1_passed else '✗ 失败'}")
    print(f"2. 多次前向传播: {'✓ 通过' if test2_passed else '✗ 失败'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 所有测试通过！钩子系统工作正常！")
        sys.exit(0)
    else:
        print("\n❌ 部分测试失败")
        sys.exit(1)
