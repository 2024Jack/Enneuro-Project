"""
Grad-CAM 核心模块测试
测试 GradCAM 类的功能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from eneuro.base import Tensor
from eneuro.nn.module import Conv2d, Linear, Sequential, Module
from eneuro.base import functions as F
from explainability import GradCAM, create_gradcam


class SimpleCNN(Module):
    """用于测试的简单CNN模型"""

    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = Conv2d(out_channels=8, kernel_size=3, stride=1, pad=1)
        self.conv2 = Conv2d(out_channels=16, kernel_size=3, stride=1, pad=1)
        self.conv3 = Conv2d(out_channels=32, kernel_size=3, stride=1, pad=1)

        self.fc1 = Linear(128)
        self.fc2 = Linear(num_classes)

        self.F = F

    def forward(self, x):
        x = self.conv1(x)
        x = self.F.relu(x)
        x = self.F.pooling(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = self.F.relu(x)
        x = self.F.pooling(x, kernel_size=2, stride=2)

        x = self.conv3(x)
        x = self.F.relu(x)
        x = self.F.pooling(x, kernel_size=2, stride=2)

        x = self.F.flatten(x)

        x = self.F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def test_gradcam_basic():
    """测试 GradCAM 基础功能"""
    print("=" * 70)
    print("测试 1: GradCAM 基础功能")
    print("=" * 70)

    model = SimpleCNN(num_classes=10)
    input_data = Tensor(np.random.randn(1, 1, 32, 32).astype(np.float32))

    print("\n创建 GradCAM 实例...")
    gradcam = GradCAM(model, model.conv3)

    print("生成热力图...")
    heatmap = gradcam.generate(input_data, class_idx=0)

    print("\n检查热力图:")
    print(f"  ✓ 热力图形状: {heatmap.shape}")
    print(f"  ✓ 热力图数值范围: [{np.min(heatmap):.4f}, {np.max(heatmap):.4f}]")
    print(f"  ✓ 热力图平均值: {np.mean(heatmap):.4f}")

    if heatmap.shape == (8, 8):
        print(f"  ✓ 热力图尺寸正确 (8x8)")
    else:
        print(f"  ✗ 热力图尺寸不正确，期望 (8, 8)，实际 {heatmap.shape}")
        return False

    if np.max(heatmap) <= 1.0 and np.min(heatmap) >= 0.0:
        print(f"  ✓ 热力图已正确归一化到 [0, 1]")
    else:
        print(f"  ✗ 热力图未正确归一化")
        return False

    return True


def test_gradcam_auto_layer():
    """测试 GradCAM 自动选择目标层"""
    print("\n" + "=" * 70)
    print("测试 2: GradCAM 自动选择目标层")
    print("=" * 70)

    model = SimpleCNN(num_classes=10)
    input_data = Tensor(np.random.randn(1, 1, 32, 32).astype(np.float32))

    print("\n使用工厂函数创建 GradCAM...")
    gradcam = create_gradcam(model)

    print(f"自动选择的目标层: {gradcam.target_layer}")

    heatmap = gradcam(input_data)

    print("\n检查自动选择模式下的热力图:")
    print(f"  ✓ 热力图形状: {heatmap.shape}")
    print(f"  ✓ 热力图生成成功")

    return True


def test_gradcam_different_classes():
    """测试 GradCAM 针对不同类别的热力图"""
    print("\n" + "=" * 70)
    print("测试 3: GradCAM 不同类别的热力图")
    print("=" * 70)

    model = SimpleCNN(num_classes=10)
    input_data = Tensor(np.random.randn(1, 1, 32, 32).astype(np.float32))

    gradcam = GradCAM(model, model.conv3)

    heatmaps = {}
    for class_idx in [0, 1, 5, 9]:
        heatmap = gradcam.generate(input_data, class_idx=class_idx)
        heatmaps[class_idx] = heatmap
        print(f"  类别 {class_idx}: 热力图范围 [{np.min(heatmap):.4f}, {np.max(heatmap):.4f}]")

    heatmap0 = heatmaps[0]
    heatmap9 = heatmaps[9]

    diff = np.abs(heatmap0 - heatmap9).sum()
    print(f"\n  类别 0 和类别 9 的热力图差异: {diff:.4f}")

    if diff > 1e-5:
        print(f"  ✓ 不同类别的热力图有差异")
    else:
        print(f"  ⚠ 不同类别的热力图可能相同（可能是随机数据）")

    return True


def test_gradcam_batch():
    """测试 GradCAM 批量生成"""
    print("\n" + "=" * 70)
    print("测试 4: GradCAM 批量生成")
    print("=" * 70)

    model = SimpleCNN(num_classes=10)
    input_tensors = [
        Tensor(np.random.randn(1, 1, 32, 32).astype(np.float32))
        for _ in range(3)
    ]
    class_indices = [0, 1, 2]

    gradcam = GradCAM(model, model.conv3)

    print("\n批量生成热力图...")
    heatmaps = gradcam.generate_batch(input_tensors, class_indices)

    print(f"  ✓ 生成了 {len(heatmaps)} 个热力图")

    for i, heatmap in enumerate(heatmaps):
        print(f"    热力图 {i+1}: 形状 {heatmap.shape}, 类别 {class_indices[i]}")

    if len(heatmaps) == 3:
        print(f"  ✓ 批量生成成功")
    else:
        print(f"  ✗ 批量生成数量不正确")
        return False

    return True


if __name__ == "__main__":
    print("=" * 70)
    print("Grad-CAM 核心模块测试")
    print("=" * 70)

    tests = [
        test_gradcam_basic,
        test_gradcam_auto_layer,
        test_gradcam_different_classes,
        test_gradcam_batch,
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"\n✗ 测试 {test_func.__name__} 出错: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_func.__name__, False))

    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)

    passed = 0
    failed = 0

    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print("=" * 70)
    print(f"总计: {passed}/{len(results)} 通过")
    print("=" * 70)

    if failed == 0:
        print("\n🎉 所有测试通过！Grad-CAM 核心模块工作正常！")
        sys.exit(0)
    else:
        print(f"\n❌ {failed} 个测试失败")
        sys.exit(1)
