"""
Grad-CAM 完整测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from eneuro.base import Tensor
from eneuro.nn.module import Conv2d, Linear, Sequential, Module
from eneuro.base import functions as F
from eneuro.utils import capture_features, capture_gradients


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


def test_gradient_extraction():
    """测试梯度提取功能"""
    print("=" * 70)
    print("测试：梯度提取功能")
    print("=" * 70)

    model = SimpleCNN(num_classes=10)
    input_data = Tensor(np.random.randn(1, 1, 32, 32).astype(np.float32))

    print("\n使用 GradientExtractor 捕获梯度...")
    extractor, storage = capture_gradients(model.conv3)

    print("执行前向传播...")
    output = model(input_data)
    print(f"输出形状: {output.shape}")

    print("\n执行反向传播...")
    target_logit = output[0, 0]
    target_logit.backward()

    print("\n检查捕获的梯度:")
    if 'output_grad' in storage and storage['output_grad'] is not None:
        print(f"  ✓ 输出梯度已捕获")
        print(f"    形状: {storage['output_grad'].shape}")
        print(f"    数值范围: [{np.min(storage['output_grad']):.4f}, {np.max(storage['output_grad']):.4f}]")
    else:
        print(f"  ✗ 输出梯度未捕获")
        print(f"    storage 内容: {storage}")

    extractor.remove()

    return 'output_grad' in storage and storage['output_grad'] is not None


def test_full_gradcam():
    """测试完整的 Grad-CAM 流程"""
    print("\n" + "=" * 70)
    print("测试：完整的 Grad-CAM 流程")
    print("=" * 70)

    model = SimpleCNN(num_classes=10)
    input_data = Tensor(np.random.randn(1, 1, 32, 32).astype(np.float32))

    target_layer = model.conv3

    feature_extractor, feature_storage = capture_features(target_layer)
    gradient_extractor, gradient_storage = capture_gradients(target_layer)

    print("\n执行前向传播...")
    output = model(input_data)
    print(f"  输出形状: {output.shape}")

    print("\n执行反向传播...")
    target_logit = output[0, 0]
    target_logit.backward()

    print("\n检查特征图和梯度:")
    print(f"  特征图: {'已捕获' if feature_storage['output'] is not None else '未捕获'}")
    print(f"  梯度: {'已捕获' if gradient_storage['output_grad'] is not None else '未捕获'}")

    if feature_storage['output'] is not None and gradient_storage['output_grad'] is not None:
        activations = feature_storage['output'][0]
        grads = gradient_storage['output_grad'][0]

        print(f"\n  特征图形状: {activations.shape}")
        print(f"  梯度形状: {grads.shape}")

        alpha_k = np.mean(grads, axis=(1, 2))
        print(f"\n  α 权重形状: {alpha_k.shape}")
        print(f"  α 权重范围: [{np.min(alpha_k):.4f}, {np.max(alpha_k):.4f}]")

        weights_sum = sum(
            alpha_k[k] * activations[k]
            for k in range(activations.shape[0])
        )
        print(f"\n  加权和形状: {weights_sum.shape}")

        heatmap = F.relu(Tensor(weights_sum)).data
        print(f"  热力图形状: {heatmap.shape}")
        print(f"  热力图范围: [{np.min(heatmap):.4f}, {np.max(heatmap):.4f}]")

        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
        print(f"  归一化热力图范围: [{np.min(heatmap):.4f}, {np.max(heatmap):.4f}]")

        feature_extractor.remove()
        gradient_extractor.remove()

        return True
    else:
        feature_extractor.remove()
        gradient_extractor.remove()
        return False


if __name__ == "__main__":
    test1_result = test_gradient_extraction()
    test2_result = test_full_gradcam()

    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    print(f"梯度提取测试: {'✓ 通过' if test1_result else '✗ 失败'}")
    print(f"完整 Grad-CAM 测试: {'✓ 通过' if test2_result else '✗ 失败'}")

    if test1_result and test2_result:
        print("\n🎉 所有测试通过！Grad-CAM 钩子系统工作正常！")
    else:
        print("\n❌ 部分测试失败")
