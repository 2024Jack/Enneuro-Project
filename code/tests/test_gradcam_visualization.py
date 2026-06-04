"""
Grad-CAM 完整功能测试

功能：
1. 创建简单的 CNN 模型
2. 使用合成数据快速训练
3. 使用 Grad-CAM 生成热力图
4. 可视化结果并保存

测试内容：
- 模型训练
- 不同层的 Grad-CAM 对比
- 多样本可视化
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from eneuro.base import Tensor
from eneuro.nn.module import Conv2d, Linear, Module, BatchNorm
from eneuro.base import functions as F
from eneuro.nn.optim import Adam
from eneuro.nn.loss import meanSquaredError, CrossEntropyLoss
from explainability import GradCAM, create_gradcam


class SimpleCNN(Module):
    """简单的 CNN 模型，用于快速测试"""

    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()

        self.conv1 = Conv2d(out_channels=16, kernel_size=3, stride=1, pad=1, in_channels=in_channels)
        self.bn1 = BatchNorm(16)

        self.conv2 = Conv2d(out_channels=32, kernel_size=3, stride=1, pad=1)
        self.bn2 = BatchNorm(32)

        self.conv3 = Conv2d(out_channels=64, kernel_size=3, stride=1, pad=1)
        self.bn3 = BatchNorm(64)

        self.fc1 = Linear(128)
        self.fc2 = Linear(num_classes)

        self.F = F

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.F.relu(x)
        x = self.F.pooling(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.F.relu(x)
        x = self.F.pooling(x, kernel_size=2, stride=2)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.F.relu(x)
        x = self.F.pooling(x, kernel_size=2, stride=2)

        x = self.F.flatten(x)

        x = self.F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def generate_synthetic_data(num_samples=100, img_size=32, num_classes=10):
    """生成合成数据集（带类别特征）"""
    print(f"生成合成数据集: {num_samples} 样本, 图像大小 {img_size}x{img_size}")

    data = []
    labels = []

    for i in range(num_samples):
        class_idx = i % num_classes

        img = np.random.randn(3, img_size, img_size).astype(np.float32) * 0.3

        cx = img_size // 4 + (class_idx % 3) * (img_size // 4)
        cy = img_size // 4 + (class_idx // 4) * (img_size // 4)
        radius = img_size // 6

        for c in range(3):
            for y in range(img_size):
                for x in range(img_size):
                    if (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2:
                        img[c, y, x] += 2.0

        data.append(img)
        labels.append(class_idx)

    return np.array(data), np.array(labels)


def quick_train(model, train_data, train_labels, epochs=10, batch_size=20, lr=0.001):
    """快速训练模型"""
    print("\n" + "=" * 70)
    print("开始快速训练")
    print("=" * 70)

    params_list = list(model.params())
    optimizer = Adam(params_list, lr=lr)
    num_samples = len(train_data)
    loss_fn = CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        num_batches = 0

        indices = np.random.permutation(num_samples)

        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_data = train_data[batch_indices]
            batch_labels = train_labels[batch_indices]

            input_tensor = Tensor(batch_data)

            output = model(input_tensor)

            loss = loss_fn(output, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.data)
            num_batches += 1

            predictions = np.argmax(output.data, axis=1)
            correct += np.sum(predictions == batch_labels)

        avg_loss = total_loss / num_batches
        accuracy = correct / num_samples

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")

    print("=" * 70)
    print("训练完成！")
    print("=" * 70)


def visualize_gradcam(model, test_data, test_labels, target_layer, save_path="gradcam_results.png"):
    """可视化 Grad-CAM 结果"""
    print("\n" + "=" * 70)
    print("生成 Grad-CAM 热力图")
    print("=" * 70)

    gradcam = GradCAM(model, target_layer)

    num_samples = min(6, len(test_data))

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        input_data = test_data[i:i + 1]
        true_label = test_labels[i]

        input_tensor = Tensor(input_data)

        output = model(input_tensor)
        pred_label = int(np.argmax(output.data, axis=1)[0])

        heatmap = gradcam.generate(input_tensor, class_idx=pred_label)

        original_img = input_data[0].transpose(1, 2, 0)
        original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min() + 1e-8)

        axes[i, 0].imshow(original_img)
        axes[i, 0].set_title(f'原始图像\n真实标签: {true_label}, 预测: {pred_label}', fontsize=12)
        axes[i, 0].axis('off')

        im1 = axes[i, 1].imshow(heatmap, cmap='jet', interpolation='bilinear', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Grad-CAM 热力图\n(目标层: {type(target_layer).__name__})', fontsize=12)
        axes[i, 1].axis('off')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

        zoom_factor = (original_img.shape[0] / heatmap.shape[0], original_img.shape[1] / heatmap.shape[1])
        heatmap_resized = zoom(heatmap, zoom_factor, order=1)

        heatmap_colored = np.array(plt.cm.jet(heatmap_resized))[:, :, :3]

        overlay = 0.5 * original_img + 0.5 * heatmap_colored
        overlay = np.clip(overlay, 0, 1)

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title('叠加可视化', fontsize=12)
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 可视化结果已保存到: {save_path}")

    plt.close()

    print("=" * 70)
    print("Grad-CAM 可视化完成！")
    print("=" * 70)


def test_gradcam_on_different_layers(model, test_data, test_labels):
    """测试不同层的 Grad-CAM"""
    print("\n" + "=" * 70)
    print("测试不同层的 Grad-CAM")
    print("=" * 70)

    layers_to_test = [
        ('conv1 (浅层)', model.conv1),
        ('conv2 (中层)', model.conv2),
        ('conv3 (深层)', model.conv3),
    ]

    input_data = test_data[0:1]
    input_tensor = Tensor(input_data)

    output = model(input_tensor)
    pred_label = int(np.argmax(output.data, axis=1)[0])

    print(f"\n预测类别: {pred_label}")

    fig, axes = plt.subplots(1, len(layers_to_test), figsize=(6 * len(layers_to_test), 5))

    for idx, (layer_name, layer) in enumerate(layers_to_test):
        gradcam = GradCAM(model, layer)
        heatmap = gradcam.generate(input_tensor, class_idx=pred_label)

        print(f"\n{layer_name}:")
        print(f"  热力图形状: {heatmap.shape}")
        print(f"  热力图范围: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
        print(f"  热力图平均值: {heatmap.mean():.4f}")
        print(f"  热力图标准差: {heatmap.std():.4f}")

        im = axes[idx].imshow(heatmap, cmap='jet', interpolation='bilinear', vmin=0, vmax=1)
        axes[idx].set_title(f'{layer_name}\n形状: {heatmap.shape}', fontsize=14)
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

    plt.suptitle(f'不同层的 Grad-CAM 对比 (预测类别: {pred_label})', fontsize=16)
    plt.tight_layout()
    plt.savefig('gradcam_different_layers.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ 不同层对比已保存到: gradcam_different_layers.png")
    plt.close()


def test_gradcam_different_classes(model, test_data, test_labels):
    """测试同一图像对不同类别的 Grad-CAM"""
    print("\n" + "=" * 70)
    print("测试不同类别的 Grad-CAM")
    print("=" * 70)

    input_data = test_data[0:1]
    input_tensor = Tensor(input_data)

    output = model(input_tensor)
    pred_label = int(np.argmax(output.data, axis=1)[0])

    classes_to_test = [pred_label, (pred_label + 1) % 10, (pred_label + 5) % 10]

    fig, axes = plt.subplots(1, len(classes_to_test), figsize=(6 * len(classes_to_test), 5))

    gradcam = GradCAM(model, model.conv3)

    for idx, class_idx in enumerate(classes_to_test):
        heatmap = gradcam.generate(input_tensor, class_idx=class_idx)

        print(f"\n类别 {class_idx}:")
        print(f"  热力图范围: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
        print(f"  热力图平均值: {heatmap.mean():.4f}")

        im = axes[idx].imshow(heatmap, cmap='jet', interpolation='bilinear', vmin=0, vmax=1)
        axes[idx].set_title(f'类别 {class_idx}' + (' (预测)' if class_idx == pred_label else ''), fontsize=14)
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

    plt.suptitle(f'同一图像对不同类别的 Grad-CAM', fontsize=16)
    plt.tight_layout()
    plt.savefig('gradcam_different_classes.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ 不同类别对比已保存到: gradcam_different_classes.png")
    plt.close()


def evaluate_model(model, test_data, test_labels):
    """评估模型性能"""
    print("\n" + "=" * 70)
    print("评估模型性能")
    print("=" * 70)

    input_tensor = Tensor(test_data)
    output = model(input_tensor)

    predictions = np.argmax(output.data, axis=1)
    accuracy = np.mean(predictions == test_labels)

    print(f"\n测试集准确率: {accuracy:.4f}")
    print(f"预测分布: {np.bincount(predictions, minlength=10)}")
    print(f"真实分布: {np.bincount(test_labels, minlength=10)}")

    return accuracy


def main():
    """主函数"""
    print("=" * 70)
    print("Grad-CAM 完整功能测试")
    print("=" * 70)

    print("\n1. 创建模型...")
    model = SimpleCNN(in_channels=3, num_classes=10)
    params_list = list(model.params())
    print(f"   ✓ 模型创建成功: {type(model).__name__}")
    print(f"   ✓ 参数数量: {len(params_list)}")

    print("\n2. 生成合成数据...")
    train_data, train_labels = generate_synthetic_data(num_samples=500, img_size=32, num_classes=10)
    test_data, test_labels = generate_synthetic_data(num_samples=50, img_size=32, num_classes=10)
    print(f"   ✓ 训练集: {train_data.shape}")
    print(f"   ✓ 测试集: {test_data.shape}")

    print("\n3. 快速训练模型...")
    quick_train(model, train_data, train_labels, epochs=15, batch_size=25, lr=0.002)

    print("\n4. 评估模型...")
    accuracy = evaluate_model(model, test_data, test_labels)

    print("\n5. 测试 Grad-CAM (主要可视化)...")
    visualize_gradcam(model, test_data, test_labels, model.conv3, save_path='gradcam_results.png')

    print("\n6. 测试不同层的 Grad-CAM...")
    test_gradcam_on_different_layers(model, test_data, test_labels)

    print("\n7. 测试不同类别的 Grad-CAM...")
    test_gradcam_different_classes(model, test_data, test_labels)

    print("\n" + "=" * 70)
    print("✅ 所有测试完成！")
    print("=" * 70)
    print("\n生成的文件:")
    print("  - gradcam_results.png: Grad-CAM 主要可视化结果")
    print("  - gradcam_different_layers.png: 不同层对比")
    print("  - gradcam_different_classes.png: 不同类别对比")
    print(f"\n模型准确率: {accuracy:.4f}")


if __name__ == "__main__":
    main()
