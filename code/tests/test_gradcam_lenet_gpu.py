"""
Grad-CAM 测试 - LeNet + GPU 支持

参考 tests/test_donkey/train.py 的结构编写
"""

import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加code目录到Python搜索路径
sys.path.append(str(Path(__file__).resolve().parent.parent))

from eneuro.base import Tensor
from eneuro.nn.module import Conv2d, Linear, Module
from eneuro.base import functions as F
from eneuro.nn.optim import Adam
from eneuro.nn.loss import CrossEntropyLoss
from eneuro.train import Trainer, Evaluator
from eneuro.data import Dataset, DataLoader
from eneuro.utils import Visualizer
from eneuro.utils.serializer import Serializer
from eneuro.explainability import GradCAM, GuidedGradCAM, suggest_target_layer
from eneuro.utils.visualization import GradCAMVisualizer


class LeNet(Module):
    """LeNet模型"""

    def __init__(self, num_classes=10, input_channels=1):
        super().__init__()
        
        self.F = F

        # 卷积层
        self.conv1 = Conv2d(out_channels=6, kernel_size=5, stride=1, in_channels=input_channels)
        self.conv2 = Conv2d(out_channels=16, kernel_size=5, stride=1)

        # 全连接层
        self.fc1 = Linear(120)
        self.fc2 = Linear(84)
        self.fc3 = Linear(num_classes)

    def forward(self, x):
        # 第一卷积块: Conv -> ReLU -> MaxPool
        x = self.conv1(x)
        x = self.F.relu(x)
        x = self.F.pooling(x, kernel_size=2, stride=2)

        # 第二卷积块: Conv -> ReLU -> MaxPool
        x = self.conv2(x)
        x = self.F.relu(x)
        x = self.F.pooling(x, kernel_size=2, stride=2)

        # 展平特征图
        x = self.F.flatten(x)

        # 全连接层
        x = self.F.relu(self.fc1(x))
        x = self.F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class MNISTDataset(Dataset):
    """MNIST数据集类"""
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def load_mnist_data(data_path=None):
    """加载MNIST数据集"""
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), 'testdata', 'MNIST_data', 'mnist.pkl')
    
    print(f"Loading MNIST dataset from: {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    train_img = data['train_img'][:5000]
    train_label = data['train_label'][:5000]
    val_img = data['train_img'][5000:6000]
    val_label = data['train_label'][5000:6000]
    test_img = data['test_img'][:1000]
    test_label = data['test_label'][:1000]
    
    train_img = train_img.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    val_img = val_img.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    test_img = test_img.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    
    return train_img, train_label, val_img, val_label, test_img, test_label


def test_gradcam(model, test_data):
    """测试Grad-CAM"""
    print("\n" + "=" * 70)
    print("Testing Grad-CAM")
    print("=" * 70)

    input_tensor = Tensor(test_data[0:1])
    
    target_layer = suggest_target_layer(model)
    print(f"Target layer: {type(target_layer).__name__}")
    
    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam.generate(input_tensor, class_idx=0)
    print(f"✓ Grad-CAM 热力图形状: {heatmap.shape}")
    
    guided_gradcam = GuidedGradCAM(model, target_layer)
    guided_result = guided_gradcam.generate(input_tensor, class_idx=0)
    print(f"✓ Guided Grad-CAM 结果形状: {guided_result.shape}")
    
    return heatmap, guided_result


def main():
    print("=" * 70)
    print("Grad-CAM Test with LeNet")
    print("参考 tests/test_donkey/train.py 结构")
    print("=" * 70)
    
    # 超参数设置
    batch_size = 64
    total_epochs = 5  # MNIST数据集较小，5轮足够
    lr = 0.001
    num_classes = 10
    
    # 创建模型保存目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_folder = os.path.join(script_dir, "gradcam_results")
    os.makedirs(save_folder, exist_ok=True)
    model_path = os.path.join(save_folder, "lenet_mnist.json")
    
    # 创建模型
    model = LeNet(num_classes=num_classes, input_channels=1)
    print(f"\nLeNet模型创建完成")
    print(f"输入尺寸: 1x28x28")
    print(f"输出类别: {num_classes}")
    
    # 加载预训练模型（如果存在）
    serializer = Serializer()
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}...")
        serializer.load(model, model_path)
        print("Model loaded successfully.")
    else:
        print("No pre-trained model found, starting training from scratch")
    
    # 创建优化器和损失函数
    optimizer = Adam(model.params(), lr=lr)
    loss_fn = CrossEntropyLoss()
    
    # 创建可视化器
    visualizer = Visualizer(num_classes=num_classes)
    
    # 加载数据
    train_img, train_label, val_img, val_label, test_img, test_label = load_mnist_data()
    
    # 创建数据集和数据加载器
    train_dataset = MNISTDataset(train_img, train_label)
    val_dataset = MNISTDataset(val_img, val_label)
    test_dataset = MNISTDataset(test_img, test_label)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    
    print(f"\n数据集统计:")
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")
    
    # 创建训练器并训练
    trainer = Trainer(model, loss_fn, optimizer, visualizer)
    
    print(f"\n开始训练 (共 {total_epochs} 轮)...")
    trainer.fit(
        train_loader,
        val_loader,
        epochs=total_epochs,
        batch_size=batch_size,
        verbose=True,
        device='cuda'
    )
    
    # 保存模型
    print(f"\nSaving model to {model_path}...")
    serializer.save(model, model_path)
    print("Model saved successfully.")
    
    # 训练曲线可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('LeNet Training on MNIST', fontsize=16)
    
    axes[0].plot(visualizer.train_loss, label='Train Loss')
    axes[0].plot(visualizer.val_loss, label='Val Loss')
    axes[0].set_title('Loss Curve')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(visualizer.train_acc, label='Train Acc')
    axes[1].plot(visualizer.val_acc, label='Val Acc')
    axes[1].set_title('Accuracy Curve')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(os.path.join(save_folder, "training_curves.png"), dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {save_folder}/training_curves.png")
    plt.close()
    
    # 模型评估
    print("\n" + "=" * 70)
    print("模型评估")
    print("=" * 70)
    
    evaluator = Evaluator(model, loss_fn, visualizer)
    test_loss, test_acc = evaluator.evaluate(
        test_loader,
        batch_size=batch_size,
        verbose=True,
        device='cuda'
    )
    
    print(f"\n测试准确率: {test_acc:.4f}")
    print(f"测试损失: {test_loss:.4f}")
    
    # Grad-CAM 测试
    test_gradcam(model, test_img)
    
    # GradCAMVisualizer 测试
    print("\n" + "=" * 70)
    print("Testing GradCAMVisualizer")
    print("=" * 70)
    
    visualizer = GradCAMVisualizer(model)
    input_tensor = Tensor(test_img[0:1])
    
    result = visualizer.visualize(input_tensor, save_path=os.path.join(save_folder, 'gradcam_single.png'), show=False)
    print(f"✓ 可视化完成，预测类别: {result['predicted_class']}")
    
    input_tensors = [Tensor(test_img[i:i+1]) for i in range(5)]
    visualizer.visualize_comparison(input_tensors, save_path=os.path.join(save_folder, 'gradcam_comparison.png'), show=False)
    print("✓ 多图像对比可视化完成")
    
    overlay = visualizer.generate_overlay(input_tensor, alpha=0.5, save_path=os.path.join(save_folder, 'gradcam_overlay.png'), show=False)
    print("✓ 热力图叠加可视化完成")
    
    print("\n" + "=" * 70)
    print("✅ 所有测试完成!")
    print("=" * 70)
    print(f"\n模型准确率: {test_acc:.4f}")
    print("\n生成的文件:")
    print(f"  - {save_folder}/lenet_mnist.json: 训练好的模型")
    print(f"  - {save_folder}/training_curves.png: 训练曲线")
    print(f"  - {save_folder}/gradcam_single.png: 单图像Grad-CAM")
    print(f"  - {save_folder}/gradcam_comparison.png: 多图像对比")
    print(f"  - {save_folder}/gradcam_overlay.png: 热力图叠加")


if __name__ == "__main__":
    main()