"""
Grad-CAM 全功能测试（使用 MNIST 数据集）

测试内容：
1. 模型训练与评估
2. GradCAMVisualizer 基础功能
3. 单个图像可视化
4. 多个图像对比
5. 不同层对比
6. 不同类别对比
7. 热力图叠加
"""

import sys
import os
import pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from eneuro.base import Tensor
from eneuro.nn.module import Conv2d, Linear, Module, BatchNorm
from eneuro.base import functions as F
from eneuro.nn.optim import Adam
from eneuro.nn.loss import CrossEntropyLoss
from eneuro.utils.visualization import GradCAMVisualizer


class SimpleCNN(Module):
    """Simple CNN model for MNIST"""

    def __init__(self, in_channels=1, num_classes=10):
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


def load_mnist_data(data_path=None, num_train=5000, num_test=500):
    """Load MNIST dataset"""
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), 'testdata', 'MNIST_data', 'mnist.pkl')
    
    print(f"Loading MNIST dataset from: {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    train_img = data['train_img'][:num_train]
    train_label = data['train_label'][:num_train]
    test_img = data['test_img'][:num_test]
    test_label = data['test_label'][:num_test]
    
    train_img = train_img.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    test_img = test_img.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    
    print(f"  Train set: {train_img.shape}, Labels: {train_label.shape}")
    print(f"  Test set: {test_img.shape}, Labels: {test_label.shape}")
    
    return train_img, train_label, test_img, test_label


def train_model(model, train_data, train_labels, epochs=10, batch_size=50, lr=0.002):
    """Train the model"""
    print("\n" + "=" * 70)
    print(f"Training model for {epochs} epochs")
    print("=" * 70)

    params_list = list(model.params())
    optimizer = Adam(params_list, lr=lr)
    loss_fn = CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        num_batches = 0

        indices = np.random.permutation(len(train_data))

        for i in range(0, len(train_data), batch_size):
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
        accuracy = correct / len(train_data)

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")

    print("=" * 70)
    print("Training completed!")
    print("=" * 70)


def evaluate_model(model, test_data, test_labels):
    """Evaluate model performance"""
    print("\n" + "=" * 70)
    print("Evaluating model performance")
    print("=" * 70)

    input_tensor = Tensor(test_data)
    output = model(input_tensor)

    predictions = np.argmax(output.data, axis=1)
    accuracy = np.mean(predictions == test_labels)

    print(f"\nTest accuracy: {accuracy:.4f}")

    return accuracy


def test_gradcam_visualizer():
    """Test GradCAMVisualizer functionality"""
    print("\n" + "=" * 70)
    print("Testing GradCAMVisualizer")
    print("=" * 70)

    # Load data
    train_data, train_labels, test_data, test_labels = load_mnist_data(num_train=3000, num_test=100)

    # Create and train model
    model = SimpleCNN()
    train_model(model, train_data, train_labels, epochs=5)

    # Evaluate
    accuracy = evaluate_model(model, test_data, test_labels)

    # Create visualizer
    print("\nCreating GradCAMVisualizer...")
    visualizer = GradCAMVisualizer(model)
    print(f"  Target layer: {type(visualizer.target_layer).__name__}")

    # Test 1: Single image visualization
    print("\n1. Testing single image visualization...")
    input_tensor = Tensor(test_data[0:1])
    result = visualizer.visualize(input_tensor, save_path='gradcam_single.png', show=False)
    print(f"  ✓ Predicted class: {result['predicted_class']}")
    print(f"  ✓ Heatmap shape: {result['heatmap'].shape}")
    print(f"  ✓ Saliency shape: {result['saliency'].shape}")
    print(f"  ✓ Guided Grad-CAM shape: {result['guided_gradcam'].shape}")

    # Test 2: Multiple images comparison
    print("\n2. Testing multiple images comparison...")
    input_tensors = [Tensor(test_data[i:i+1]) for i in range(5)]
    visualizer.visualize_comparison(input_tensors, save_path='gradcam_comparison.png', show=False)
    print("  ✓ Comparison visualization saved")

    # Test 3: Layer comparison
    print("\n3. Testing layer comparison...")
    visualizer.visualize_layer_comparison(input_tensor, save_path='gradcam_layers.png', show=False)
    print("  ✓ Layer comparison saved")

    # Test 4: Class comparison
    print("\n4. Testing class comparison...")
    visualizer.visualize_class_comparison(input_tensor, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                                          save_path='gradcam_classes.png', show=False)
    print("  ✓ Class comparison saved")

    # Test 5: Overlay visualization
    print("\n5. Testing overlay visualization...")
    overlay = visualizer.generate_overlay(input_tensor, save_path='gradcam_overlay.png', show=False)
    print(f"  ✓ Overlay shape: {overlay.shape}")

    print("\n" + "=" * 70)
    print("✅ All GradCAMVisualizer tests completed!")
    print("=" * 70)
    print(f"\nModel accuracy: {accuracy:.4f}")
    print("\nGenerated files:")
    print("  - gradcam_single.png: Single image visualization")
    print("  - gradcam_comparison.png: Multiple images comparison")
    print("  - gradcam_layers.png: Different layers comparison")
    print("  - gradcam_classes.png: Different classes comparison")
    print("  - gradcam_overlay.png: Grad-CAM overlay")


def test_gradcam_full():
    """Complete Grad-CAM test suite"""
    print("=" * 70)
    print("Grad-CAM Full Function Test Suite")
    print("=" * 70)

    test_gradcam_visualizer()


if __name__ == "__main__":
    test_gradcam_full()