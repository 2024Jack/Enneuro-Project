"""
Grad-CAM 完整功能测试

功能：
1. 创建简单的 CNN 模型
2. 使用 MNIST 数据集训练
3. 使用 Grad-CAM 生成热力图
4. 可视化结果并保存

测试内容：
- 模型训练
- 不同层的 Grad-CAM 对比
- 多样本可视化
"""

import sys
import os
import pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from eneuro.base import Tensor
from eneuro.nn.module import Conv2d, Linear, Module, BatchNorm
from eneuro.base import functions as F
from eneuro.nn.optim import Adam
from eneuro.nn.loss import meanSquaredError, CrossEntropyLoss
from eneuro.explainability import GradCAM, create_gradcam


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
    
    print(f"Loading MNIST dataset: {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    train_img = data['train_img'][:num_train]
    train_label = data['train_label'][:num_train]
    test_img = data['test_img'][:num_test]
    test_label = data['test_label'][:num_test]
    
    train_img = train_img.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    test_img = test_img.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    
    print(f"   ✓ Train set: {train_img.shape}, Labels: {train_label.shape}")
    print(f"   ✓ Test set: {test_img.shape}, Labels: {test_label.shape}")
    
    return train_img, train_label, test_img, test_label


def quick_train(model, train_data, train_labels, epochs=10, batch_size=20, lr=0.001):
    """Quick training function"""
    print("\n" + "=" * 70)
    print("Starting training")
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
    print("Training completed!")
    print("=" * 70)


def visualize_gradcam(model, test_data, test_labels, target_layer, save_path="gradcam_results.png"):
    """Visualize Grad-CAM results"""
    print("\n" + "=" * 70)
    print("Generating Grad-CAM heatmaps")
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

        original_img = input_data[0].squeeze()
        original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min() + 1e-8)

        axes[i, 0].imshow(original_img, cmap='gray')
        axes[i, 0].set_title(f'Original Image\nTrue: {true_label}, Pred: {pred_label}', fontsize=12)
        axes[i, 0].axis('off')

        im1 = axes[i, 1].imshow(heatmap, cmap='jet', interpolation='bilinear', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Grad-CAM Heatmap\n(Layer: {type(target_layer).__name__})', fontsize=12)
        axes[i, 1].axis('off')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

        zoom_factor = (original_img.shape[0] / heatmap.shape[0], original_img.shape[1] / heatmap.shape[1])
        heatmap_resized = zoom(heatmap, zoom_factor, order=1)

        heatmap_colored = np.array(plt.cm.jet(heatmap_resized))[:, :, :3]

        overlay = 0.5 * original_img[:, :, np.newaxis] + 0.5 * heatmap_colored
        overlay = np.clip(overlay, 0, 1)

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title('Overlay', fontsize=12)
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {save_path}")

    plt.close()

    print("=" * 70)
    print("Grad-CAM visualization completed!")
    print("=" * 70)


def test_gradcam_on_different_layers(model, test_data, test_labels):
    """Test Grad-CAM on different layers"""
    print("\n" + "=" * 70)
    print("Testing Grad-CAM on different layers")
    print("=" * 70)

    layers_to_test = [
        ('conv1 (shallow)', model.conv1),
        ('conv2 (middle)', model.conv2),
        ('conv3 (deep)', model.conv3),
    ]

    input_data = test_data[0:1]
    input_tensor = Tensor(input_data)

    output = model(input_tensor)
    pred_label = int(np.argmax(output.data, axis=1)[0])

    print(f"\nPredicted class: {pred_label}")

    fig, axes = plt.subplots(1, len(layers_to_test), figsize=(6 * len(layers_to_test), 5))

    for idx, (layer_name, layer) in enumerate(layers_to_test):
        gradcam = GradCAM(model, layer)
        heatmap = gradcam.generate(input_tensor, class_idx=pred_label)

        print(f"\n{layer_name}:")
        print(f"  Heatmap shape: {heatmap.shape}")
        print(f"  Heatmap range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
        print(f"  Heatmap mean: {heatmap.mean():.4f}")
        print(f"  Heatmap std: {heatmap.std():.4f}")

        im = axes[idx].imshow(heatmap, cmap='jet', interpolation='bilinear', vmin=0, vmax=1)
        axes[idx].set_title(f'{layer_name}\nShape: {heatmap.shape}', fontsize=14)
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

    plt.suptitle(f'Grad-CAM on Different Layers (Predicted: {pred_label})', fontsize=16)
    plt.tight_layout()
    plt.savefig('gradcam_different_layers.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Different layers comparison saved to: gradcam_different_layers.png")
    plt.close()


def test_gradcam_different_classes(model, test_data, test_labels):
    """Test Grad-CAM for different classes"""
    print("\n" + "=" * 70)
    print("Testing Grad-CAM for different classes")
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

        print(f"\nClass {class_idx}:")
        print(f"  Heatmap range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
        print(f"  Heatmap mean: {heatmap.mean():.4f}")

        im = axes[idx].imshow(heatmap, cmap='jet', interpolation='bilinear', vmin=0, vmax=1)
        axes[idx].set_title(f'Class {class_idx}' + (' (predicted)' if class_idx == pred_label else ''), fontsize=14)
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

    plt.suptitle(f'Grad-CAM for Different Classes', fontsize=16)
    plt.tight_layout()
    plt.savefig('gradcam_different_classes.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Different classes comparison saved to: gradcam_different_classes.png")
    plt.close()


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
    print(f"Prediction distribution: {np.bincount(predictions, minlength=10)}")
    print(f"True distribution: {np.bincount(test_labels, minlength=10)}")

    return accuracy


def main():
    """Main function"""
    print("=" * 70)
    print("Grad-CAM Full Function Test (MNIST)")
    print("=" * 70)

    print("\n1. Creating model...")
    model = SimpleCNN(in_channels=1, num_classes=10)
    params_list = list(model.params())
    print(f"   ✓ Model created: {type(model).__name__}")
    print(f"   ✓ Number of parameters: {len(params_list)}")

    print("\n2. Loading MNIST data...")
    train_data, train_labels, test_data, test_labels = load_mnist_data(num_train=5000, num_test=500)

    print("\n3. Training model...")
    quick_train(model, train_data, train_labels, epochs=10, batch_size=50, lr=0.002)

    print("\n4. Evaluating model...")
    accuracy = evaluate_model(model, test_data, test_labels)

    print("\n5. Testing Grad-CAM (main visualization)...")
    visualize_gradcam(model, test_data, test_labels, model.conv3, save_path='gradcam_results_mnist.png')

    print("\n6. Testing Grad-CAM on different layers...")
    test_gradcam_on_different_layers(model, test_data, test_labels)

    print("\n7. Testing Grad-CAM for different classes...")
    test_gradcam_different_classes(model, test_data, test_labels)

    print("\n" + "=" * 70)
    print("✅ All tests completed!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - gradcam_results_mnist.png: Main Grad-CAM visualization")
    print("  - gradcam_different_layers.png: Different layers comparison")
    print("  - gradcam_different_classes.png: Different classes comparison")
    print(f"\nModel accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
