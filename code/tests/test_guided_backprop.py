"""
Guided Backpropagation 和 Guided Grad-CAM 测试

测试内容：
1. Guided Backpropagation 基础功能
2. Guided Grad-CAM 组合功能
3. 可视化对比（Grad-CAM vs Guided BP vs Guided Grad-CAM）
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
from eneuro.nn.loss import CrossEntropyLoss
from eneuro.explainability import (
    GradCAM,
    GuidedBackpropagation,
    GuidedGradCAM,
    create_guided_gradcam
)


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


def load_mnist_data(data_path=None, num_train=5000, num_test=50):
    """Load MNIST dataset"""
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), 'testdata', 'MNIST_data', 'mnist.pkl')
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    train_img = data['train_img'][:num_train]
    train_label = data['train_label'][:num_train]
    test_img = data['test_img'][:num_test]
    test_label = data['test_label'][:num_test]
    
    train_img = train_img.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    test_img = test_img.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    
    return train_img, train_label, test_img, test_label


def quick_train(model, train_data, train_labels, epochs=5, batch_size=50, lr=0.002):
    """Quick training"""
    params_list = list(model.params())
    optimizer = Adam(params_list, lr=lr)
    loss_fn = CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        indices = np.random.permutation(len(train_data))
        
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[indices[i:i+batch_size]]
            batch_labels = train_labels[indices[i:i+batch_size]]
            
            output = model(Tensor(batch_data))
            loss = loss_fn(output, batch_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += float(loss.data)
            num_batches += 1
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/num_batches:.4f}")


def test_guided_backpropagation():
    """Test Guided Backpropagation"""
    print("\n" + "=" * 70)
    print("Testing Guided Backpropagation")
    print("=" * 70)
    
    # Load data
    train_data, train_labels, test_data, test_labels = load_mnist_data(num_train=2000, num_test=10)
    
    # Create and train model
    model = SimpleCNN()
    quick_train(model, train_data, train_labels, epochs=3)
    
    # Test Guided Backpropagation
    guided_bp = GuidedBackpropagation(model)
    
    input_tensor = Tensor(test_data[0:1])
    saliency_map = guided_bp.generate(input_tensor)
    
    print(f"\nGuided Backpropagation results:")
    print(f"  Shape: {saliency_map.shape}")
    print(f"  Range: [{saliency_map.min():.4f}, {saliency_map.max():.4f}]")
    print(f"  Mean: {saliency_map.mean():.4f}")
    
    return model, test_data, test_labels, saliency_map


def test_guided_gradcam():
    """Test Guided Grad-CAM"""
    print("\n" + "=" * 70)
    print("Testing Guided Grad-CAM")
    print("=" * 70)
    
    # Load data
    train_data, train_labels, test_data, test_labels = load_mnist_data(num_train=3000, num_test=10)
    
    # Create and train model
    model = SimpleCNN()
    quick_train(model, train_data, train_labels, epochs=3)
    
    # Test all three methods
    gradcam = GradCAM(model, model.conv3)
    guided_bp = GuidedBackpropagation(model)
    guided_gradcam = GuidedGradCAM(model, model.conv3)
    
    input_tensor = Tensor(test_data[0:1])
    
    # Get prediction
    output = model(input_tensor)
    pred_label = int(np.argmax(output.data, axis=1)[0])
    print(f"\nPredicted class: {pred_label}")
    
    # Generate visualizations
    heatmap = gradcam.generate(input_tensor, class_idx=pred_label)
    saliency = guided_bp.generate(input_tensor, class_idx=pred_label)
    guided_gc = guided_gradcam.generate(input_tensor, class_idx=pred_label)
    
    print("\nGrad-CAM:")
    print(f"  Shape: {heatmap.shape}, Range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
    
    print("\nGuided Backpropagation:")
    print(f"  Shape: {saliency.shape}, Range: [{saliency.min():.4f}, {saliency.max():.4f}]")
    
    print("\nGuided Grad-CAM:")
    print(f"  Shape: {guided_gc.shape}, Range: [{guided_gc.min():.4f}, {guided_gc.max():.4f}]")
    
    # Visualize
    visualize_comparison(test_data[0], heatmap, saliency, guided_gc, pred_label)
    
    return model


def visualize_comparison(original_img, heatmap, saliency, guided_gradcam, pred_label):
    """Visualize comparison of different methods"""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    axes[0].imshow(original_img.squeeze(), cmap='gray')
    axes[0].set_title(f'Original Image\n(Predicted: {pred_label})', fontsize=14)
    axes[0].axis('off')
    
    # Grad-CAM
    im1 = axes[1].imshow(heatmap, cmap='jet', interpolation='bilinear')
    axes[1].set_title('Grad-CAM', fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Guided Backpropagation
    saliency_vis = saliency.squeeze() if saliency.ndim == 3 else saliency
    im2 = axes[2].imshow(saliency_vis, cmap='gray', interpolation='bilinear')
    axes[2].set_title('Guided Backpropagation', fontsize=14)
    axes[2].axis('off')
    
    # Guided Grad-CAM
    guided_vis = guided_gradcam.squeeze() if guided_gradcam.ndim == 3 else guided_gradcam
    im3 = axes[3].imshow(guided_vis, cmap='gray', interpolation='bilinear')
    axes[3].set_title('Guided Grad-CAM', fontsize=14)
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig('guided_gradcam_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Comparison visualization saved to: guided_gradcam_comparison.png")


def main():
    """Main function"""
    print("=" * 70)
    print("Guided Backpropagation and Guided Grad-CAM Test")
    print("=" * 70)
    
    # Test Guided Backpropagation
    print("\n1. Testing Guided Backpropagation...")
    test_guided_backpropagation()
    
    # Test Guided Grad-CAM
    print("\n2. Testing Guided Grad-CAM...")
    test_guided_gradcam()
    
    print("\n" + "=" * 70)
    print("✅ All tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
