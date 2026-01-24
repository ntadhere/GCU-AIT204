"""
Quick test to verify MNIST data format
"""
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

# Load MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Check first few images
print("MNIST Data Format Check")
print("=" * 50)
print(f"Image shape: {x_train[0].shape}")
print(f"Pixel value range: {x_train[0].min()} to {x_train[0].max()}")
print(f"Data type: {x_train[0].dtype}")
print()

# Check a few samples
for i in range(5):
    img = x_train[i]
    label = y_train[i]

    # Get digit pixels (where value > 0)
    digit_pixels = img[img > 0]
    background_pixels = img[img == 0]

    print(f"Sample {i}: Label = {label}")
    print(f"  Background pixels (0): {len(background_pixels)} pixels")
    print(f"  Digit pixels (>0): {len(digit_pixels)} pixels")
    if len(digit_pixels) > 0:
        print(f"  Digit pixel values: min={digit_pixels.min()}, max={digit_pixels.max()}, mean={digit_pixels.mean():.1f}")
    print()

# Visualize first 5 digits
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    axes[i].imshow(x_train[i], cmap='gray')
    axes[i].set_title(f'Label: {y_train[i]}')
    axes[i].axis('off')
plt.tight_layout()
plt.savefig('mnist_samples.png')
print("Saved visualization to mnist_samples.png")

# Check pixel value distribution for digit "1"
idx_ones = np.where(y_train == 1)[0][:10]
print("\nDetailed check for digit '1':")
for idx in idx_ones[:3]:
    img = x_train[idx]
    print(f"  Sample {idx}: min={img.min()}, max={img.max()}, mean={img.mean():.1f}")
    print(f"    Shape: {img.shape}")
    print(f"    Non-zero pixels: {np.count_nonzero(img)}")
