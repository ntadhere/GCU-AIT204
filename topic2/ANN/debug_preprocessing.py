"""
Debug script to test preprocessing outside of Streamlit
Simulates what the canvas does
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from backend.model import MNISTModel

# Create a simple test image: white "1" on black background
# Simulate canvas: 280x280 image
img = np.zeros((280, 280), dtype=np.uint8)  # Black background

# Draw a vertical line (like digit "1")
img[50:230, 135:145] = 255  # White vertical line

print("Created test image (280x280):")
print(f"  Min: {img.min()}, Max: {img.max()}, Mean: {img.mean():.2f}")
print(f"  White pixels: {np.sum(img == 255)}")
print(f"  Black pixels: {np.sum(img == 0)}")
print()

# Now do the same preprocessing as the app
img_pil = Image.fromarray(img, mode='L')
print("After PIL conversion:")
print(f"  Size: {img_pil.size}, Mode: {img_pil.mode}")
print()

# Resize to 28x28
img_resized = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
img_array = np.array(img_resized)

print("After resize to 28x28:")
print(f"  Shape: {img_array.shape}")
print(f"  Min: {img_array.min()}, Max: {img_array.max()}, Mean: {img_array.mean():.2f}")
print(f"  Dtype: {img_array.dtype}")
print()

# Check pixel distribution
print("Pixel value distribution:")
unique, counts = np.unique(img_array, return_counts=True)
print(f"  Unique values: {len(unique)}")
print(f"  Sample values: {unique[:10]}")
print()

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(img, cmap='gray', vmin=0, vmax=255)
axes[0].set_title('Original 280x280')
axes[0].axis('off')

axes[1].imshow(img_array, cmap='gray', vmin=0, vmax=255)
axes[1].set_title('Resized 28x28')
axes[1].axis('off')

axes[2].imshow(img_array / 255.0, cmap='gray', vmin=0, vmax=1)
axes[2].set_title('Normalized (0-1)')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('debug_preprocessing.png')
print("Saved visualization to debug_preprocessing.png")
print()

# Now test with the model
print("Testing with model...")
model = MNISTModel()
if model.load_model():
    predicted, confidence = model.predict(img_array)
    print(f"Predicted: {predicted}")
    print(f"Confidence: {confidence[predicted]:.2%}")
    print()
    print("All confidences:")
    for i in range(10):
        print(f"  {i}: {confidence[i]:.2%}")
else:
    print("Model not found!")

# Compare with actual MNIST "1"
print("\n" + "="*50)
print("Comparing with actual MNIST data:")
from tensorflow import keras
(x_train, y_train), _ = keras.datasets.mnist.load_data()

# Find a "1" in MNIST
idx = np.where(y_train == 1)[0][0]
mnist_one = x_train[idx]

print(f"MNIST digit '1' (sample {idx}):")
print(f"  Min: {mnist_one.min()}, Max: {mnist_one.max()}, Mean: {mnist_one.mean():.2f}")

# Test it
predicted_mnist, conf_mnist = model.predict(mnist_one)
print(f"  Model prediction: {predicted_mnist}")
print(f"  Confidence: {conf_mnist[predicted_mnist]:.2%}")

# Show side by side
fig2, axes2 = plt.subplots(1, 2, figsize=(8, 4))
axes2[0].imshow(img_array, cmap='gray')
axes2[0].set_title(f'Our preprocessed (predicts: {predicted})')
axes2[0].axis('off')

axes2[1].imshow(mnist_one, cmap='gray')
axes2[1].set_title(f'MNIST sample (predicts: {predicted_mnist})')
axes2[1].axis('off')

plt.tight_layout()
plt.savefig('comparison.png')
print("\nSaved comparison to comparison.png")
