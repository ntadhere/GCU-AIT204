"""
Test the created images with the model
"""
from PIL import Image
import numpy as np
from backend.model import MNISTModel

# Load model
model = MNISTModel()
model.load_model()

# Test each digit
print("Testing uploaded images:")
print("=" * 60)

for digit in range(10):
    img_path = f'test_images/digit_{digit}.png'

    # Load image
    img = Image.open(img_path).convert('L')

    # Resize to 28x28
    img_resized = img.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized)

    print(f"\nDigit {digit} (from {img_path}):")
    print(f"  Image stats: min={img_array.min()}, max={img_array.max()}, mean={img_array.mean():.2f}")

    # Check if needs inversion (as per upload preprocessing)
    mean_value = np.mean(img_array)
    if mean_value > 127:
        print(f"  Mean > 127, would invert")
        img_array = 255 - img_array
    else:
        print(f"  Mean <= 127, no inversion needed")

    # Predict
    predicted, confidence = model.predict(img_array)

    if predicted == digit:
        print(f"  ✓ CORRECT: Predicted {predicted} (confidence: {confidence[predicted]:.2%})")
    else:
        print(f"  ✗ WRONG: Predicted {predicted} instead of {digit} (confidence: {confidence[predicted]:.2%})")
        print(f"    True digit confidence: {confidence[digit]:.2%}")
