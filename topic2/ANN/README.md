# MNIST Handwritten Digit Recognition App

A Streamlit web application that recognizes handwritten digits using an Artificial Neural Network (ANN) trained on the MNIST dataset.

## Project Structure

```
ANN/
├── backend/
│   ├── __init__.py
│   └── model.py          # ANN model for digit recognition
├── frontend/
│   └── app.py            # Streamlit web interface
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── student_handout.html # Student assignment handout
```

## Features

- **Draw Mode**: Draw digits directly on a canvas
- **Upload Mode**: Upload images of handwritten digits
- **Real-time Prediction**: Get instant predictions with confidence scores
- **Separate Backend/Frontend**: Clean architecture with ML logic separated from UI

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- TensorFlow (deep learning framework)
- Streamlit (web app framework)
- NumPy (numerical computations)
- Pillow (image processing)
- streamlit-drawable-canvas (drawing interface)

## Training the Model

### Step 1: Navigate to Project Directory

```bash
cd /Users/isac/Desktop/GCU\ COURSES/AIT-204/ANN
```

### Step 2: Run the Training Script

```bash
python backend/model.py
```

**What happens during training:**
1. Downloads the MNIST dataset (60,000 training images, 10,000 test images)
2. Builds the ANN architecture with 4 layers
3. Trains for 10 epochs using the training data
4. Validates performance on test data
5. Saves the trained model as `mnist_model.h5`

**Expected Output:**
```
Building and training ANN model...
Model Architecture: Input(784) -> Dense(512) -> Dense(256) -> Dense(128) -> Output(10)

Epoch 1/10
422/422 [==============================] - 3s 6ms/step - loss: 0.2891 - accuracy: 0.9156 - val_loss: 0.1345 - val_accuracy: 0.9605
...
Epoch 10/10
422/422 [==============================] - 2s 5ms/step - loss: 0.0421 - accuracy: 0.9867 - val_loss: 0.0891 - val_accuracy: 0.9783

Test accuracy: 0.9761
Test loss: 0.0912

Model saved to mnist_model.h5
==================================================
Training complete!
Final Test Accuracy: 0.9761 (97.61%)
Final Test Loss: 0.0912
==================================================
```

**Training Time:** Approximately 3-5 minutes on a standard CPU

**Expected Accuracy:** 97-98%

### Step 3: Verify Model File

```bash
ls -lh mnist_model.h5
```

You should see a file around 5-10 MB in size.

## Testing the Model

### Using Test Images

Test images are provided in `test_images/` directory (digits 0-9). Upload these to verify the model works correctly.

If you don't have test images, create them:
```bash
python create_test_images.py
```

### Option 1: Using the Streamlit App (Recommended)

```bash
cd frontend
streamlit run app.py
```

1. The app will open in your browser at `http://localhost:8501`
2. Click "Load Model" in the sidebar
3. Test the model using:
   - **Image upload (recommended)**: Upload an image from `test_images/` folder
   - **Canvas drawing**: Draw a digit (note: less accurate than upload)

### Option 2: Command Line Testing

Create a test script `test_model.py`:

```python
from backend.model import MNISTModel
import numpy as np
from tensorflow import keras

# Load the model
model = MNISTModel()
model.load_model()

# Load test dataset
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()

# Test on a few random samples
for i in range(5):
    idx = np.random.randint(0, len(x_test))
    test_image = x_test[idx]
    true_label = y_test[idx]

    # Predict
    predicted_digit, confidence = model.predict(test_image)

    print(f"Sample {i+1}:")
    print(f"  True Label: {true_label}")
    print(f"  Predicted: {predicted_digit}")
    print(f"  Confidence: {confidence[predicted_digit]:.2%}")
    print()
```

Run the test:
```bash
python test_model.py
```

### Option 3: Python Interactive Testing

```python
from backend.model import MNISTModel
from tensorflow import keras
import matplotlib.pyplot as plt

# Load model and test data
model = MNISTModel()
model.load_model()
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()

# Test on first image
predicted, confidence = model.predict(x_test[0])
print(f"Predicted: {predicted}")
print(f"Confidence: {confidence[predicted]:.2%}")

# Visualize (optional)
plt.imshow(x_test[0], cmap='gray')
plt.title(f"Predicted: {predicted}")
plt.show()
```

## Model Architecture

The ANN model consists of:
- **Input Layer**: 784 neurons (flattened 28x28 pixel image)
- **Hidden Layer 1**: 512 neurons, ReLU activation, 20% dropout
- **Hidden Layer 2**: 256 neurons, ReLU activation, 20% dropout
- **Hidden Layer 3**: 128 neurons, ReLU activation, 20% dropout
- **Output Layer**: 10 neurons, Softmax activation (one per digit 0-9)

**Total Parameters:** ~670,000 trainable parameters

**Loss Function:** Categorical Cross-Entropy

**Optimizer:** Adam

## Performance Metrics

| Metric | Expected Value |
|--------|---------------|
| Training Accuracy | ~98.5% |
| Validation Accuracy | ~97.8% |
| Test Accuracy | 97-98% |
| Training Time | 3-5 minutes (CPU) |

## Troubleshooting

### Model file not found
- Ensure you ran `python backend/model.py` first
- Check that `mnist_model.h5` exists in the project root

### Import errors
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Use Python 3.8 or higher

### Low accuracy
- Retrain the model with more epochs: modify `epochs=10` to `epochs=20` in model.py
- See student handout for improvement suggestions

### Streamlit not loading model
- Ensure you're in the `frontend` directory when running streamlit
- Click "Load Model" button in the sidebar

## Additional Resources

- **Student Handout**: See `student_handout.html` for learning objectives and assignments
- **MNIST Dataset**: http://yann.lecun.com/exdb/mnist/
- **TensorFlow Documentation**: https://www.tensorflow.org/
- **Streamlit Documentation**: https://docs.streamlit.io/

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- Streamlit 1.28+
- NumPy 1.24+
- Pillow 10.0+
- streamlit-drawable-canvas 0.9.3+

## Important Notes

### Preprocessing
- Input images are automatically resized to 28×28 pixels
- The model expects grayscale images
- **Canvas format**: White strokes on black background (matches MNIST - no inversion needed!)
- **Uploaded images**: Automatically inverted if background is brighter than digit
- Model uses flattened input (784 dimensions) instead of 2D convolution
- All images normalized to 0-1 range by dividing by 255

### MNIST Format
MNIST expects:
- Bright digits (high pixel values ~255 before normalization, ~1.0 after)
- Dark background (low pixel values ~0 before normalization, ~0.0 after)
- 28×28 grayscale images, flattened to 784 dimensions for ANN input

### Common Issues
- **Wrong predictions on canvas**: Canvas preprocessing can be inconsistent. Use image upload instead.
- **Model not loading**: Ensure you trained the model first (`python backend/model.py`)
- **Need test images**: Run `python create_test_images.py` to generate test digit images
- **Upload predictions wrong**: Make sure images have bright digits on dark background (like MNIST)
