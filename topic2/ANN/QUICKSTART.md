# Quick Start Guide

## Setup (5 minutes)

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Train the model:**
```bash
python backend/model.py
```
Wait 3-5 minutes. You should see accuracy ~97-98%.

3. **Run the app:**
```bash
cd frontend
streamlit run app.py
```

4. **Use the app:**
- Click "Load Model" in sidebar
- Upload a test image from `test_images/` folder (recommended)
- Or draw a digit on canvas
- Click recognize button

## File Structure

```
ANN/
├── backend/
│   ├── __init__.py              # Package marker
│   ├── model.py                 # TensorFlow/Keras ANN model
│   └── model_pytorch.py         # PyTorch ANN model (optional)
├── frontend/
│   └── app.py                   # Streamlit web interface
├── requirements.txt             # Python dependencies
├── README.md                    # Full documentation
├── student_handout.html         # Assignment instructions
└── QUICKSTART.md               # This file
```

## Common Commands

**Create test images:**
```bash
python create_test_images.py
```
This creates digit images (0-9) in `test_images/` folder for testing.

**Train TensorFlow model:**
```bash
python backend/model.py
```

**Train PyTorch model (optional):**
```bash
pip install torch torchvision
python backend/model_pytorch.py
```

**Test model:**
```bash
python -c "from backend.model import MNISTModel; m = MNISTModel(); m.load_model(); print('Model loaded successfully!')"
```

**Run Streamlit app:**
```bash
cd frontend && streamlit run app.py
```

## Troubleshooting

**"Module not found":**
```bash
pip install -r requirements.txt
```

**"Model file not found":**
```bash
python backend/model.py  # Train the model first
```

**"Permission denied":**
```bash
chmod +x backend/model.py
```

## Next Steps

1. Read `README.md` for detailed documentation
2. Open `student_handout.html` in browser for assignments
3. Start improving the model!
4. Deploy to Streamlit Cloud

## Quick Tips

- Model file is saved as `mnist_model.h5` (TensorFlow) or `mnist_model_pytorch.pth` (PyTorch)
- Training takes 3-5 minutes on CPU
- Expected accuracy: 97-98%
- **Use image upload for testing** (more reliable than canvas)
- Test images are in `test_images/` folder
- Images are auto-resized to 28x28 pixels
- Best format: white/bright digits on black/dark background (like MNIST)

## Need Help?

- Check `README.md` for detailed instructions
- See `student_handout.html` for learning resources
- Contact your instructor
