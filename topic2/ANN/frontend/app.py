import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from backend.model import MNISTModel

# Page configuration
st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.model_loaded = False


@st.cache_resource
def load_model():
    """Load the trained model"""
    model = MNISTModel(model_path='../mnist_model.h5')
    if model.load_model():
        return model
    else:
        return None


def preprocess_canvas_image(canvas_data):
    """
    Preprocess the canvas drawing for model prediction
    Note: Canvas preprocessing can be unreliable. Upload is recommended for better results.
    """
    if canvas_data is None:
        return None

    # Get image data (RGBA format from streamlit-drawable-canvas)
    img_rgba = canvas_data.image_data

    # Extract grayscale channel
    if len(img_rgba.shape) == 3 and img_rgba.shape[2] >= 3:
        img_gray = img_rgba[:, :, 0].copy()
    else:
        img_gray = img_rgba.copy()

    # Convert to PIL Image for resizing
    img_pil = Image.fromarray(img_gray.astype('uint8'), mode='L')

    # Resize to 28x28
    img_resized = img_pil.resize((28, 28), Image.Resampling.LANCZOS)

    # Convert back to numpy array
    img_array = np.array(img_resized)

    return img_array


def preprocess_uploaded_image(uploaded_file):
    """Preprocess an uploaded image for model prediction"""
    img = Image.open(uploaded_file)

    # Convert to grayscale
    img = img.convert('L')

    # Resize to 28x28
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    # Convert to numpy array
    img_array = np.array(img)

    # Check if image needs inversion (if background is brighter than digit)
    # MNIST expects bright digits on dark background
    mean_value = np.mean(img_array)
    if mean_value > 127:  # Image has bright background
        img_array = 255 - img_array  # Invert

    return img_array


# App title and description
st.title("🔢 MNIST Handwritten Digit Recognition")
st.markdown("""
This app uses an Artificial Neural Network (ANN) to recognize handwritten digits (0-9).
You can either draw a digit or upload an image!
""")

# Sidebar
with st.sidebar:
    st.header("Model Information")

    # Load model button
    if st.button("Load Model"):
        with st.spinner("Loading model..."):
            model = load_model()
            if model:
                st.session_state.model = model
                st.session_state.model_loaded = True
                st.success("Model loaded successfully!")
            else:
                st.error("Model not found. Please train the model first by running: python backend/model.py")

    if st.session_state.model_loaded:
        st.success("✓ Model is ready")
    else:
        st.warning("⚠ Model not loaded")

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    - **Dataset**: MNIST (60K training, 10K test)
    - **Model**: ANN (Fully Connected)
    - **Architecture**: 784→512→256→128→10
    - **Input**: 28x28 grayscale images (flattened to 784)
    - **Output**: Digit (0-9)
    - **Accuracy**: ~97-98%
    """)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Draw a Digit")

    # Canvas for drawing
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    st.info("💡 Tip: Upload method is more reliable than canvas drawing")

    if st.button("Recognize Drawn Digit"):
        if not st.session_state.model_loaded:
            st.error("Please load the model first!")
        elif canvas_result.image_data is not None:
            # Preprocess image
            img_array = preprocess_canvas_image(canvas_result)

            if img_array is not None and np.sum(img_array) > 0:
                # Make prediction
                with st.spinner("Recognizing..."):
                    predicted_digit, confidence = st.session_state.model.predict(img_array)

                # Display results
                st.success(f"Predicted Digit: **{predicted_digit}**")
                st.write(f"Confidence: **{confidence[predicted_digit]:.2%}**")

                # Display confidence for all digits
                with st.expander("View all confidence scores"):
                    for digit in range(10):
                        st.write(f"{digit}: {confidence[digit]:.2%}")

                # Show preprocessed image
                with st.expander("View preprocessed image"):
                    st.image(img_array, caption="28x28 preprocessed", width=200, clamp=True)
            else:
                st.warning("Please draw a digit first!")

with col2:
    st.subheader("Upload an Image (Recommended)")

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image of a handwritten digit. White digit on black background works best."
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Recognize Uploaded Digit"):
            if not st.session_state.model_loaded:
                st.error("Please load the model first!")
            else:
                # Preprocess image
                img_array = preprocess_uploaded_image(uploaded_file)

                # Make prediction
                with st.spinner("Recognizing..."):
                    predicted_digit, confidence = st.session_state.model.predict(img_array)

                # Display results
                st.success(f"Predicted Digit: **{predicted_digit}**")
                st.write(f"Confidence: **{confidence[predicted_digit]:.2%}**")

                # Display confidence for all digits
                with st.expander("View all confidence scores"):
                    for digit in range(10):
                        st.write(f"{digit}: {confidence[digit]:.2%}")

                # Show preprocessed image
                with st.expander("View preprocessed image"):
                    st.image(img_array, caption="28x28 preprocessed", width=200, clamp=True)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and TensorFlow")
