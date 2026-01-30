# # streamlit_app.py
# import json, joblib, numpy as np, torch, torch.nn as nn
# import streamlit as st

# st.set_page_config(page_title="20 Newsgroups Classifier", layout="wide")

# @st.cache_resource  # load once per process
# def load_resources():
#     vectorizer = joblib.load("vectorizer.pkl")
#     with open("label_names.json") as f:
#         label_names = json.load(f)

#     class NewsMLP(nn.Module):
#         def __init__(self, input_dim, num_classes):
#             super().__init__()
#             self.net = nn.Sequential(
#                 nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.0),
#                 nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.0),
#                 nn.Linear(256, num_classes)
#             )
#         def forward(self, x): return self.net(x)

#     model = NewsMLP(input_dim=vectorizer.max_features or len(vectorizer.vocabulary_), 
#                     num_classes=len(label_names))
#     model.load_state_dict(torch.load("model_state_dict.pt", map_location="cpu"))
#     model.eval()
#     return vectorizer, label_names, model

# vectorizer, label_names, model = load_resources()

# st.title("20 Newsgroups Text Classifier (PyTorch + TF-IDF)")
# st.caption("Enter text, get predicted topic with probabilities.")

# with st.form("predict"):
#     text = st.text_area("Paste text", height=200, placeholder="Type or paste an email/article…")
#     submitted = st.form_submit_button("Classify")

# def predict(texts):
#     X = vectorizer.transform(texts).toarray()  # small batch OK; stays CPU
#     with torch.no_grad():
#         logits = model(torch.tensor(X, dtype=torch.float32))
#         probs = torch.softmax(logits, dim=1).numpy()
#         preds = probs.argmax(axis=1)
#     return preds, probs

# if submitted:
#     if not text.strip():
#         st.warning("Please enter some text.")
#     else:
#         pred, probs = predict([text])
#         label = label_names[int(pred[0])]
#         st.subheader(f"Prediction: {label}")
#         # Show top-5 classes
#         top = np.argsort(-probs[0])[:5]
#         st.write({label_names[i]: float(probs[0][i]) for i in top})

# streamlit_app.py
# Fixed version with proper path handling

import json
import joblib
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
import os
from pathlib import Path

st.set_page_config(page_title="20 Newsgroups Classifier", layout="wide")

@st.cache_resource
def load_resources():
    """
    Load model resources with proper path handling
    """
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Define file paths
    vectorizer_path = script_dir / "vectorizer.pkl"
    labels_path = script_dir / "label_names.json"
    model_path = script_dir / "model_state_dict.pt"
    
    # Debug: Show what directory we're in
    st.write(f"🔍 Script directory: {script_dir}")
    st.write(f"🔍 Files in directory: {list(script_dir.glob('*'))}")
    
    # Check if files exist
    if not vectorizer_path.exists():
        st.error(f"❌ vectorizer.pkl not found at: {vectorizer_path}")
        st.error(f"Current directory: {os.getcwd()}")
        st.error(f"Files in current dir: {os.listdir('.')}")
        st.stop()
    
    if not labels_path.exists():
        st.error(f"❌ label_names.json not found at: {labels_path}")
        st.stop()
    
    if not model_path.exists():
        st.error(f"❌ model_state_dict.pt not found at: {model_path}")
        st.stop()
    
    try:
        # Load vectorizer
        vectorizer = joblib.load(str(vectorizer_path))
        st.success(f"✅ Loaded vectorizer from {vectorizer_path}")
        
        # Load label names
        with open(labels_path) as f:
            label_names = json.load(f)
        st.success(f"✅ Loaded {len(label_names)} labels")
        
        # Define model architecture (2 hidden layers: 512, 256)
        class NewsMLP(nn.Module):
            def __init__(self, input_dim, num_classes):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, 512)
                self.relu1 = nn.ReLU()
                self.dropout1 = nn.Dropout(0.0)
                
                self.fc2 = nn.Linear(512, 256)
                self.relu2 = nn.ReLU()
                self.dropout2 = nn.Dropout(0.0)
                
                self.fc3 = nn.Linear(256, num_classes)
            
            def forward(self, x):
                x = self.fc1(x)
                x = self.relu1(x)
                x = self.dropout1(x)
                
                x = self.fc2(x)
                x = self.relu2(x)
                x = self.dropout2(x)
                
                x = self.fc3(x)
                return x
        
        # Create model
        input_dim = vectorizer.max_features or len(vectorizer.vocabulary_)
        model = NewsMLP(input_dim=input_dim, num_classes=len(label_names))
        
        # Load weights
        model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
        model.eval()
        st.success(f"✅ Model loaded successfully!")
        
        return vectorizer, label_names, model
    
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        import traceback
        st.error(traceback.format_exc())
        st.stop()

# Load resources
vectorizer, label_names, model = load_resources()

# UI
st.title("🤖 20 Newsgroups Text Classifier")
st.caption("Enter text, get predicted topic with probabilities.")

with st.form("predict"):
    text = st.text_area("Paste text", height=200, 
                       placeholder="Type or paste an email/article…",
                       help="Enter any text and the model will classify it into one of 20 newsgroup categories")
    submitted = st.form_submit_button("🔍 Classify", use_container_width=True)

def predict(texts):
    """Make prediction on input text"""
    X = vectorizer.transform(texts).toarray()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).numpy()
        preds = probs.argmax(axis=1)
    return preds, probs

if submitted:
    if not text.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        with st.spinner("Classifying..."):
            pred, probs = predict([text])
            label = label_names[int(pred[0])]
            confidence = probs[0][pred[0]]
            
            # Display result
            st.markdown("---")
            st.markdown(f"### 🎯 Prediction: **{label}**")
            st.markdown(f"**Confidence:** {confidence:.2%}")
            
            # Show top-5 classes
            st.markdown("#### 📊 Top 5 Predictions:")
            top = np.argsort(-probs[0])[:5]
            
            for i, idx in enumerate(top, 1):
                prob = probs[0][idx]
                label_name = label_names[idx]
                
                # Create progress bar
                col1, col2, col3 = st.columns([0.5, 3, 1])
                with col1:
                    st.write(f"**{i}.**")
                with col2:
                    st.write(f"**{label_name}**")
                    st.progress(float(prob))
                with col3:
                    st.write(f"{prob:.1%}")

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This app classifies text into one of 20 newsgroup categories using:
    - **TF-IDF** vectorization
    - **Neural Network** with 2 hidden layers (512→256)
    - **PyTorch** framework
    """)
    
    st.header("📚 Categories")
    with st.expander("Show all 20 categories"):
        for i, cat in enumerate(label_names, 1):
            st.write(f"{i}. {cat}")
    
    st.header("🎯 Try Examples")
    if st.button("Space/NASA", use_container_width=True):
        st.session_state.example_text = "NASA announced a new mission to Mars with advanced rovers."
    if st.button("Computer/Graphics", use_container_width=True):
        st.session_state.example_text = "The new graphics card offers incredible ray tracing performance."
    if st.button("Sports/Baseball", use_container_width=True):
        st.session_state.example_text = "The team won the championship with a walk-off home run."

# Load example if button clicked
if 'example_text' in st.session_state:
    st.info(f"💡 Example loaded! Click 'Classify' above to see results.")