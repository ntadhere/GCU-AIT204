# streamlit_app.py
# FIXED - Matches the actual trained model architecture

import json
import joblib
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="20 Newsgroups Classifier", layout="wide")

@st.cache_resource
def load_resources():
    script_dir = Path(__file__).parent.absolute()
    
    # Load vectorizer
    vectorizer_path = script_dir / "vectorizer.pkl"
    vectorizer = joblib.load(str(vectorizer_path))
    
    # Load labels
    labels_path = script_dir / "label_names.json"
    with open(labels_path) as f:
        label_names = json.load(f)
    
    # Define model - MUST match training architecture!
    # The trained model uses nn.Sequential, so we use that here too
    class NewsMLP(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            # This matches the Sequential architecture from training
            self.net = nn.Sequential(
                nn.Linear(input_dim, 512),   # net.0
                nn.ReLU(),                    # net.1
                nn.Dropout(0.3),             # net.2
                nn.Linear(512, 256),          # net.3
                nn.ReLU(),                    # net.4
                nn.Dropout(0.3),             # net.5
                nn.Linear(256, num_classes)   # net.6
            )
        
        def forward(self, x):
            return self.net(x)
    
    # Create model
    input_dim = vectorizer.max_features or len(vectorizer.vocabulary_)
    model = NewsMLP(input_dim=input_dim, num_classes=len(label_names))
    
    # Load weights
    model_path = script_dir / "model_state_dict.pt"
    model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
    model.eval()
    
    return vectorizer, label_names, model

# Load resources
vectorizer, label_names, model = load_resources()

# UI
st.title("🤖 20 Newsgroups Text Classifier")
st.markdown("*Powered by PyTorch Neural Network with TF-IDF Features*")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This classifier uses:
    - **TF-IDF** vectorization (5000 features)
    - **Neural Network** with 2 hidden layers
    - **PyTorch** framework
    """)
    
    st.header("📚 Categories")
    with st.expander("Show 20 categories"):
        for i, cat in enumerate(label_names, 1):
            st.write(f"{i}. {cat}")

# Main form
st.markdown("---")
with st.form("predict_form"):
    text = st.text_area(
        "📝 Enter text to classify:",
        height=200,
        placeholder="Example: NASA announces new Mars mission with advanced rovers...",
        help="Paste any text and the model will predict which newsgroup category it belongs to"
    )
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
        st.warning("⚠️ Please enter some text to classify.")
    else:
        with st.spinner("Classifying..."):
            pred, probs = predict([text])
            predicted_idx = int(pred[0])
            predicted_label = label_names[predicted_idx]
            confidence = probs[0][predicted_idx]
            
            # Display result
            st.markdown("---")
            st.success(f"### 🎯 **{predicted_label}**")
            st.metric("Confidence", f"{confidence:.1%}")
            
            # Top 5 predictions
            st.markdown("#### 📊 Top 5 Predictions")
            top_5 = np.argsort(-probs[0])[:5]
            
            for rank, idx in enumerate(top_5, 1):
                prob = probs[0][idx]
                label = label_names[idx]
                
                col1, col2, col3 = st.columns([0.5, 4, 1])
                with col1:
                    if rank == 1:
                        st.write("🥇")
                    elif rank == 2:
                        st.write("🥈")
                    elif rank == 3:
                        st.write("🥉")
                    else:
                        st.write(f"**{rank}.**")
                with col2:
                    st.write(f"**{label}**")
                    st.progress(float(prob))
                with col3:
                    st.write(f"{prob:.1%}")
            
            # Show all probabilities in expander
            with st.expander("📈 View All Probabilities"):
                all_probs = [(label_names[i], probs[0][i]) for i in range(len(label_names))]
                all_probs.sort(key=lambda x: x[1], reverse=True)
                
                for label, prob in all_probs:
                    st.write(f"**{label}**: {prob:.4f} ({prob*100:.2f}%)")

# Sample examples
st.markdown("---")
st.markdown("### 💡 Try These Examples")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🚀 Space/Science", use_container_width=True):
        st.session_state.sample = "NASA announces a new mission to Mars with advanced rovers equipped to search for signs of ancient microbial life."

with col2:
    if st.button("💻 Computer Graphics", use_container_width=True):
        st.session_state.sample = "The new graphics card features ray tracing technology and delivers incredible performance at 4K resolution."

with col3:
    if st.button("⚾ Baseball", use_container_width=True):
        st.session_state.sample = "The team won the championship after a thrilling game with a walk-off home run in the bottom of the ninth inning."

if 'sample' in st.session_state:
    st.info(f"📋 Example loaded! Paste it in the text box above and click 'Classify'")
    st.code(st.session_state.sample)

# Footer
st.markdown("---")
st.caption("Built with PyTorch, scikit-learn, and Streamlit")