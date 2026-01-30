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

# streamlit_app.py
# DIAGNOSTIC VERSION - Shows what Streamlit sees

import json
import joblib
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
import os
from pathlib import Path

st.set_page_config(page_title="20 Newsgroups Classifier", layout="wide")

# SHOW DIAGNOSTIC INFO FIRST
st.header("🔍 Diagnostic Information")

st.write("**Current Working Directory:**")
st.code(os.getcwd())

st.write("**Script Location:**")
st.code(__file__)

st.write("**Parent Directory:**")
script_dir = Path(__file__).parent.absolute()
st.code(str(script_dir))

st.write("**Files in Current Directory:**")
st.code(str(os.listdir('.')))

st.write("**Files in Script Directory:**")
st.code(str(list(script_dir.iterdir())))

st.write("**Looking for these files:**")
files_to_find = ["vectorizer.pkl", "model_state_dict.pt", "label_names.json"]
for fname in files_to_find:
    # Try current directory
    exists_cwd = os.path.exists(fname)
    # Try script directory
    exists_script = (script_dir / fname).exists()
    
    st.write(f"- `{fname}`")
    st.write(f"  - In current dir: {'✅ YES' if exists_cwd else '❌ NO'}")
    st.write(f"  - In script dir: {'✅ YES' if exists_script else '❌ NO'}")
    
    # Try to find it
    if exists_script:
        st.success(f"Found at: {script_dir / fname}")
    elif exists_cwd:
        st.success(f"Found at: {os.path.abspath(fname)}")
    else:
        st.error(f"NOT FOUND!")

st.markdown("---")
st.header("📁 Directory Tree")

# Show directory structure
def show_tree(directory, prefix="", max_depth=3, current_depth=0):
    if current_depth >= max_depth:
        return []
    
    items = []
    try:
        paths = sorted(Path(directory).iterdir())
        for i, path in enumerate(paths):
            is_last = i == len(paths) - 1
            items.append(f"{prefix}{'└── ' if is_last else '├── '}{path.name}")
            if path.is_dir() and not path.name.startswith('.'):
                extension = "    " if is_last else "│   "
                items.extend(show_tree(path, prefix + extension, max_depth, current_depth + 1))
    except Exception as e:
        items.append(f"{prefix}[Error: {e}]")
    return items

tree = show_tree(script_dir)
st.code("\n".join(tree))

st.markdown("---")
st.info("👆 **Check the diagnostic info above to see where files are!**")

# NOW TRY TO LOAD
st.header("🔄 Attempting to Load Files...")

@st.cache_resource
def load_resources():
    script_dir = Path(__file__).parent.absolute()
    
    # Try to load from script directory
    vectorizer_path = script_dir / "vectorizer.pkl"
    labels_path = script_dir / "label_names.json"  
    model_path = script_dir / "model_state_dict.pt"
    
    st.write(f"Trying to load from: {script_dir}")
    
    # Load vectorizer
    if not vectorizer_path.exists():
        st.error(f"❌ vectorizer.pkl not found at: {vectorizer_path}")
        st.error("Files in directory:")
        st.write(list(script_dir.iterdir()))
        st.stop()
    
    vectorizer = joblib.load(str(vectorizer_path))
    st.success(f"✅ Loaded vectorizer")
    
    # Load labels
    with open(labels_path) as f:
        label_names = json.load(f)
    st.success(f"✅ Loaded {len(label_names)} labels")
    
    # Define model
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
    
    input_dim = vectorizer.max_features or len(vectorizer.vocabulary_)
    model = NewsMLP(input_dim=input_dim, num_classes=len(label_names))
    model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
    model.eval()
    st.success("✅ Model loaded!")
    
    return vectorizer, label_names, model

try:
    vectorizer, label_names, model = load_resources()
    
    st.markdown("---")
    st.success("🎉 All files loaded successfully!")
    
    # UI
    st.header("🤖 20 Newsgroups Text Classifier")
    st.caption("Enter text to classify")
    
    with st.form("predict"):
        text = st.text_area("Paste text here:", height=200)
        submitted = st.form_submit_button("🔍 Classify")
    
    def predict(texts):
        X = vectorizer.transform(texts).toarray()
        with torch.no_grad():
            logits = model(torch.tensor(X, dtype=torch.float32))
            probs = torch.softmax(logits, dim=1).numpy()
            preds = probs.argmax(axis=1)
        return preds, probs
    
    if submitted:
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            pred, probs = predict([text])
            label = label_names[int(pred[0])]
            
            st.success(f"**Prediction: {label}**")
            st.write(f"Confidence: {probs[0][pred[0]]:.2%}")
            
            st.write("**Top 5:**")
            top = np.argsort(-probs[0])[:5]
            for i, idx in enumerate(top, 1):
                st.write(f"{i}. {label_names[idx]}: {probs[0][idx]:.2%}")

except Exception as e:
    st.error(f"❌ Error: {e}")
    import traceback
    st.code(traceback.format_exc())