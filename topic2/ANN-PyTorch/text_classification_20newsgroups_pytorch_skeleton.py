# text_classification_20newsgroups_pytorch_FINAL.py
# Complete implementation: TF-IDF + PyTorch MLP (2 Hidden Layers)
# 20 Newsgroups Text Classification

import os
import random
import numpy as np
import string
import re
import time
import json

# ---- Reproducibility ----
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter
import joblib

print("="*80)
print("20 NEWSGROUPS TEXT CLASSIFICATION - COMPLETE PIPELINE")
print("="*80)

# =========================
# STEP 1: Load Dataset
# =========================
print("\n[STEP 1] Loading 20 Newsgroups dataset...")
data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
X_raw, y = data.data, data.target
num_classes = len(data.target_names)
print(f"✓ Loaded {len(X_raw):,} documents, {num_classes} categories")

# =========================
# STEP 2: Preprocess Text
# =========================
print("\n[STEP 2] Preprocessing text...")

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    return text

X_raw = [preprocess_text(doc) for doc in X_raw]
print("✓ Preprocessing complete")

# =========================
# STEP 3: TF-IDF Vectorization
# =========================
print("\n[STEP 3] Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    lowercase=True,
    stop_words='english',
    strip_accents='unicode',
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
)

X_vec = vectorizer.fit_transform(X_raw).toarray()
print(f"✓ Created {X_vec.shape} feature matrix")

# =========================
# STEP 4: Train-Test Split
# =========================
print("\n[STEP 4] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, stratify=y, random_state=SEED
)
print(f"✓ Train: {len(X_train):,}, Test: {len(X_test):,}")

# =========================
# STEP 5: Create PyTorch DataLoaders
# =========================
print("\n[STEP 5] Creating PyTorch dataloaders...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Using device: {device}")

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_ds = TensorDataset(X_train_t, y_train_t)
test_ds = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

# =========================
# STEP 6: Define Neural Network (2 HIDDEN LAYERS)
# =========================
print("\n[STEP 6] Defining neural network architecture...")

class NewsMLP(nn.Module):
    """
    2-Layer MLP for text classification
    Architecture: Input(5000) → Hidden1(512) → Hidden2(256) → Output(20)
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        
        # Input → Hidden Layer 1
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        
        # Hidden Layer 1 → Hidden Layer 2
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        
        # Hidden Layer 2 → Output
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

input_dim = X_train.shape[1]
model = NewsMLP(input_dim=input_dim, num_classes=num_classes).to(device)

print("✓ Model Architecture:")
print(f"  Input:    {input_dim} features")
print(f"  Hidden 1: 512 neurons (ReLU + Dropout 0.3)")
print(f"  Hidden 2: 256 neurons (ReLU + Dropout 0.3)")
print(f"  Output:   {num_classes} classes")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# =========================
# STEP 7: Setup Training
# =========================
print("\n[STEP 7] Configuring optimizer and loss...")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)
print("✓ Using Adam optimizer + CrossEntropyLoss")

# =========================
# STEP 8: Training Loop
# =========================
print("\n[STEP 8] Training model...")
print("-"*80)
print(f"{'Epoch':>5} | {'Train Loss':>12} | {'Train Acc':>10} | {'Val Loss':>12} | {'Val Acc':>10} | {'Time':>8}")
print("-"*80)

def evaluate_model(data_loader):
    model.eval()
    running_loss, running_correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for xb, yb in data_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            preds = logits.argmax(dim=1)
            
            running_loss += loss.item() * xb.size(0)
            running_correct += (preds == yb).sum().item()
            total += xb.size(0)
    
    return running_loss / total, running_correct / total

NUM_EPOCHS = 15
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    
    # Training
    model.train()
    running_loss, running_correct, total = 0.0, 0, 0
    
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        
        preds = logits.argmax(dim=1)
        running_loss += loss.item() * xb.size(0)
        running_correct += (preds == yb).sum().item()
        total += xb.size(0)
    
    train_loss = running_loss / total
    train_acc = running_correct / total
    
    # Validation
    val_loss, val_acc = evaluate_model(test_loader)
    
    # Save history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    # Update scheduler
    scheduler.step(val_loss)
    
    epoch_time = time.time() - epoch_start
    print(f"{epoch+1:5d} | {train_loss:12.6f} | {train_acc:10.4f} | {val_loss:12.6f} | {val_acc:10.4f} | {epoch_time:7.2f}s")

print("-"*80)
print("✓ Training complete!")

# =========================
# STEP 9: Final Evaluation
# =========================
print("\n[STEP 9] Final evaluation on test set...")

model.eval()
all_preds, all_targets = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        preds = logits.argmax(dim=1)
        
        all_preds.append(preds.cpu().numpy())
        all_targets.append(yb.cpu().numpy())

y_pred = np.concatenate(all_preds)
y_true = np.concatenate(all_targets)

test_accuracy = accuracy_score(y_true, y_pred)
print(f"\n🎯 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

print("\n" + "="*80)
print("CLASSIFICATION REPORT")
print("="*80)
print(classification_report(y_true, y_pred, target_names=data.target_names, digits=4))

# =========================
# STEP 10: Save Model & Artifacts for Streamlit
# =========================
print("\n[STEP 10] Saving model and artifacts...")

# Save model weights
torch.save(model.state_dict(), 'model_state_dict.pt')
print("✓ Saved: model_state_dict.pt")

# Save vectorizer (CRITICAL for Streamlit!)
joblib.dump(vectorizer, 'vectorizer.pkl')
print("✓ Saved: vectorizer.pkl")

# Save label names (CRITICAL for Streamlit!)
with open('label_names.json', 'w') as f:
    json.dump(list(data.target_names), f, indent=2)
print("✓ Saved: label_names.json")

# Save training history
history_dict = {
    'train_loss': [float(x) for x in history['train_loss']],
    'train_acc': [float(x) for x in history['train_acc']],
    'val_loss': [float(x) for x in history['val_loss']],
    'val_acc': [float(x) for x in history['val_acc']],
    'test_accuracy': float(test_accuracy),
    'num_epochs': NUM_EPOCHS
}

with open('training_history.json', 'w') as f:
    json.dump(history_dict, f, indent=2)
print("✓ Saved: training_history.json")

print("\n" + "="*80)
print("📦 FILES FOR STREAMLIT DEPLOYMENT:")
print("="*80)
print("  1. model_state_dict.pt   ← Model weights")
print("  2. vectorizer.pkl        ← TF-IDF vectorizer (converts text to numbers)")
print("  3. label_names.json      ← Category names")
print("\n💡 Upload these 3 files with your streamlit_app.py")
print("="*80)

print("\n✨ Complete! Ready for deployment!")