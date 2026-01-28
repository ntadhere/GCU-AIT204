# text_classification_20newsgroups_pytorch_skeleton.py
# Purpose: TF-IDF + PyTorch MLP for 20 Newsgroups (STRUCTURE-ONLY SKELETON)

import os
import random
import numpy as np

# ---- Reproducibility (optional but recommended) ----
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

import torch
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================
# Load Dataset
# =========================
print("Loading 20 Newsgroups dataset...")
data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
X_raw, y = data.data, data.target
num_classes = len(data.target_names)

print(f"✓ Dataset loaded: {len(X_raw)} documents, {num_classes} categories")

# =========================
# Preprocess text data
#  - Convert to lowercase, remove punctuation if desired
#  - Tokenization is handled by TfidfVectorizer, but you can add custom steps if needed
# =========================
import string
import re

def preprocess_text(text):
    """
    Preprocess a single text document.
    
    Steps:
    1. Convert to lowercase
    2. Remove punctuation
    3. Remove numbers (optional)
    4. Remove extra whitespace
    
    Args:
        text (str): Raw text document
    Returns:
        str: Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    
    # Remove numbers (optional - comment out if you want to keep numbers)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Apply preprocessing to all documents
print("Preprocessing text documents (lowercase, remove punctuation, etc.)...")
X_raw = [preprocess_text(doc) for doc in X_raw]
print("✓ Preprocessing complete!")

# =========================
# Convert Text Data to Numerical Format using TF-IDF
# =========================
print("\n" + "="*80)
print("CONVERTING TEXT TO NUMERICAL FORMAT (TF-IDF)")
print("="*80)

print("\nTF-IDF (Term Frequency-Inverse Document Frequency) Configuration:")
print("  - max_features: 5000 (keep top 5000 most frequent words)")
print("  - lowercase: True (normalize text)")
print("  - stop_words: 'english' (remove common words like 'the', 'is', 'and')")
print("  - strip_accents: 'unicode' (handle special characters)")
print("  - token_pattern: alphabetic words with 2+ characters only")

vectorizer = TfidfVectorizer(
    max_features=5000,          # Limit to 5000 most frequent words
    lowercase=True,              # Convert to lowercase
    stop_words='english',        # Remove English stop words
    strip_accents='unicode',     # Normalize accented characters
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"  # Only alphabetic words, 2+ chars
)

print("\nFitting vectorizer and transforming documents...")
X_vec = vectorizer.fit_transform(X_raw)

print(f"\n✓ Vectorization complete!")
print(f"  - Original: {len(X_raw)} text documents")
print(f"  - Converted to: {X_vec.shape} numerical matrix")
print(f"  - Vocabulary size: {len(vectorizer.vocabulary_):,} unique words")
print(f"  - Matrix sparsity: {(1 - X_vec.nnz / (X_vec.shape[0] * X_vec.shape[1])) * 100:.2f}%")

# Show sample vocabulary
feature_names = vectorizer.get_feature_names_out()
print(f"\n📋 Sample vocabulary (first 20 words): {', '.join(feature_names[:20])}")

# Convert sparse matrix to dense array for PyTorch
print("\nConverting sparse matrix to dense array for PyTorch...")
X_vec = X_vec.toarray()         # NOTE: densifies; OK at 5k features, but watch RAM on small machines
print(f"✓ Converted to dense array: {X_vec.shape}, Memory: {X_vec.nbytes / (1024**2):.2f} MB")

# =========================
# Split data into Training and Testing Sets
# =========================
print("\n" + "="*80)
print("SPLITTING DATA INTO TRAINING AND TESTING SETS")
print("="*80)

print("\nSplit Configuration:")
print("  - Test size: 20% (0.2)")
print("  - Train size: 80% (0.8)")
print("  - Stratified: Yes (maintains class distribution)")
print("  - Random state: 42 (for reproducibility)")

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, 
    test_size=0.2,      # 20% for testing
    stratify=y,         # Maintain class distribution across train/test
    random_state=SEED
)

print(f"\n✓ Data split complete!")
print(f"  - Training samples: {len(X_train):,} ({len(X_train)/len(X_vec)*100:.1f}%)")
print(f"  - Testing samples: {len(X_test):,} ({len(X_test)/len(X_vec)*100:.1f}%)")
print(f"  - Features per sample: {X_train.shape[1]:,}")

# Verify class distribution
from collections import Counter
train_dist = Counter(y_train)
test_dist = Counter(y_test)

print(f"\n✓ Class distribution verification:")
print(f"  - All {num_classes} classes present in training: {len(train_dist) == num_classes}")
print(f"  - All {num_classes} classes present in testing: {len(test_dist) == num_classes}")

print(f"\n📊 Sample class distribution (first 5 classes):")
for i in range(min(5, num_classes)):
    train_count = train_dist[i]
    test_count = test_dist[i]
    total_count = train_count + test_count
    print(f"  {data.target_names[i]:35s} - Train: {train_count:4d}, Test: {test_count:4d}, Total: {total_count:4d}")

print(f"\n✓ Data ready for PyTorch!")

# =========================
# Torch Tensors & Dataloaders
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

from torch.utils.data import TensorDataset, DataLoader

train_ds = TensorDataset(X_train_t, y_train_t)
test_ds  = TensorDataset(X_test_t,  y_test_t)

# TODO: Tune batch_size
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  drop_last=False)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, drop_last=False)

# =========================
# Design Neural Network Architecture
# =========================
print("\n" + "="*80)
print("STEP 2: DESIGNING NEURAL NETWORK ARCHITECTURE")
print("="*80)

import torch.nn as nn

class NewsMLP(nn.Module):
    """
    Multi-Layer Perceptron for 20 Newsgroups Classification
    
    Architecture:
    - Input Layer: input_dim features (TF-IDF vectors)
    - Hidden Layer 1: 512 neurons with ReLU activation + Dropout
    - Hidden Layer 2: 256 neurons with ReLU activation + Dropout
    - Output Layer: num_classes neurons (logits for softmax)
    
    Note: Softmax is applied automatically in CrossEntropyLoss, 
    so we output raw logits here.
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        
        # Input Layer → Hidden Layer 1
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        
        # Hidden Layer 1 → Hidden Layer 2
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        
        # Hidden Layer 2 → Output Layer
        self.fc3 = nn.Linear(256, num_classes)
        
        # Note: No softmax here! CrossEntropyLoss applies it internally
        # If you need softmax for inference, use: torch.softmax(output, dim=1)

    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Input → Hidden Layer 1
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Hidden Layer 1 → Hidden Layer 2
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Hidden Layer 2 → Output
        x = self.fc3(x)
        
        # Return raw logits (softmax applied in loss function)
        return x

# Get input dimension from data
input_dim = X_train_t.shape[1]

# Create model and move to device (GPU if available, else CPU)
print(f"\nCreating neural network...")
print(f"  - Input dimension: {input_dim} (TF-IDF features)")
print(f"  - Output dimension: {num_classes} (number of classes)")
print(f"  - Device: {device}")

model = NewsMLP(input_dim=input_dim, num_classes=num_classes).to(device)

print(f"\n✓ Model created successfully!")
print(f"\n📊 Network Architecture:")
print("="*80)
print(f"  Input Layer:       {input_dim:>6d} features")
print(f"  Hidden Layer 1:    {512:>6d} neurons (ReLU + Dropout 0.3)")
print(f"  Hidden Layer 2:    {256:>6d} neurons (ReLU + Dropout 0.3)")
print(f"  Output Layer:      {num_classes:>6d} neurons (logits → softmax)")
print("="*80)

# Calculate total parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n📈 Model Parameters:")
print(f"  - Total parameters: {total_params:,}")
print(f"  - Trainable parameters: {trainable_params:,}")

# Show detailed layer parameters
print(f"\n📋 Layer-by-Layer Parameters:")
for name, param in model.named_parameters():
    print(f"  - {name:20s}: {str(param.shape):20s} = {param.numel():>10,} params")

# Show model summary
print(f"\n📐 Model Summary:")
print(model)

# =========================
# Compile the Model (PyTorch-style setup)
# =========================
print("\n" + "="*80)
print("STEP 3: CONFIGURING OPTIMIZER AND LOSS FUNCTION")
print("="*80)

# Define Loss Function (Categorical Cross-Entropy for multiclass classification)
print("\n[1] Loss Function: CrossEntropyLoss")
print("  - Type: Categorical Cross-Entropy")
print("  - Purpose: Measures difference between predicted and actual class probabilities")
print("  - Note: Combines LogSoftmax + NLLLoss for numerical stability")
criterion = nn.CrossEntropyLoss()
print("  ✓ CrossEntropyLoss initialized")

# Define Optimizer (Adam - Adaptive Moment Estimation)
print("\n[2] Optimizer: Adam")
print("  - Learning rate: 0.001 (1e-3)")
print("  - Purpose: Updates weights using backpropagation")
print("  - Advantages: Adaptive learning rates, works well for sparse data")
LEARNING_RATE = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
print("  ✓ Adam optimizer initialized")

# Optional: Learning Rate Scheduler
print("\n[3] Learning Rate Scheduler: ReduceLROnPlateau")
print("  - Reduces learning rate when validation loss stops improving")
print("  - Factor: 0.5 (halve the learning rate)")
print("  - Patience: 3 epochs")
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',      # Minimize validation loss
    factor=0.5,      # Multiply LR by 0.5
    patience=3,      # Wait 3 epochs before reducing
    verbose=True     # Print when LR is reduced
)
print("  ✓ Scheduler initialized")

print("\n✓ Training configuration complete!")
print("="*80)

# =========================
# Train the Model
# =========================
print("\n" + "="*80)
print("STEP 3: TRAINING THE MODEL")
print("="*80)

def train_model(num_epochs=15):
    """
    Train the neural network using backpropagation.
    
    Training Process:
    1. Forward pass: Compute predictions
    2. Compute loss: Compare predictions to actual labels
    3. Backward pass: Compute gradients using backpropagation
    4. Update weights: Apply optimizer to adjust parameters
    
    Args:
        num_epochs: Number of complete passes through the training data
    """
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"  - Training samples: {len(train_loader.dataset):,}")
    print(f"  - Batch size: {train_loader.batch_size}")
    print(f"  - Batches per epoch: {len(train_loader)}")
    print(f"  - Total training steps: {len(train_loader) * num_epochs:,}")
    print("\n" + "-"*80)
    print(f"{'Epoch':>5} | {'Train Loss':>12} | {'Train Acc':>10} | {'Val Loss':>12} | {'Val Acc':>10} | {'Time':>8}")
    print("-"*80)
    
    # Track metrics for visualization
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    import time
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # =========================
        # Training Phase
        # =========================
        model.train()  # Set model to training mode (enables dropout)
        
        running_loss = 0.0
        running_correct = 0
        total = 0
        
        for batch_idx, (xb, yb) in enumerate(train_loader):
            # Move data to device (GPU/CPU)
            xb, yb = xb.to(device), yb.to(device)
            
            # STEP 1: Zero the gradients (clear previous gradients)
            optimizer.zero_grad()
            
            # STEP 2: Forward pass (compute predictions)
            logits = model(xb)
            
            # STEP 3: Compute loss (categorical cross-entropy)
            loss = criterion(logits, yb)
            
            # STEP 4: Backward pass (compute gradients using backpropagation)
            loss.backward()
            
            # STEP 5: Update weights (apply optimizer)
            optimizer.step()
            
            # Track metrics
            preds = logits.argmax(dim=1)  # Get predicted class
            running_loss += loss.item() * xb.size(0)
            running_correct += (preds == yb).sum().item()
            total += xb.size(0)
        
        # Calculate epoch metrics
        epoch_loss = running_loss / total
        epoch_acc = running_correct / total
        
        # =========================
        # Validation Phase
        # =========================
        val_loss, val_acc = evaluate_model(test_loader)
        
        # Save to history
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Learning rate scheduling (reduce LR if validation loss plateaus)
        scheduler.step(val_loss)
        
        # Print epoch results
        epoch_time = time.time() - epoch_start_time
        print(f"{epoch+1:5d} | {epoch_loss:12.6f} | {epoch_acc:10.4f} | {val_loss:12.6f} | {val_acc:10.4f} | {epoch_time:7.2f}s")
        
        # Early stopping check (if validation loss hasn't improved in many epochs)
        if epoch > 5 and val_loss > min(history['val_loss'][:-1]) * 1.05:
            # If current val_loss is 5% worse than best, consider stopping
            patience_counter = sum(1 for vl in history['val_loss'][-5:] if vl > min(history['val_loss']))
            if patience_counter >= 5:
                print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
                print(f"  Validation loss hasn't improved for 5 epochs")
                break
    
    print("-"*80)
    print("✓ Training complete!")
    
    return history


def evaluate_model(data_loader):
    """
    Evaluate the model on a dataset (validation or test).
    
    Args:
        data_loader: DataLoader for the dataset to evaluate
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()  # Set model to evaluation mode (disables dropout)
    
    running_loss = 0.0
    running_correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient computation for efficiency
        for xb, yb in data_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            # Forward pass
            logits = model(xb)
            loss = criterion(logits, yb)
            
            # Track metrics
            preds = logits.argmax(dim=1)
            running_loss += loss.item() * xb.size(0)
            running_correct += (preds == yb).sum().item()
            total += xb.size(0)
    
    avg_loss = running_loss / total
    accuracy = running_correct / total
    
    return avg_loss, accuracy


# =========================
# Execute Training
# =========================
print("\n🚀 Starting training process...")
NUM_EPOCHS = 15
history = train_model(num_epochs=NUM_EPOCHS)

# Print final results
print("\n" + "="*80)
print("TRAINING RESULTS SUMMARY")
print("="*80)
print(f"Best Training Accuracy:   {max(history['train_acc']):.4f} ({max(history['train_acc'])*100:.2f}%)")
print(f"Best Validation Accuracy: {max(history['val_acc']):.4f} ({max(history['val_acc'])*100:.2f}%)")
print(f"Final Training Loss:      {history['train_loss'][-1]:.6f}")
print(f"Final Validation Loss:    {history['val_loss'][-1]:.6f}")
print("="*80)

# =========================
# Evaluate the Model on Test Set
# =========================
print("\n" + "="*80)
print("FINAL EVALUATION ON TEST SET")
print("="*80)

def final_evaluation():
    """
    Perform comprehensive evaluation on the test set.
    Includes accuracy, classification report, and confusion matrix.
    """
    print("\nEvaluating model on test set...")
    
    model.eval()  # Set to evaluation mode
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():  # Disable gradient computation
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            # Forward pass
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)  # Convert to probabilities
            preds = logits.argmax(dim=1)
            
            # Store predictions and targets
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    # Concatenate all batches
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    y_probs = np.concatenate(all_probs)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
    test_accuracy = accuracy_score(y_true, y_pred)
    
    print(f"\n🎯 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    print("\n" + "="*80)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(y_true, y_pred, target_names=data.target_names, digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "="*80)
    print("CONFUSION MATRIX")
    print("="*80)
    print("(Rows = True labels, Columns = Predicted labels)")
    print(cm)
    
    # Per-class accuracy
    print("\n" + "="*80)
    print("PER-CLASS ACCURACY")
    print("="*80)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(per_class_acc):
        print(f"{data.target_names[i]:35s}: {acc:.4f} ({acc*100:.2f}%)")
    
    return y_true, y_pred, y_probs, test_accuracy


# Run final evaluation
y_true, y_pred, y_probs, test_accuracy = final_evaluation()

# =========================
# Save Model
# =========================
print("\n" + "="*80)
print("SAVING MODEL")
print("="*80)

# Save model state dict
torch.save(model.state_dict(), 'model_state_dict.pt')
print("✓ Saved model weights: model_state_dict.pt")

# Save complete model
torch.save(model, 'complete_model.pt')
print("✓ Saved complete model: complete_model.pt")

# Save training history
import json
history_dict = {
    'train_loss': [float(x) for x in history['train_loss']],
    'train_acc': [float(x) for x in history['train_acc']],
    'val_loss': [float(x) for x in history['val_loss']],
    'val_acc': [float(x) for x in history['val_acc']],
    'test_accuracy': float(test_accuracy),
    'num_epochs': len(history['train_loss'])
}

with open('training_history.json', 'w') as f:
    json.dump(history_dict, f, indent=2)
print("✓ Saved training history: training_history.json")

print("\n✨ All training and evaluation complete!")
print("="*80)