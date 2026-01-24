"""
PyTorch implementation of MNIST digit recognition model
This is an alternative implementation using PyTorch instead of TensorFlow
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os


class MNISTNet(nn.Module):
    """
    ANN model for MNIST digit recognition using PyTorch
    Architecture: Input(784) -> Dense(512) -> Dense(256) -> Dense(128) -> Output(10)
    """

    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Flatten input
        x = x.view(-1, 784)

        # Layer 1
        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        # Layer 2
        x = self.relu(self.fc2(x))
        x = self.dropout(x)

        # Layer 3
        x = self.relu(self.fc3(x))
        x = self.dropout(x)

        # Output layer (no activation here, CrossEntropyLoss applies softmax)
        x = self.fc4(x)

        return x


class MNISTModelPyTorch:
    def __init__(self, model_path='mnist_model_pytorch.pth'):
        self.model_path = model_path
        self.model = MNISTNet()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f'Using device: {self.device}')

    def train(self, epochs=10, batch_size=128):
        """Train the model on MNIST dataset"""
        # Data preprocessing
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])

        # Load datasets
        train_dataset = datasets.MNIST(root='./data', train=True,
                                       download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False,
                                      download=True, transform=transform)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=2)

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        print('Building and training ANN model (PyTorch)...')
        print('Model Architecture: Input(784) -> Dense(512) -> Dense(256) -> Dense(128) -> Output(10)')
        print()

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Statistics
                train_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                # Print progress
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch + 1}/{epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                          f'({100. * batch_idx / len(train_loader):.0f}%)] '
                          f'Loss: {loss.item():.4f}')

            # Epoch statistics
            train_acc = 100. * correct / total
            avg_loss = train_loss / len(train_loader)
            print(f'Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {train_acc:.2f}%')

        # Evaluate on test set
        print('\nEvaluating on test set...')
        test_loss, test_acc = self.evaluate(test_loader, criterion)

        print('\n' + '=' * 50)
        print('Training complete!')
        print(f'Final Test Accuracy: {test_acc:.2f}%')
        print(f'Final Test Loss: {test_loss:.4f}')
        print('=' * 50)

        # Save model
        self.save_model()

        return test_loss, test_acc

    def evaluate(self, test_loader, criterion):
        """Evaluate model on test set"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        test_loss /= len(test_loader)
        test_acc = 100. * correct / total

        print(f'Test Loss: {test_loss:.4f}')
        print(f'Test Accuracy: {test_acc:.2f}%')

        return test_loss, test_acc

    def save_model(self):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_architecture': 'MNISTNet'
        }, self.model_path)
        print(f'Model saved to {self.model_path}')

    def load_model(self):
        """Load a trained model"""
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f'Model loaded from {self.model_path}')
            return True
        else:
            print(f'Model file {self.model_path} not found')
            return False

    def predict(self, image):
        """
        Predict digit from image

        Args:
            image: numpy array of shape (28, 28)

        Returns:
            predicted digit (0-9) and confidence scores
        """
        if self.model is None:
            raise Exception('Model not loaded or trained')

        self.model.eval()

        with torch.no_grad():
            # Convert numpy array to tensor
            if isinstance(image, np.ndarray):
                # Ensure 2D shape
                if len(image.shape) == 3:
                    image = image.squeeze()

                # Normalize using MNIST stats
                # Note: Image should already be in MNIST format (bright digit on dark background)
                image = (image.astype('float32') / 255.0 - 0.1307) / 0.3081
                image_tensor = torch.FloatTensor(image).unsqueeze(0).to(self.device)
            else:
                image_tensor = image.to(self.device)

            # Get model output
            output = self.model(image_tensor)

            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(output, dim=1)

            # Get prediction
            predicted_digit = output.argmax(1).item()
            confidence = probabilities[0].cpu().numpy()

        return predicted_digit, confidence


if __name__ == '__main__':
    # Train and save model
    model = MNISTModelPyTorch()
    model.train(epochs=10)
