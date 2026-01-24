import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pickle
import os


class MNISTModel:
    def __init__(self, model_path='mnist_model.h5'):
        self.model_path = model_path
        self.model = None

    def build_model(self):
        """Build an ANN model for MNIST digit recognition"""
        model = keras.Sequential([
            layers.Input(shape=(784,)),  # Flattened 28x28 image
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(10, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model

    def train(self, epochs=10, batch_size=128):
        """Train the model on MNIST dataset"""
        # Load MNIST dataset
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        # Preprocess data
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        # Flatten images for ANN (28x28 -> 784)
        x_train = x_train.reshape(-1, 784)
        x_test = x_test.reshape(-1, 784)

        # Convert labels to categorical
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

        # Build model if not already built
        if self.model is None:
            self.build_model()

        # Train model
        history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            verbose=1
        )

        # Evaluate on test set
        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=0)
        print(f'\nTest accuracy: {test_acc:.4f}')
        print(f'Test loss: {test_loss:.4f}')

        # Save model
        self.save_model()

        return history, (test_loss, test_acc)

    def save_model(self):
        """Save the trained model"""
        if self.model is not None:
            self.model.save(self.model_path)
            print(f'Model saved to {self.model_path}')

    def load_model(self):
        """Load a trained model"""
        if os.path.exists(self.model_path):
            self.model = keras.models.load_model(self.model_path)
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

        # Preprocess image - flatten to 784 dimensions for ANN
        if len(image.shape) == 3:
            image = image.squeeze()  # Remove channel dimension if present

        image = image.reshape(1, 784)  # Flatten to (1, 784)
        image = image.astype('float32') / 255.0

        # Make prediction
        predictions = self.model.predict(image, verbose=0)
        predicted_digit = np.argmax(predictions[0])
        confidence = predictions[0]

        return predicted_digit, confidence


if __name__ == '__main__':
    # Train and save model
    mnist_model = MNISTModel()
    print('Building and training ANN model...')
    print('Model Architecture: Input(784) -> Dense(512) -> Dense(256) -> Dense(128) -> Output(10)')
    history, (test_loss, test_acc) = mnist_model.train(epochs=10)
    print('\n' + '='*50)
    print('Training complete!')
    print(f'Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)')
    print(f'Final Test Loss: {test_loss:.4f}')
    print('='*50)
