import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt


# This function was created with the help of chatgpt 4.o
def visualize_predictions(X_test, predictions, labels, num_samples=5):
    for i in range(num_samples):
        plt.figure(figsize=(4, 4))
        plt.imshow(X_test[i].reshape(32, 32), cmap='gray')  # Reshape for visualization
        plt.title(f"Predicted: {predictions[i]}, Actual: {labels[i]}")
        plt.axis('off')
        plt.show()



# Load your custom dataset
data = np.load("digit_data.npz")
X = data['inputs']  # Input features
y = data['targets']  # Labels

# Normalize input data
X = X / 255.0

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# True labels
true_labels = np.argmax(y_test, axis=1)


# Define the model
model = Sequential([
    #Flatten(input_shape=(32, 32)),  # Flatten the 32x32 input images
    Dense(256, activation='relu'),  # Hidden layer with ReLU activation
    Dense(128, activation='relu'),  # Additional hidden layer
    Dense(10, activation='softmax')  # Output layer with softmax activation
])

# Compile the model
model.compile(
    optimizer='adam',  # Adam optimizer
    loss='categorical_crossentropy',  # Cross-entropy loss
    metrics=['accuracy']  # Track accuracy during training
)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_split=0.2,  # Use 20% of training data for validation
    epochs=1,
    batch_size=32,
    verbose=1  # Print progress
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save the trained model
model.save("digit_classifier_model.h5")
print("Model saved!")

# Make predictions on new data
predictions = model.predict(X_test[:5])  # Predict on the first 5 samples from the test set
print("Predictions:", np.argmax(predictions, axis=1))

visualize_predictions(
    X_test,
    predictions,
    true_labels,
)




