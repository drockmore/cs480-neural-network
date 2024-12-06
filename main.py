from numpy import random, dot
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from config import LEARNING_RATE, WEIGHT_SCALE, HIDDEN_SIZE, OUTPUT_SIZE, DATA_FILE, TRAINING_ITERATIONS, TEST_SIZE, RANDOM_STATE

# This function was created with the help of chat gpt 4.o
def visualize_predictions(X_test, predictions, labels):
    num_samples = len(predictions)  # Total number of samples
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.2)  # Leave space for the slider
    
    # Initial display
    img = ax.imshow(X_test[0].reshape(32, 32), cmap='gray')  # Reshape for visualization
    title = ax.set_title(f"Predicted: {predictions[0]}, Actual: {labels[0]}")
    ax.axis('off')
    
    # Add a slider for navigation
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Sample', 0, num_samples - 1, valinit=0, valstep=1)
    
    # Update function
    def update(val):
        idx = int(slider.val)  # Get the current index
        img.set_data(X_test[idx].reshape(32, 32))  # Update image data
        title.set_text(f"Predicted: {predictions[idx]}, Actual: {labels[idx]}")  # Update title
        fig.canvas.draw_idle()  # Redraw the figure
    
    slider.on_changed(update)  # Connect slider to the update function
    plt.show()

# The original base class for this code is from a blog post by Milo Spencer-Harper
# https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1
# Chat gpt 4.o helped with modifications to this class
class NeuralNetwork():
    
    def __init__(self, input_size, hidden_size, output_size):
        
        random.seed(1)
        self.hidden_weights = random.randn(input_size, hidden_size) * WEIGHT_SCALE
        self.output_weights = random.randn(hidden_size, output_size) * WEIGHT_SCALE

    def __relu(self, x):
        return np.maximum(0, x)
    
    def __relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def __softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Numerical stability
        return e_x / e_x.sum(axis=1, keepdims=True)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
   
        for iteration in range(number_of_training_iterations):
            
            # Forward pass
            hidden_layer_output = self.__relu(dot(training_set_inputs, self.hidden_weights))
            final_output = self.__softmax(dot(hidden_layer_output, self.output_weights))
            
            # Calculate error
            error = training_set_outputs - final_output
            
            # Backpropagation
            output_adjustment = dot(hidden_layer_output.T, error)
            hidden_error = dot(error, self.output_weights.T)
            hidden_adjustment = dot(training_set_inputs.T, hidden_error * self.__relu_derivative(hidden_layer_output))
            
            # Update weights
            self.output_weights += LEARNING_RATE * output_adjustment
            self.hidden_weights += LEARNING_RATE * hidden_adjustment
            
            # Debug error
            if iteration % 1000 == 0:
                loss = -np.mean(training_set_outputs * np.log(final_output + 1e-9))
                print(f"Iteration {iteration}, Loss: {loss}")


    def think(self, inputs):
        hidden_layer_output = self.__relu(dot(inputs, self.hidden_weights))
        return self.__softmax(dot(hidden_layer_output, self.output_weights))


if __name__ == "__main__":
    
    # Load and preprocess the dataset
    data = np.load(DATA_FILE)
    X = data['inputs'] / 255.0  # Normalize inputs
    y = data['targets']

    # One-hot encode labels
    encoder = OneHotEncoder(sparse_output=False)
    y_one_hot = encoder.fit_transform(y.reshape(-1, 1))

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Get scalar true labels for testing
    true_labels = np.argmax(y_test, axis=1)

    # Initialize neural network
    input_size = X_train.shape[1]

    neural_network = NeuralNetwork(input_size, HIDDEN_SIZE, OUTPUT_SIZE)

    # Train the network
    print("Training the neural network...")
    neural_network.train(X_train, y_train, TRAINING_ITERATIONS)

    # Test the network
    print("Testing the network...")
    predicted_labels = []
    for i in range(X_test.shape[0]):
        input_sample = X_test[i]
        predicted_probs = neural_network.think(input_sample.reshape(1, -1))
        predicted_label = np.argmax(predicted_probs)
        predicted_labels.append(predicted_label)

    print("True Labels:")
    print(true_labels[:20])  # First 20 true labels
    print("Predicted Labels:")
    print(predicted_labels[:20])  # First 20 predicted labels

    # Compute accuracy
    accuracy = np.mean(np.array(predicted_labels) == true_labels)
    print(f"Accuracy on test set: {accuracy * 100:.2f}%")
    
    visualize_predictions(
    X_test,
    predicted_labels,
    true_labels,
    )

