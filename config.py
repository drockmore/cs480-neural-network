# Configurations for the process.py script
INPUT_FILE = 'optdigits-orig.windep'
# The output file will be named the DATA_FILE variable in the neural network configurations below.

# Configurations for the neural network
DATA_FILE = "digit_data.npz" 
LEARNING_RATE = 0.01
WEIGHT_SCALE = 0.01 # Scale the random weights
HIDDEN_SIZE = 64 # Number of neurons in the hidden layer
OUTPUT_SIZE = 10
TRAINING_ITERATIONS = 1000
TEST_SIZE = .2 # 20% of the data will be used for testing
RANDOM_STATE = 42 # Sets a seed for the random number generator