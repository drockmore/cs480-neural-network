# Handwriting digit recognization

## Overview
This project is a nerual network to recognize handwritten digits.

## Requirements
- Python 3.8 or higher
- Virtual environment setup (optional but recommended)

## Installation
```
pip install -r requirements.txt
```

### Running the program
```
python main.py
```

### Configuration
The configuration variables are located in the `config.py` file. The following variables are available:

*INPUT_FILE*: This is the filename for that will be used for encoding in the `process.py` file.

*DATA_FILE*: This is the file name for the input file for the neural network training data and also the output file name for the encoder. 

*LEARNING_RATE*: The learning rate for the neural network.

*WEIGHT_SCALE*: The scale for the random weights used in the neural network.

*HIDDEN_SIZE*: The number of neurons in the hidden layer used in the neural network.

*OUTPUT_SIZE*: The output size used in the neural network.

*TRAINING_ITERATIONS*: The training iteration used in the neural network.

*TEST_SIZE*: The percentage of the data that will be used for testing in the neural network.

*RANDOM_STATE*: Sets a seed for the random number generator used in the neural network.

### Encoding
The encoder can be used by running the command `python process.py`. Ensure that the file location is specified in `config.py`.

### Training
The neural network can be ran using the command `python main.py`. The neural network configurations can be modified in `config.py`. After the training is complete a matplotlib popup will appear that displays the digit, label, and predicated value. 

There is also a neural network that uses tensorflow. It can be ran by using the command `python tensor-flow.py`. 