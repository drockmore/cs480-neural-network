import numpy as np
from config import INPUT_FILE, DATA_FILE

# This function was created with the help of chat gpt 4.o
def process_digit_file():
    """
    Processes a file containing handwritten digit data with 32x32 binary grids and corresponding labels.
    Returns:
        tuple: (inputs, targets) where
            - inputs: NumPy array of shape (n_samples, 1024) with flattened grids.
            - targets: NumPy array of shape (n_samples,) with labels.
    """
    inputs = []
    targets = []

    with open(INPUT_FILE, 'r') as file:
        lines = file.readlines()

    # Skip metadata
    metadata_end = 20
    lines = lines[metadata_end:]

    current_image = []

    for line in lines:
        line = line.strip()
        
        if len(line) == 32 and all(c in '01' for c in line):  # Valid grid line
            current_image.append([int(pixel) for pixel in line])

            # When 32 rows are collected, expect a label next
            if len(current_image) == 32:
                continue

        elif line.isdigit() and len(current_image) == 32:  # Valid label line
            label = int(line)
            targets.append(label)
            inputs.append(np.array(current_image, dtype=int).flatten())
            current_image = []  # Reset for the next image

    # Convert to NumPy arrays
    inputs = np.array(inputs, dtype=int)
    targets = np.array(targets, dtype=int)
    
    print("Data processing complete.")
    print(f"Number of samples: {len(targets)}")
    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")
    
    return inputs, targets


inputs, targets = process_digit_file()

# Save the processed data
np.savez(DATA_FILE, inputs=inputs, targets=targets)