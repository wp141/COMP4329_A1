import numpy as np


# Load training and testing data
train_data = np.load("train_data.npy")
train_labels = np.load("train_label.npy")
test_data = np.load("test_data.npy")
test_labels = np.load("test_label.npy")

# (Optional) Print the shapes to confirm data dimensions
print("Training data shape:", train_data.shape)
print("Training labels shape:", train_labels.shape)
print("Test data shape:", test_data.shape)
print("Test labels shape:", test_labels.shape)

# Build the multilayer neural network model
model = Sequential()

# Input layer + first hidden layer with 128 neurons and ReLU activation
model.add(Dense(128, activation='relu', input_shape=(train_data.shape[1],)))
model.add(Dropout(0.2))  # Dropout layer for regularization

# Second hidden layer with 64 neurons
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

# Output layer
# For binary classification use 1 neuron with sigmoid activation.
# If using multi-class classification, replace with:
#   model.add(Dense(num_classes, activation='softmax'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model.
# If your task is binary classification, use 'binary_crossentropy'.
# For multi-class classification, use 'sparse_categorical_crossentropy' or 'categorical_crossentropy' (if one-hot encoded).
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model. Adjust epochs and batch_size as needed.
model.fit(train_data, train_labels, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model on test data
loss, accuracy = model.evaluate(test_data, test_labels, verbose=0)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
