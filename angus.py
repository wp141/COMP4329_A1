import numpy as np
import time

#load training data
train_data = np.load("Ass1/Assignment1-Dataset/test_data.npy")
train_labels = np.load("Ass1/Assignment1-Dataset/train_label.npy")

#load testing data 
test_data = np.load("Ass1/Assignment1-Dataset/test_data.npy")
test_labels = np.load("Ass1/Assignment1-Dataset/test_label.npy")

# Check shapes and data types
# print("Train data shape:", train_data.shape)
# print("Train labels shape:", train_labels.shape)
# print("Train data type:", train_data.dtype)
# print("Train labels type:", train_labels.dtype)

print(train_data)
print(train_labels)