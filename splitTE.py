import csv
import random

# Load encoded dataset from CSV file
dataset = []
with open('encoded_dataset.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        dataset.append(row)  # Append the entire row as a list

# Shuffle the dataset
random.shuffle(dataset)

# Define the percentage of data for training and validation
train_percentage = 0.8
val_percentage = 0.2

# Calculate the number of samples for each split
num_samples = len(dataset)
num_train_samples = int(train_percentage * num_samples)
num_val_samples = int(val_percentage * num_samples)

# Split the dataset
train_dataset = dataset[:num_train_samples]
val_dataset = dataset[num_train_samples:]

# Save the split datasets to separate files
with open('train_dataset.txt', 'w') as train_file:
    for example in train_dataset:
        train_file.write(f"{example[0]},{example[1]},{example[2]}\n")

with open('val_dataset.txt', 'w') as val_file:
    for example in val_dataset:
        val_file.write(f"{example[0]},{example[1]},{example[2]}\n")

print("Dataset split into training and validation sets.")
