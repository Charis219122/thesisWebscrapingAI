import csv
import json

# Load the tokenized dataset from the CSV file
tokenized_dataset = []
with open('tokenized_dataset.csv', mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for row in reader:
        tokenized_name = row[0]  # Tokenized name is already a string
        product_price = row[1]
        tokenized_dataset.append((tokenized_name, product_price))

# Initialize the encoding dictionary
encoding_dict = {}

# Save the encoded dataset
with open('encoded_dataset.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['tokenized_name', 'product_price'])
    for tokenized_name, product_price in tokenized_dataset:
        # Convert the tokenized name to JSON string and store it along with the product price
        writer.writerow([json.dumps(tokenized_name), product_price])

print("Encoded dataset saved successfully.")
