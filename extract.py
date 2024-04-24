import csv
import json
from transformers import AutoTokenizer


def tokenize_dataset(dataset):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Change model name as needed

    tokenized_dataset = []
    for product_name, product_price in dataset:
        tokenized_product_name = tokenizer(product_name, return_tensors="pt")
        tokenized_product_price = tokenizer(product_price, return_tensors="pt")
        tokenized_dataset.append((tokenized_product_name, tokenized_product_price))

    return tokenized_dataset


def load_dataset_from_csv(csv_file):
    dataset = []
    with open(csv_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            product_name, product_price = row
            dataset.append((product_name, product_price))
    return dataset


def save_tokenized_dataset(tokenized_dataset, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for tokenized_product_name, tokenized_product_price in tokenized_dataset:
            data = {
                'input_ids': tokenized_product_name['input_ids'].tolist(),
                'token_type_ids': tokenized_product_name['token_type_ids'].tolist(),
                'attention_mask': tokenized_product_name['attention_mask'].tolist()
            }
            file.write(json.dumps(data) + '\n')

            data = {
                'input_ids': tokenized_product_price['input_ids'].tolist(),
                'token_type_ids': tokenized_product_price['token_type_ids'].tolist(),
                'attention_mask': tokenized_product_price['attention_mask'].tolist()
            }
            file.write(json.dumps(data) + '\n')

    print("Tokenized dataset saved successfully.")


if __name__ == "__main__":
    csv_file_path = 'cleaned_dataset.csv'  # Path to your CSV file
    output_file_path = 'cleaned_tokenized_dataset.json'  # Path to save tokenized dataset
    dataset = load_dataset_from_csv(csv_file_path)
    print("Dataset loaded successfully.")
    print(f"Number of products: {len(dataset)}")

    # Tokenize the dataset
    tokenized_dataset = tokenize_dataset(dataset)

    # Save the tokenized dataset
    save_tokenized_dataset(tokenized_dataset, output_file_path)
