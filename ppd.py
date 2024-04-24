from bs4 import BeautifulSoup
import os
import csv


def parse_html_file(html_path):
    with open(html_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
        soup = BeautifulSoup(html_content, 'html.parser')
        # Extract product name and price using BeautifulSoup
        product_name_element = soup.find(attrs={"klarna-ai-label": "Name"})
        product_price_element = soup.find(attrs={"klarna-ai-label": "Price"})

        if product_name_element is not None and product_price_element is not None:
            product_name = product_name_element.text.strip()
            product_price = product_price_element.text.strip()
            return product_name, product_price
        else:
            return None, None


def load_dataset(data_path):
    dataset = []
    for root, dirs, files in os.walk(data_path):
        for file_name in files:
            if file_name.endswith('.html'):
                html_path = os.path.join(root, file_name)
                product_name, product_price = parse_html_file(html_path)
                if product_name is not None and product_price is not None:
                    dataset.append((product_name, product_price))
    return dataset


def save_dataset_to_csv(dataset, csv_file):
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Product Name', 'Product Price'])  # Write the header row
        for product_name, product_price in dataset:
            writer.writerow([product_name, product_price])


if __name__ == "__main__":
    data_path = r'C:\Users\Pc1\Desktop\klarna_product_page_dataset_WTL_50k\train\AT'
    dataset = load_dataset(data_path)
    print("Dataset loaded successfully.")
    print(f"Number of products: {len(dataset)}")

    # Save the dataset to a CSV file
    csv_file = 'dataset.csv'
    save_dataset_to_csv(dataset, csv_file)
    print(f"Dataset saved to '{csv_file}'")
