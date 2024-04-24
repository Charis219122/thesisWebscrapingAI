import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Load pre-trained GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Define paths to your training and validation datasets
train_dataset_path = "train_test.txt"
val_dataset_path = "val_test.txt"

# Load training and validation datasets
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_dataset_path,
    block_size=128  # Adjust block size as needed
)
val_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=val_dataset_path,
    block_size=128  # Adjust block size as needed
)

# Define data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",  # Directory to save the fine-tuned model
    overwrite_output_dir=True,
    num_train_epochs=3,  # Number of training epochs
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,  # Save checkpoint every 500 steps
    save_total_limit=2,  # Only keep the last 2 checkpoints
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Fine-tune the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(eval_results)
