from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch

# Load dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# Load tokenizer and model
model_name = "facebook/opt-1.3b"  # Replace with any causal LLM
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Avoid padding warnings

# Tokenize dataset
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

# Model
model = AutoModelForCausalLM.from_pretrained(model_name)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments (Accelerate-compatible)
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    warmup_steps=100,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,  # or bf16=True
    deepspeed="ds_config.json",  # Point to your DeepSpeed config
    save_total_limit=2,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Fine-tune
trainer.train()
