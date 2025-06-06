# LoRA Implementation for AG News Classification
# Converted from notebook to script with improved LoRA and training settings

import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import evaluate
import numpy as np
from datasets import load_dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score

# Check if CUDA is available and set device accordingly
print("Checking device...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the dataset
print("Loading AG News dataset...")
dataset = load_dataset('ag_news')
n_labels = len(set(dataset['train']['label']))

model_checkpoint = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=n_labels)

# --- Model 1: Evaluation without fine-tuning ---
print("\nEvaluating base model...")
encodings = tokenizer(dataset['test']['text'], truncation=True, padding=True)
input_ids = torch.tensor(encodings['input_ids']).to(device)
attention_masks = torch.tensor(encodings['attention_mask']).to(device)
labels = torch.tensor(dataset['test']['label']).to(device)
model.to(device)
tensor_dataset = TensorDataset(input_ids, attention_masks, labels)
dataloader = DataLoader(tensor_dataset, batch_size=32)

predictions = []
true_labels = []
with torch.no_grad():
    model.eval()
    for batch in tqdm(dataloader, desc="Processing batches"):
        input_ids, attention_mask, label = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label = label.to(device)
        logits = model(input_ids, attention_mask=attention_mask).logits
        preds = torch.argmax(logits, dim=1).tolist()
        predictions.extend(preds)
        true_labels.extend(label.cpu().tolist())
macro_f1 = f1_score(true_labels, predictions, average='macro')
micro_f1 = f1_score(true_labels, predictions, average='micro')
print(f"Base Model Macro F1-Score: {macro_f1}")
print(f"Base Model Micro F1-Score: {micro_f1}")

# --- Model 2: Fine-tune with LoRA ---
print("\nSetting up LoRA configuration...")
peft_config = LoraConfig(
    task_type="SEQ_CLS",
    r=8,  # Increased from 4
    lora_alpha=32,
    lora_dropout=0.05,  # Increased dropout
    target_modules=['q_lin', 'k_lin', 'v_lin', 'out_lin']  # More modules
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

print("Tokenizing training data...")
train_encodings = tokenizer(dataset['train']['text'], truncation=True, padding=True)
input_ids = torch.tensor(train_encodings['input_ids']).to(device)
attention_masks = torch.tensor(train_encodings['attention_mask']).to(device)
labels = torch.tensor(dataset['train']['label']).to(device)

model.to(device)
train_dataset = TensorDataset(input_ids, attention_masks, labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Increased batch size
num_epochs = 5  # Increased epochs

optimizer = AdamW(model.parameters(), lr=2e-5)  # Lower learning rate
num_warmup_steps = len(train_loader) // 2  # Add warmup steps
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=len(train_loader) * num_epochs
)

# Training Loop
print("\nStarting training...")
model.train()
for epoch in range(num_epochs):
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        batch = [b.to(device) for b in batch]
        inputs, masks, labels = batch
        outputs = model(inputs, attention_mask=masks, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    torch.save(model.state_dict(), f'epoch{epoch+1}_lora.pth')
    print(f"Epoch {epoch+1} finished and model saved.")

# --- Evaluation after LoRA fine-tuning ---
print("\nEvaluating LoRA fine-tuned model...")
# Reload model weights from last epoch (optional, for demonstration)
model.load_state_dict(torch.load(f'epoch{num_epochs}_lora.pth'))
model.to(device)
encodings = tokenizer(dataset['test']['text'], truncation=True, padding=True)
input_ids = torch.tensor(encodings['input_ids']).to(device)
attention_masks = torch.tensor(encodings['attention_mask']).to(device)
labels = torch.tensor(dataset['test']['label']).to(device)
dataset_test = TensorDataset(input_ids, attention_masks, labels)
dataloader = DataLoader(dataset_test, batch_size=32)
predictions = []
true_labels = []
with torch.no_grad():
    model.eval()
    for batch in tqdm(dataloader, desc="Processing batches"):
        input_ids, attention_mask, label = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label = label.to(device)
        logits = model(input_ids, attention_mask=attention_mask).logits
        preds = torch.argmax(logits, dim=1).tolist()
        predictions.extend(preds)
        true_labels.extend(label.cpu().tolist())
macro_f1 = f1_score(true_labels, predictions, average='macro')
micro_f1 = f1_score(true_labels, predictions, average='micro')
print(f"LoRA Model Macro F1-Score: {macro_f1}")
print(f"LoRA Model Micro F1-Score: {micro_f1}") 