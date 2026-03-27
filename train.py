# train.py
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (BertForSequenceClassification, BertTokenizer,
                          get_linear_schedule_with_warmup)
from torch.optim import AdamW
from datasets import load_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_data(train_size=None, test_size=None):
    """
    Load the IMDB dataset from HuggingFace.

    Args:
        train_size: int, limit training samples (e.g. 2000 for fast testing)
        test_size:  int, limit test samples
    Returns:
        (train_dataset, test_dataset)
    """
    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb")

    train = dataset['train']
    test  = dataset['test']

    if train_size:
        train = train.select(range(train_size))
    if test_size:
        test = test.select(range(test_size))

    print(f"✓ Train samples : {len(train)}")
    print(f"✓ Test  samples : {len(test)}")
    print(f"✓ Columns       : {train.column_names}")
    return train, test

class IMDBDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=128):
        self.dataset    = hf_dataset
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item  = self.dataset[idx]
        text  = item['text']
        label = item['label']

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids'     : encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels'        : torch.tensor(label, dtype=torch.long)
        }
    


def get_dataloaders(train_data, test_data, tokenizer, batch_size=16):
    train_dataset = IMDBDataset(train_data, tokenizer)
    test_dataset  = IMDBDataset(test_data,  tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    print(f"✓ Train batches : {len(train_loader)}")
    print(f"✓ Test  batches : {len(test_loader)}")
    return train_loader, test_loader

def setup_training(train_loader, num_epochs=2):
    """
    Load BERT model, configure optimizer and learning rate scheduler.
    Returns model, optimizer, scheduler.
    """
    print("\nLoading BERT model...")
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2
    )
    model = model.to(device)

    # AdamW with weight decay — standard BERT fine-tuning recipe
    # lr=2e-5 is the canonical learning rate — don't change it
    optimizer = AdamW(
        model.parameters(),
        lr=2e-5,
        weight_decay=0.01
    )

    total_steps  = len(train_loader) * num_epochs
    warmup_steps = total_steps // 10     # warm up over first 10% of steps

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded      : {total_params:,} parameters")
    print(f"✓ Device            : {next(model.parameters()).device}")
    print(f"✓ Total steps       : {total_steps}")
    print(f"✓ Warmup steps      : {warmup_steps}")
    print(f"✓ Learning rate     : 2e-5")

    return model, optimizer, scheduler


if __name__ == "__main__":
    print("=== Testing Model Setup + Forward Pass ===\n")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Small dataset for fast testing
    train_data, test_data = load_data(train_size=32, test_size=16)
    train_loader, test_loader = get_dataloaders(
        train_data, test_data, tokenizer, batch_size=8
    )

    # Load model, optimizer, scheduler
    model, optimizer, scheduler = setup_training(train_loader, num_epochs=2)

    # Grab one real batch from your data
    batch = next(iter(train_loader))

    # Move batch to device
    input_ids      = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels         = batch['labels'].to(device)

    # Forward pass
    print("\n=== Forward Pass ===")
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    print(f"Loss         : {outputs.loss.item():.4f}")
    print(f"Logits shape : {outputs.logits.shape}")

    probs = torch.softmax(outputs.logits, dim=-1)
    preds = probs.argmax(dim=-1)
    print(f"Predictions  : {preds.tolist()}")
    print(f"True labels  : {labels.tolist()}")

    # Quick accuracy on this one batch
    acc = (preds == labels).float().mean().item()
    print(f"Batch accuracy (random model): {acc:.2f}")
    print(f"\n✅ Forward pass successful! Ready to train.")