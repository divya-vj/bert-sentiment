import torch
import json
import os
from torch.utils.data import Dataset, DataLoader
from transformers import (BertForSequenceClassification, BertTokenizer,
                          get_linear_schedule_with_warmup)
from torch.optim import AdamW
from datasets import load_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(train_size=None, test_size=None):
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
    train_loader  = DataLoader(train_dataset, batch_size=batch_size,
                               shuffle=True,  num_workers=0)
    test_loader   = DataLoader(test_dataset,  batch_size=batch_size,
                               shuffle=False, num_workers=0)
    print(f"✓ Train batches : {len(train_loader)}")
    print(f"✓ Test  batches : {len(test_loader)}")
    return train_loader, test_loader


def setup_training(train_loader, num_epochs=2):
    print("\nLoading BERT model...")
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=2
    )
    model = model.to(device)
    optimizer    = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps  = len(train_loader) * num_epochs
    warmup_steps = total_steps // 10
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded  : {total_params:,} parameters")
    print(f"✓ Device        : {next(model.parameters()).device}")
    print(f"✓ Total steps   : {total_steps}")
    print(f"✓ Warmup steps  : {warmup_steps}")
    return model, optimizer, scheduler


def train_epoch(model, loader, optimizer, scheduler, epoch, loss_history):
    model.train()
    total_loss    = 0
    correct       = 0
    total_samples = 0

    for step, batch in enumerate(loader):
        optimizer.zero_grad()
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss    += loss.item()
        preds          = outputs.logits.argmax(dim=-1)
        correct       += (preds == labels).sum().item()
        total_samples += labels.size(0)

        if (step + 1) % 100 == 0:
            avg_loss = total_loss / (step + 1)
            acc      = correct / total_samples
            print(f"  Epoch {epoch+1} | Step {step+1:4d}/{len(loader)} "
                  f"| Loss: {avg_loss:.4f} | Acc: {acc:.4f}")
            loss_history.append(round(avg_loss, 4))

    return total_loss / len(loader), correct / total_samples


def evaluate_epoch(model, loader):
    model.eval()
    total_loss    = 0
    correct       = 0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)
            outputs        = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            total_loss    += outputs.loss.item()
            preds          = outputs.logits.argmax(dim=-1)
            correct       += (preds == labels).sum().item()
            total_samples += labels.size(0)

    return total_loss / len(loader), correct / total_samples


if __name__ == "__main__":
    NUM_EPOCHS = 2
    BATCH_SIZE = 16

    print("=" * 55)
    print("  BERT Fine-Tuning — IMDB Sentiment")
    print("=" * 55)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_data, test_data = load_data()
    train_loader, test_loader = get_dataloaders(
        train_data, test_data, tokenizer, batch_size=BATCH_SIZE
    )

    model, optimizer, scheduler = setup_training(
        train_loader, num_epochs=NUM_EPOCHS
    )

    loss_history = []
    train_accs   = []
    test_accs    = []

    print(f"\n{'='*55}")
    print(f"  Starting Training")
    print(f"{'='*55}\n")

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer,
            scheduler, epoch, loss_history
        )
        print(f"\n  Evaluating on test set...")
        test_loss, test_acc = evaluate_epoch(model, test_loader)

        train_accs.append(round(train_acc, 4))
        test_accs.append(round(test_acc, 4))

        print(f"\n  Epoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}")
        print(f"  Test  Loss: {test_loss:.4f}  Test  Acc: {test_acc:.4f}")

    os.makedirs('models/bert-imdb', exist_ok=True)
    model.save_pretrained('models/bert-imdb')
    tokenizer.save_pretrained('models/bert-imdb')

    history = {
        'loss_history': loss_history,
        'train_accs'  : train_accs,
        'test_accs'   : test_accs
    }
    with open('data/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  Training Complete!")
    print(f"  Final Test Accuracy : {test_accs[-1]:.4f}")
    print(f"  Model saved to      : models/bert-imdb/")
    print(f"  History saved to    : data/training_history.json")
    print(f"{'='*55}")