# evaluate.py
# Run this on Colab with GPU for speed.
# All outputs saved to data/

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import classification_report, confusion_matrix
from train import load_data, get_dataloaders
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load saved model
print("\nLoading saved model from models/bert-imdb/...")
model     = BertForSequenceClassification.from_pretrained('models/bert-imdb')
tokenizer = BertTokenizer.from_pretrained('models/bert-imdb')
model     = model.to(device)
model.eval()
print("✓ Model loaded")

# Load test data
_, test_data = load_data()
_, test_loader = get_dataloaders(test_data, test_data, tokenizer, batch_size=32)

# Run inference
print("\nRunning inference on test set...")
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['labels']
        outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
        probs          = torch.softmax(outputs.logits, dim=-1)
        preds          = probs.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())
        if (i + 1) % 50 == 0:
            print(f"  {(i+1)*32:5d}/25000 done...")

accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
print(f"\n✓ Accuracy: {accuracy*100:.2f}%")

# Classification report
print("\n=== Classification Report ===")
report = classification_report(all_labels, all_preds,
                               target_names=['Negative', 'Positive'])
print(report)
os.makedirs('data', exist_ok=True)
with open('data/classification_report.txt', 'w') as f:
    f.write("BERT IMDB Sentiment Classification Report\n")
    f.write("=" * 45 + "\n\n")
    f.write(report)

# Plot 1: Loss curve
with open('data/training_history.json') as f:
    history = json.load(f)
loss_steps = list(range(1, len(history['loss_history']) + 1))
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(loss_steps, history['loss_history'], color='#8b5cf6',
        linewidth=2, marker='o', markersize=4)
ax.fill_between(loss_steps, history['loss_history'], alpha=0.1, color='#8b5cf6')
ax.set_xlabel('Steps (×100)')
ax.set_ylabel('Training Loss')
ax.set_title('BERT Fine-Tuning — Training Loss Curve')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data/loss_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: loss_curve.png")

# Plot 2: Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            linewidths=0.5, ax=ax,
            annot_kws={'size': 14, 'weight': 'bold'})
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title(f'Confusion Matrix — BERT IMDB\nAccuracy: {accuracy*100:.2f}%')
plt.tight_layout()
plt.savefig('data/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: confusion_matrix.png")

# Plot 3: Confidence distribution
pos_probs = [p for p, l in zip(all_probs, all_labels) if l == 1]
neg_probs = [p for p, l in zip(all_probs, all_labels) if l == 0]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(pos_probs, bins=50, alpha=0.6, color='#34d399',
             label='Positive', edgecolor='none')
axes[0].hist(neg_probs, bins=50, alpha=0.6, color='#f87171',