# train.py
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from datasets import load_dataset


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


if __name__ == "__main__":
    print("=== Testing Data Pipeline ===\n")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_data, test_data = load_data(train_size=100, test_size=50)
    train_loader, test_loader = get_dataloaders(
        train_data, test_data, tokenizer, batch_size=16
    )

    batch = next(iter(train_loader))

    print(f"\n=== One Batch Shapes ===")
    print(f"input_ids      : {batch['input_ids'].shape}")
    print(f"attention_mask : {batch['attention_mask'].shape}")
    print(f"labels         : {batch['labels'].shape}")

    decoded = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)
    print(f"\nDecoded text (first 100 chars):")
    print(f"  {decoded[:100]}")

    print("\n✅ Data pipeline working correctly!")