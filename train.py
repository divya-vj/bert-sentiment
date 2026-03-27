# train.py
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


if __name__ == "__main__":
    # Quick test — run with: python train.py
    train, test = load_data()

    # Show one sample
    print(f"\nSample review (first 200 chars):")
    print(f"  Label : {train[0]['label']}")
    print(f"  Text  : {train[0]['text'][:200]}")