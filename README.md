# 🎬 BERT Sentiment Classifier

Fine-tuned **BERT** (bert-base-uncased) on the IMDB dataset for binary sentiment classification.

## Results

| Metric         | Value  |
|----------------|--------|
| Test Accuracy  | 89.13% |
| Precision      | 0.89   |
| Recall         | 0.89   |
| F1 Score       | 0.89   |
| Training Epochs| 2      |
| Parameters     | 109M   |

## Demo
```bash
git clone https://github.com/divya-vj/bert-sentiment.git
cd bert-sentiment
pip install -r requirements.txt
streamlit run app.py
```

## Features

- Fine-tuned BERT on 25,000 IMDB training reviews
- Live sentiment prediction with confidence scores
- Attention heatmap visualisation — see what BERT focuses on
- Training loss curve, confusion matrix, confidence distribution
- Interactive layer/head selector for attention exploration

## Stack

- **Model**: bert-base-uncased (HuggingFace Transformers)
- **Training**: PyTorch, AdamW, linear warmup scheduler
- **Dataset**: IMDB (50K reviews, binary sentiment)
- **App**: Streamlit
- **Evaluation**: scikit-learn, seaborn, matplotlib

## Project Structure
```
bert-sentiment/
├── train.py          # Dataset class, training loop
├── evaluate.py       # Metrics, evaluation plots
├── app.py            # Streamlit app
├── data/             # Plots and training history
├── models/bert-imdb/ # Saved model weights (gitignored)
└── notebooks/        # Exploration and attention viz
```

