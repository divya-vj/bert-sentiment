# app.py — BERT Sentiment Classifier — Streamlit App
import streamlit as st
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import numpy as np
import json
from transformers import BertForSequenceClassification, BertTokenizer

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="BERT Sentiment Classifier",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #8b5cf6, #2dd4bf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 0;
    }
    .metric-card {
        background: #1a1a26;
        border: 1px solid rgba(139,92,246,0.3);
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .result-positive {
        background: rgba(52,211,153,0.1);
        border: 2px solid #34d399;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .result-negative {
        background: rgba(248,113,113,0.1);
        border: 2px solid #f87171;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model (cached — only runs once) ─────────────────────────
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained(
        'models/bert-imdb',
        output_attentions=True
    )
    tokenizer = BertTokenizer.from_pretrained('models/bert-imdb')
    model.eval()
    return model, tokenizer

# ── Predict function ─────────────────────────────────────────────
def predict(text, model, tokenizer):
    inputs = tokenizer(
        text,
        return_tensors='pt',
        max_length=128,
        padding='max_length',
        truncation=True
    )
    with torch.no_grad():
        outputs = model(**inputs)

    probs      = torch.softmax(outputs.logits, dim=-1)[0]
    pred_class = probs.argmax().item()
    label      = "Positive" if pred_class == 1 else "Negative"
    confidence = probs[pred_class].item()
    neg_conf   = probs[0].item()
    pos_conf   = probs[1].item()

    return label, confidence, neg_conf, pos_conf, outputs.attentions, inputs

# ── Attention heatmap function ───────────────────────────────────
def plot_attention(attentions, inputs, tokenizer, layer=11, head=0):
    tokens = tokenizer.convert_ids_to_tokens(
        inputs['input_ids'][0].tolist()
    )
    # Remove padding tokens for cleaner plot
    real_len = inputs['attention_mask'][0].sum().item()
    tokens   = tokens[:real_len]
    attn     = attentions[layer][0, head, :real_len, :real_len].detach().numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='Purples',
        ax=ax,
        square=True,
        cbar_kws={'shrink': 0.5}
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=7)
    ax.set_title(f'Attention Weights — Layer {layer+1}, Head {head+1}',
                 fontsize=10)
    plt.tight_layout()
    return fig

# ════════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════════

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    attn_layer = st.slider("Attention Layer", 1, 12, 12) - 1
    attn_head  = st.slider("Attention Head",  1, 12,  1) - 1
    st.divider()
    st.markdown("## 📊 Model Info")
    st.markdown("""
    - **Model**: bert-base-uncased
    - **Dataset**: IMDB (50K reviews)
    - **Parameters**: 109M
    - **Test Accuracy**: 89.13%
    - **Epochs**: 2
    - **Batch size**: 16
    - **Learning rate**: 2e-5
    """)
    st.divider()
    st.markdown("## 🔗 Links")
    st.markdown("[GitHub Repo](https://github.com/divya-vj/bert-sentiment)")

# ── Header ───────────────────────────────────────────────────────
st.markdown('<p class="main-title">🎬 BERT Sentiment Classifier</p>',
            unsafe_allow_html=True)
st.markdown('<p class="subtitle">Fine-tuned bert-base-uncased on IMDB · '
            '89% accuracy · 109M parameters</p>',
            unsafe_allow_html=True)
st.divider()

# ── Load model ───────────────────────────────────────────────────
with st.spinner("Loading BERT model... (first load takes ~10 seconds)"):
    model, tokenizer = load_model()

# ── Prediction Section ───────────────────────────────────────────
st.header("🔍 Try It — Paste Any Movie Review")

example_reviews = {
    "Select an example...": "",
    "⭐⭐⭐⭐⭐ Masterpiece": "This film is an absolute masterpiece. The acting was superb, the cinematography breathtaking, and the story kept me completely hooked from start to finish. One of the best films I have ever seen.",
    "⭐ Terrible": "Terrible movie. Boring, predictable plot with awful acting. Complete waste of two hours. I cannot believe how bad this was.",
    "⭐⭐⭐ Mixed": "It was okay I guess. Some parts were interesting but overall the film felt too long and the ending was disappointing. Not terrible but not great either.",
    "⭐⭐⭐⭐⭐ Brilliant acting": "Brilliant performances all round. The lead actor delivered one of the most convincing portrayals I have seen in years. Highly recommended.",
    "⭐ Walk out": "I walked out after 30 minutes. Painfully slow, zero character development, and dialogue that made me cringe. Avoid at all costs."
}

selected = st.selectbox("Or pick an example:", list(example_reviews.keys()))
default_text = example_reviews[selected]

user_text = st.text_area(
    "Review text:",
    value=default_text,
    height=130,
    placeholder="Type or paste a movie review here..."
)

col_btn1, col_btn2 = st.columns([1, 5])
with col_btn1:
    analyse = st.button("Analyse ▶", type="primary", use_container_width=True)
with col_btn2:
    clear = st.button("Clear", use_container_width=False)

if clear:
    user_text = ""

if analyse and user_text.strip():
    with st.spinner("Running BERT inference..."):
        label, confidence, neg_conf, pos_conf, attentions, inputs = predict(
            user_text, model, tokenizer
        )

    st.divider()

    # ── Result display ────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        emoji = "😊" if label == "Positive" else "😞"
        color = "#34d399" if label == "Positive" else "#f87171"
        st.markdown(f"""
        <div style="background: rgba(139,92,246,0.08);
                    border: 2px solid {color};
                    border-radius: 14px; padding: 24px;
                    text-align: center;">
            <div style="font-size: 3rem;">{emoji}</div>
            <div style="font-size: 1.6rem; font-weight: 800;
                        color: {color};">{label}</div>
            <div style="font-size: 1.2rem; color: #e2e8f0;
                        margin-top: 6px;">{confidence*100:.1f}% confident</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("**Confidence Scores**")
        st.markdown(f"😞 Negative")
        st.progress(neg_conf)
        st.caption(f"{neg_conf*100:.1f}%")
        st.markdown(f"😊 Positive")
        st.progress(pos_conf)
        st.caption(f"{pos_conf*100:.1f}%")

    with col3:
        st.markdown("**Attention Heatmap**")
        st.caption(f"Layer {attn_layer+1}, Head {attn_head+1} "
                   f"— adjust in sidebar")
        fig = plot_attention(attentions, inputs, tokenizer,
                             layer=attn_layer, head=attn_head)
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

elif analyse and not user_text.strip():
    st.warning("Please enter a review first.")

# ── Training Results Section ─────────────────────────────────────
st.header("📈 Training Results")

tab1, tab2, tab3, tab4 = st.tabs([
    "Loss Curve",
    "Confusion Matrix",
    "Confidence Distribution",
    "Attention Heatmap"
])

with tab1:
    st.subheader("Training Loss Curve")
    st.caption("Loss decreasing over training steps — confirms model learned successfully")
    try:
        st.image('data/loss_curve.png', use_column_width=True)
        with open('data/training_history.json') as f:
            history = json.load(f)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Start Loss", f"{history['loss_history'][0]:.3f}")
        c2.metric("End Loss",   f"{history['loss_history'][-1]:.3f}")
        c3.metric("Epoch 1 Acc", f"{history['train_accs'][0]*100:.1f}%")
        c4.metric("Epoch 2 Acc", f"{history['train_accs'][1]*100:.1f}%")
    except FileNotFoundError:
        st.info("Run evaluate.py first to generate this plot.")

with tab2:
    st.subheader("Confusion Matrix")
    st.caption("True vs predicted labels across all 25,000 test reviews")
    try:
        st.image('data/confusion_matrix.png', use_column_width=True)
    except FileNotFoundError:
        st.info("Run evaluate.py first to generate this plot.")

with tab3:
    st.subheader("Confidence Score Distribution")
    st.caption("How confident the model is across positive and negative reviews")
    try:
        st.image('data/confidence_distribution.png', use_column_width=True)
    except FileNotFoundError:
        st.info("Run evaluate.py first to generate this plot.")

with tab4:
    st.subheader("Attention Visualisation")
    st.caption("What the model focuses on — darker = stronger attention")
    try:
        col_a, col_b = st.columns(2)
        with col_a:
            st.image('data/attention_heatmap.png',
                     caption="Single head heatmap",
                     use_column_width=True)
        with col_b:
            st.image('data/attention_comparison.png',
                     caption="Positive vs Negative comparison",
                     use_column_width=True)
    except FileNotFoundError:
        st.info("Run attention_viz notebook first.")

# ── Classification Report ─────────────────────────────────────────
st.divider()
st.header("📋 Classification Report")
try:
    with open('data/classification_report.txt') as f:
        report = f.read()
    st.code(report)
except FileNotFoundError:
    st.info("Run evaluate.py first.")

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Model: bert-base-uncased · Dataset: IMDB (50K reviews) · "
    "Accuracy: 89.13% · Stack: HuggingFace Transformers, PyTorch, Streamlit"
)