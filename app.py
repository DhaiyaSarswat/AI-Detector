import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import nltk
from nltk.probability import FreqDist
from collections import Counter
from nltk.corpus import stopwords
import string
import math
import plotly.express as px
import time
import os

# ------------------------------------
# Page Config
# ------------------------------------
st.set_page_config(
    page_title="Genesis: AI Text Detector",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------------
# Attribution Banner
# ------------------------------------
st.markdown(
    """
    <div style="text-align:center; padding:10px; border-radius:10px; 
                background-color:#1A2234; border:1px solid #333A4A; 
                box-shadow: 0 4px 10px rgba(0,0,0,0.4); margin-bottom:20px;">
        <h4 style="color:#92b6ff; margin:0;">
            🔬 This project is made for <b>testing purposes</b> by:<br>
            Dhairya Sarswat, Shashwat Shinghal, Harsh Agarwal, Hannaan Akhtar <br>
            <span style="font-size:14px; color:#ccc;">Moradabad Institute of Technology</span>
        </h4>
    </div>
    """, unsafe_allow_html=True
)

# ------------------------------------
# NLTK Setup (Robust for Streamlit Cloud)
# ------------------------------------
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Ensure downloads happen quietly & path is added
nltk.download("punkt", download_dir=nltk_data_dir, quiet=True)
nltk.download("punkt_tab", download_dir=nltk_data_dir, quiet=True)
nltk.download("stopwords", download_dir=nltk_data_dir, quiet=True)
nltk.data.path.append(nltk_data_dir)

# ------------------------------------
# Load GPT-2
# ------------------------------------
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, model

tokenizer, model = load_model()

# ------------------------------------
# Detection Functions
# ------------------------------------
def calculate_perplexity(text):
    """Perplexity: measures how predictable the text is to GPT-2"""
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return torch.exp(loss).item()

def calculate_burstiness(text):
    """Burstiness: measures repetition of words"""
    tokens = nltk.word_tokenize(text.lower())
    if not tokens:
        return 0.0
    word_frequency = FreqDist(tokens)
    repeated_count = sum(count > 1 for count in word_frequency.values())
    return repeated_count / len(word_frequency)

def calculate_entropy(text):
    """Entropy: diversity/unpredictability of word usage"""
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in stopwords.words("english") and t not in string.punctuation]
    if not tokens:
        return 0.0
    word_counts = Counter(tokens)
    total = sum(word_counts.values())
    probs = [count/total for count in word_counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs)
    return entropy / math.log2(len(word_counts))  # normalized 0–1

def ai_probability(perplexity, burstiness, entropy):
    """
    Combine metrics into AI probability score.
    Lower perplexity + low burstiness + low entropy => AI-like
    """
    perplexity_score = min(perplexity / 1000, 1.0)
    burstiness_score = 1 - burstiness
    entropy_score = 1 - entropy

    score = (0.5 * (1 - perplexity_score)) + (0.25 * burstiness_score) + (0.25 * entropy_score)
    return max(0, min(score, 1)) * 100

def plot_top_repeated_words(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in stopwords.words("english") and t not in string.punctuation]
    word_counts = Counter(tokens)
    top_words = word_counts.most_common(10)
    if not top_words:
        return
    words, counts = zip(*top_words)
    fig = px.bar(
        x=words, y=counts,
        labels={'x': 'Word', 'y': 'Count'},
        title="✨ Top 10 Most Frequent Words",
        text=counts
    )
    fig.update_layout(
        title_font_color="#FFFFFF",
        font_color="#CCCCCC",
        plot_bgcolor="#1E232E",
        paper_bgcolor="#1E232E"
    )
    fig.update_traces(marker_color='#4A90E2', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------
# UI
# ------------------------------------
st.title("🛡️ Genesis: AI Text Detector")
st.markdown("<h3>Discover the origin of your text.</h3>", unsafe_allow_html=True)

st.subheader("Enter Text Below")
text_area = st.text_area("✍️", height=300, label_visibility="hidden")

col_btn1, col_btn2, col_btn3 = st.columns([1,1,1])
with col_btn2:
    if st.button("Analyze Text", use_container_width=True):
        if not text_area.strip():
            st.warning("Please enter some text to analyze.")
        else:
            word_count = len(text_area.split())
            if not (50 <= word_count <= 1000):
                st.warning(f"⚠️ Please enter between 50 and 1000 words. Your text has {word_count} words.")
            else:
                with st.spinner("Analyzing content..."):
                    time.sleep(2)

                st.divider()

                # Metrics
                perplexity = calculate_perplexity(text_area)
                burstiness = calculate_burstiness(text_area)
                entropy = calculate_entropy(text_area)
                probability = ai_probability(perplexity, burstiness, entropy)

                # Verdict
                st.subheader("Verdict")
                if probability > 60:
                    st.error(f"🤖 Likely AI-Generated (Confidence {probability:.1f}%)")
                elif 40 < probability <= 60:
                    st.warning(f"⚖️ Mixed / Uncertain (Confidence {probability:.1f}%)")
                else:
                    st.success(f"🧑 Likely Human-Written (Confidence {100 - probability:.1f}%)")

                st.markdown("<p class='red-disclaimer'>⚠️ Disclaimer: No AI detector is 100% accurate. Use results cautiously.</p>", unsafe_allow_html=True)
                st.divider()

                # Metrics
                st.subheader("Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Perplexity", f"{perplexity:.2f}")
                col2.metric("Burstiness", f"{burstiness:.2f}")
                col3.metric("Entropy", f"{entropy:.2f}")

                st.divider()
                st.subheader("Insights")
                plot_top_repeated_words(text_area)
