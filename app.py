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
# Custom CSS for styling
# ------------------------------------
st.markdown("""
<style>
    /* Main Button Style */
    .stButton>button {
        background-color: #5A99E3; /* Brighter, more engaging blue */
        color: white;
        border-radius: 10px;
        border: 1px solid #3E6A9E; /* Subtle border for depth */
        padding: 12px 24px;
        font-weight: bold;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .stButton>button:hover {
        background-color: #4A88D3;
        transform: translateY(-2px); /* Add a lift effect on hover */
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.4);
    }

    /* Disclaimer Text */
    .red-disclaimer {
        color: #FF8B8B; /* Softer red for better readability on dark backgrounds */
        font-size: 14px;
        text-align: center;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)


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
# Create a directory for NLTK data if it doesn't exist
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Define the packages to download
nltk_packages = ["punkt", "stopwords"]
for package in nltk_packages:
    try:
        nltk.data.find(f"tokenizers/{package}")
    except LookupError:
        nltk.download(package, download_dir=nltk_data_dir, quiet=True)

# Add the custom path to NLTK's data path
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

# ------------------------------------
# Load GPT-2 Model and Tokenizer
# ------------------------------------
@st.cache_resource
def load_model():
    """Loads the GPT-2 model and tokenizer and caches them."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, model

tokenizer, model = load_model()

# ------------------------------------
# Helper & Detection Functions
# ------------------------------------
def get_clean_tokens(text):
    """Helper function to get cleaned tokens from text."""
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    return [t for t in tokens if t.isalpha() and t not in stop_words]

def calculate_perplexity(text):
    """
    Perplexity: Measures how predictable the text is to GPT-2.
    Lower values suggest more predictable (AI-like) text.
    """
    if not text.strip():
        return 0.0
    try:
        encodings = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        input_ids = encodings.input_ids
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
        return torch.exp(loss).item()
    except Exception as e:
        st.error(f"Error calculating perplexity: {e}")
        return 0.0

def calculate_burstiness(text):
    """
    Burstiness: Measures the repetition of words.
    AI text tends to have lower burstiness (less repetition).
    """
    tokens = nltk.word_tokenize(text.lower())
    if not tokens:
        return 0.0
    word_frequency = FreqDist(tokens)
    repeated_count = sum(count > 1 for count in word_frequency.values())
    return repeated_count / len(word_frequency) if len(word_frequency) > 0 else 0.0

def calculate_entropy(text):
    """
    Entropy: Measures the diversity and unpredictability of word usage.
    AI text often has lower, more uniform entropy.
    """
    tokens = get_clean_tokens(text)
    if not tokens:
        return 0.0
    word_counts = Counter(tokens)
    total_words = sum(word_counts.values())
    probs = [count / total_words for count in word_counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    num_unique_words = len(word_counts)
    if num_unique_words <= 1:
        return 0.0
    return entropy / math.log2(num_unique_words)

def ai_probability(perplexity, burstiness, entropy):
    """
    Combine metrics into an AI probability score with corrected perplexity scaling.
    """
    PPL_AI_THRESHOLD = 40.0
    PPL_HUMAN_THRESHOLD = 150.0
    raw_ppl_score = (PPL_HUMAN_THRESHOLD - perplexity) / (PPL_HUMAN_THRESHOLD - PPL_AI_THRESHOLD)
    perplexity_score = max(0, min(raw_ppl_score, 1.0))
    burstiness_score = 1 - burstiness
    entropy_score = 1 - entropy
    score = (0.5 * perplexity_score) + (0.25 * burstiness_score) + (0.25 * entropy_score)
    return max(0, min(score, 1)) * 100

def plot_top_repeated_words(text):
    """Generates and displays a bar chart of the top 10 most frequent words."""
    tokens = get_clean_tokens(text)
    if not tokens:
        st.write("No significant words found to plot.")
        return
    word_counts = Counter(tokens)
    top_words = word_counts.most_common(10)
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
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        xaxis_title=None,
        yaxis_title=None,
    )
    fig.update_traces(marker_color='#5A99E3', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------
# UI Layout
# ------------------------------------
st.title("🛡️ Genesis: AI Text Detector")
st.markdown("<h4>Discover the origin of your text by analyzing its linguistic patterns.</h4>", unsafe_allow_html=True)

st.subheader("Enter Text Below")
text_area = st.text_area(
    "✍️", 
    height=300, 
    label_visibility="hidden", 
    placeholder="Paste your text here (we recommend 50-1000 words for best results)..."
)

_, col_btn, _ = st.columns([2, 1, 2])
with col_btn:
    if st.button("Analyze Text", use_container_width=True):
        if not text_area.strip():
            st.warning("Please enter some text to analyze.")
        else:
            word_count = len(text_area.split())
            if not (50 <= word_count <= 1000):
                st.warning(f"⚠️ Please enter between 50 and 1000 words. Your text has {word_count} words.")
            else:
                with st.spinner("Performing linguistic analysis..."):
                    perplexity = calculate_perplexity(text_area)
                    burstiness = calculate_burstiness(text_area)
                    entropy = calculate_entropy(text_area)
                    probability = ai_probability(perplexity, burstiness, entropy)

                st.divider()

                # --- Verdict Section ---
                st.subheader("Verdict")
                if probability > 70:
                    st.error(f"**🤖 Likely AI-Generated** (Confidence: {probability:.1f}%)")
                elif 40 < probability <= 70:
                    st.warning(f"**⚖️ Mixed / Uncertain** (Confidence: {probability:.1f}%)")
                else:
                    human_confidence = 100 - probability
                    st.success(f"**🧑 Likely Human-Written** (Confidence: {human_confidence:.1f}%)")

                st.markdown("<p class='red-disclaimer'>⚠️ Disclaimer: No AI detector is 100% accurate. Use results cautiously as a guide, not a definitive judgment.</p>", unsafe_allow_html=True)
                st.divider()

                # --- Metrics Section (REVISED WITH st.metric) ---
                st.subheader("Linguistic Metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        label="Perplexity", 
                        value=f"{perplexity:.2f}",
                        help="Measures text predictability. Lower scores are more AI-like. 🧠"
                    )
                
                with col2:
                    st.metric(
                        label="Burstiness", 
                        value=f"{burstiness:.2f}",
                        help="Measures word repetition. Lower scores are more AI-like. 🔄"
                    )

                with col3:
                    st.metric(
                        label="Entropy", 
                        value=f"{entropy:.2f}",
                        help="Measures word diversity. Lower scores are more AI-like. 🎲"
                    )

                st.divider()

                # --- Insights Section ---
                st.subheader("Insights")
                plot_top_repeated_words(text_area)
