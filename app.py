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
    
    # Handle longer texts by processing in chunks
    max_length = model.config.n_positions
    stride = 512
    
    nlls = []
    for i in range(0, input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = i + stride
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids_chunk = input_ids[:, begin_loc:end_loc]
        target_ids = input_ids_chunk.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids_chunk, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()

def calculate_burstiness(text):
    """Burstiness: measures repetition of words"""
    tokens = nltk.word_tokenize(text.lower())
    if len(tokens) < 10:  # Too short for meaningful analysis
        return 0.5
    
    # Filter out stopwords and punctuation
    filtered_tokens = [t for t in tokens if t not in stopwords.words("english") and t not in string.punctuation]
    
    if not filtered_tokens:
        return 0.5
        
    word_frequency = FreqDist(filtered_tokens)
    total_words = len(filtered_tokens)
    unique_words = len(word_frequency)
    
    # Calculate type-token ratio
    ttr = unique_words / total_words
    
    # Calculate how many words are repeated
    repeated_words = sum(1 for count in word_frequency.values() if count > 1)
    repetition_ratio = repeated_words / unique_words if unique_words > 0 else 0
    
    # Combined burstiness score
    burstiness = 0.7 * (1 - ttr) + 0.3 * repetition_ratio
    return burstiness

def calculate_entropy(text):
    """Entropy: diversity/unpredictability of word usage"""
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in stopwords.words("english") and t not in string.punctuation]
    
    if len(tokens) < 5:
        return 0.5
        
    word_counts = Counter(tokens)
    total = sum(word_counts.values())
    
    # Calculate entropy
    probs = [count/total for count in word_counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs)
    
    # Normalize between 0-1 based on reasonable bounds for text
    # Typical entropy for English text is between 9-13 bits
    normalized_entropy = min(max((entropy - 9) / 4, 0), 1)
    return normalized_entropy

def ai_probability(perplexity, burstiness, entropy):
    """
    Combine metrics into AI probability score.
    Lower perplexity + low burstiness + low entropy => AI-like
    """
    # Normalize perplexity (GPT-2 perplexity typically between 20-100 for coherent text)
    perplexity_norm = min(max((perplexity - 20) / 80, 0), 1)
    
    # AI text tends to have lower perplexity, lower burstiness, and lower entropy
    ai_perplexity = 1 - perplexity_norm  # Lower perplexity = more AI-like
    ai_burstiness = 1 - burstiness       # Lower burstiness = more AI-like  
    ai_entropy = 1 - entropy             # Lower entropy = more AI-like
    
    # Adjust weights based on empirical testing
    score = (0.4 * ai_perplexity) + (0.35 * ai_burstiness) + (0.25 * ai_entropy)
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

