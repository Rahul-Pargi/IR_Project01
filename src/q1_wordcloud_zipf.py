# === INSTALL (run once in Colab if needed) ===
# !pip install wordcloud
# !pip install nltk

# === IMPORTS ===
from collections import Counter
import xml.etree.ElementTree as ET
import pandas as pd
import re
from html import unescape
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import os

# === DOWNLOAD NLTK DATA ===
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# === HELPERS ===
def load_posts(file_path, max_rows=None):
    """Load XML posts/comments safely using iterparse."""
    rows = []
    context = ET.iterparse(file_path, events=('end',))
    for event, elem in context:
        if elem.tag == 'row':
            rows.append(elem.attrib.copy())
            elem.clear()
        if max_rows and len(rows) >= max_rows:
            break
    return pd.DataFrame(rows)

def clean_html(text):
    """Clean HTML tags, URLs, and non-letter characters."""
    if not isinstance(text, str):
        return ''
    text = unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^A-Za-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# === FILE PATHS ===
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
posts_path = os.path.join(data_dir, "Posts.xml")
comments_path = os.path.join(data_dir, "Comments.xml")

print(f"Loading posts from {posts_path} ...")
posts_df = load_posts(posts_path, max_rows=10000)
print("Loaded posts:", posts_df.shape)

print(f"Loading comments from {comments_path} ...")
comments_df = load_posts(comments_path, max_rows=10000)
print("Loaded comments:", comments_df.shape)

if 'Body' not in posts_df.columns:
    raise RuntimeError("No 'Body' column found in Posts.xml rows.")

# === PREPROCESS TEXT ===
bodies_clean = posts_df['Body'].astype(str).fillna('').map(clean_html)
all_text = " ".join(bodies_clean.tolist()).lower()
tokens = word_tokenize(all_text)
tokens_alpha = [t for t in tokens if t.isalpha()]

# === FREQUENCY COUNTS ===
counts_all = Counter(tokens_alpha)
top20_all = counts_all.most_common(20)
top20_all_dict = dict(top20_all)

stop_words = set(stopwords.words('english'))
tokens_no_stop = [w for w in tokens_alpha if w not in stop_words]
counts_no_stop = Counter(tokens_no_stop)
top20_no_stop = counts_no_stop.most_common(20)
top20_no_stop_dict = dict(top20_no_stop)

# === GENERATE WORDCLOUDS ===
wc_all = WordCloud(width=800, height=400, background_color='white', max_words=20)
wc_all.generate_from_frequencies(top20_all_dict)
wc_all.to_file("top20_raw.png")

wc_nostop = WordCloud(width=800, height=400, background_color='white', max_words=20)
wc_nostop.generate_from_frequencies(top20_no_stop_dict)
wc_nostop.to_file("top20_nostop.png")

print("✅ WordCloud images saved: top20_raw.png and top20_nostop.png")

# === PLOT TOP-20 BARS ===
top20_all_df = pd.DataFrame(top20_all, columns=['word', 'count']).reset_index(drop=True)
top20_no_stop_df = pd.DataFrame(top20_no_stop, columns=['word', 'count']).reset_index(drop=True)

plt.figure(figsize=(12,6))
plt.bar(top20_all_df['word'], top20_all_df['count'], color='skyblue')
plt.title("Top-20 words (raw tokens — includes stopwords)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("top20_raw_bar.png")
plt.close()

plt.figure(figsize=(12,6))
plt.bar(top20_no_stop_df['word'], top20_no_stop_df['count'], color='orange')
plt.title("Top-20 words (stopwords removed)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("top20_nostop_bar.png")
plt.close()

print("✅ Top-20 bar plots saved: top20_raw_bar.png and top20_nostop_bar.png")

# === ZIPF'S LAW PLOTS ===
freqs = np.array(sorted(counts_no_stop.values(), reverse=True))
N = min(1000, len(freqs))
ranks = np.arange(1, N + 1)
freqs_top = freqs[:N]

# Linear rank-frequency
plt.figure(figsize=(10,5))
plt.plot(ranks, freqs_top)
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.title(f'Rank vs Frequency (top {N} words)')
plt.grid(True)
plt.savefig("rank_vs_freq.png")
plt.close()

# Log-Log plot
log_ranks = np.log(ranks)
log_freqs = np.log(freqs_top)
slope, intercept = np.polyfit(log_ranks, log_freqs, 1)
pred_log = intercept + slope * log_ranks

plt.figure(figsize=(10,6))
plt.scatter(log_ranks, log_freqs, s=10)
plt.plot(log_ranks, pred_log, linewidth=2, color='red')
plt.xlabel('log(Rank)')
plt.ylabel('log(Frequency)')
plt.title(f'Zipf plot (log-log). Slope ~= {slope:.3f}, R^2 = {1 - np.sum((log_freqs - pred_log)**2)/np.sum((log_freqs - np.mean(log_freqs))**2):.3f}')
plt.grid(True)
plt.savefig("zipf_loglog.png")
plt.close()

print("✅ Zipf plots saved: rank_vs_freq.png and zipf_loglog.png")

# === PRINT TOP-20 TABLES ===
print("\nTop-20 (raw tokens):")
print(top20_all_df.to_string(index=False))
print("\nTop-20 (stopwords removed):")
print(top20_no_stop_df.to_string(index=False))

set_all = set(top20_all_df['word'])
set_nostop = set(top20_no_stop_df['word'])
removed_by_stopword = set_all - set_nostop
added_after_removal = set_nostop - set_all
print("\nWords removed from top-20 by stopword filtering (likely stopwords):", removed_by_stopword)
print("New words that appear in top-20 after stopword removal:", added_after_removal)

print("\n✅ All processing complete. Check PNG files for graphical plots.")
