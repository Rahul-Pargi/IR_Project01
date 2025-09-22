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
    """Load posts/comments safely from XML using iterparse."""
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
    """Clean HTML tags, URLs, non-letters, and extra spaces."""
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

# === LOAD DATA ===
print(f"Loading posts from {posts_path} ...")
posts_df = load_posts(posts_path, max_rows=10000)
print("Loaded posts:", posts_df.shape)

print(f"Loading comments from {comments_path} ...")
comments_df = load_posts(comments_path, max_rows=10000)
print("Loaded comments:", comments_df.shape)

if 'Body' not in posts_df.columns:
    raise RuntimeError("No 'Body' column found in Posts.xml rows. Check the XML structure.")

# === PREPROCESS TEXT ===
bodies_clean = posts_df['Body'].astype(str).fillna('').map(clean_html)
all_text = " ".join(bodies_clean.tolist()).lower()
tokens = word_tokenize(all_text)
tokens_alpha = [t for t in tokens if t.isalpha()]

# === FREQUENCY COUNTS ===
counts_all = Counter(tokens_alpha)
top20_all = counts_all.most_common(20)
top20_all_dict = dict(top20_all)

wc_all = WordCloud(width=800, height=400, background_color='white', max_words=20)
wc_all.generate_from_frequencies(top20_all_dict)

stop_words = set(stopwords.words('english'))
tokens_no_stop = [w for w in tokens_alpha if w not in stop_words]
counts_no_stop = Counter(tokens_no_stop)
top20_no_stop = counts_no_stop.most_common(20)
top20_no_stop_dict = dict(top20_no_stop)

wc_nostop = WordCloud(width=800, height=400, background_color='white', max_words=20)
wc_nostop.generate_from_frequencies(top20_no_stop_dict)

# === DISPLAY WORDCLOUDS INLINE ===
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
axes[0].imshow(wc_all, interpolation='bilinear')
axes[0].set_title('Top-20 words (raw tokens — includes stopwords)', fontsize=14)
axes[0].axis('off')
axes[1].imshow(wc_nostop, interpolation='bilinear')
axes[1].set_title('Top-20 words (stopwords removed)', fontsize=14)
axes[1].axis('off')
plt.show()

# === PRINT TOP-20 TABLES ===
top20_all_df = pd.DataFrame(top20_all, columns=['word', 'count']).reset_index(drop=True)
top20_no_stop_df = pd.DataFrame(top20_no_stop, columns=['word', 'count']).reset_index(drop=True)
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

# === ZIPF'S LAW PLOT ===
freqs = np.array(sorted(counts_no_stop.values(), reverse=True))
N = min(1000, len(freqs))
ranks = np.arange(1, N + 1)
freqs_top = freqs[:N]

plt.figure(figsize=(10, 5))
plt.plot(ranks, freqs_top)
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.title('Rank vs Frequency (top {} words)'.format(N))
plt.grid(True)
plt.show()

log_ranks = np.log(ranks)
log_freqs = np.log(freqs_top)
slope, intercept = np.polyfit(log_ranks, log_freqs, 1)
pred_log = intercept + slope * log_ranks
ss_res = np.sum((log_freqs - pred_log) ** 2)
ss_tot = np.sum((log_freqs - np.mean(log_freqs)) ** 2)
r2 = 1 - ss_res / ss_tot

plt.figure(figsize=(10, 6))
plt.scatter(log_ranks, log_freqs, s=10)
plt.plot(log_ranks, pred_log, linewidth=2)
plt.xlabel('log(Rank)')
plt.ylabel('log(Frequency)')
plt.title('Zipf plot (log-log). Slope ~= {:.3f}, R^2 = {:.3f}'.format(slope, r2))
plt.grid(True)
plt.show()

print("Log-log linear fit: slope = {:.4f}, intercept = {:.4f}, R^2 = {:.4f}".format(slope, intercept, r2))
print("Zipf interpretation: frequency ~ c * rank^{slope}. Zipf's 's' is -slope (expected ~1).")
