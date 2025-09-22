import os
import xml.etree.ElementTree as ET
import pandas as pd
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

nltk.download("punkt")
nltk.download("stopwords")

# -----------------------------
# 🔎 Helper to find files ignoring case
# -----------------------------
def find_file(folder, filename_lower):
    for f in os.listdir(folder):
        if f.lower() == filename_lower.lower():
            return os.path.join(folder, f)
    raise FileNotFoundError(f"{filename_lower} not found in {folder}")

# -----------------------------
# 📂 Setup paths
# -----------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "..", "data")

posts_path = find_file(data_dir, "posts.xml")
comments_path = find_file(data_dir, "comments.xml")

# -----------------------------
# 📝 Load posts
# -----------------------------
def load_posts(file_path, max_rows=10000):
    rows = []
    context = ET.iterparse(file_path, events=("end",))
    for i, (event, elem) in enumerate(context):
        if elem.tag == "row":
            rows.append(elem.attrib)
            if i >= max_rows:
                break
        elem.clear()
    return pd.DataFrame(rows)

# -----------------------------
# 🚀 Run Question 1 Analysis
# -----------------------------
print(f"Loading posts from {posts_path} ...")
posts_df = load_posts(posts_path, max_rows=10000)

print("Cleaning text ...")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    tokens = nltk.word_tokenize(text.lower())
    return [w for w in tokens if w.isalpha() and w not in stop_words]

all_tokens = []
for _, row in posts_df.iterrows():
    if "Body" in row:
        all_tokens.extend(clean_text(row["Body"]))

# WordCloud
print("Generating WordCloud ...")
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(all_tokens))
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Zipf’s Law
print("Analyzing Zipf’s Law ...")
freq_dist = Counter(all_tokens)
most_common = freq_dist.most_common(50)

ranks = range(1, len(most_common) + 1)
freqs = [freq for _, freq in most_common]

plt.figure(figsize=(10, 5))
plt.loglog(ranks, freqs, marker="o")
plt.title("Zipf's Law - Word Frequency Distribution")
plt.xlabel("Rank (log scale)")
plt.ylabel("Frequency (log scale)")
plt.show()
