# term_frequency_inverted_index.py
import os
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords
import string
import time

# -----------------------
# 0️⃣ Ensure NLTK stopwords are available
# -----------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -----------------------
# 1️⃣ Locate Posts.xml
# -----------------------
posts_file = "/content/IR_Project01/data/Posts.xml"
if not os.path.exists(posts_file):
    posts_file = "Posts.xml"
if not os.path.exists(posts_file):
    raise FileNotFoundError(f"Posts.xml not found in 'data/' or current directory ({os.getcwd()})")

print(f"Using Posts.xml from: {posts_file}")

# -----------------------
# 2️⃣ Load and clean posts
# -----------------------
tree = ET.parse(posts_file)
root = tree.getroot()

posts = []
for row in root.findall("row"):
    post_id = int(row.attrib.get("Id"))
    body = row.attrib.get("Body", "")
    text = BeautifulSoup(body, "html.parser").get_text()
    posts.append((post_id, text))

# -----------------------
# 3️⃣ Preprocessing function
# -----------------------
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [w for w in text.split() if w.isalpha() and w not in stop_words]
    return tokens

# -----------------------
# 4️⃣ Build inverted index with term frequency
# -----------------------
start_time = time.time()
inverted_index = defaultdict(dict)  # term -> {post_id: freq}

for post_id, text in posts:
    tokens = preprocess(text)
    freqs = Counter(tokens)
    for term, freq in freqs.items():
        inverted_index[term][post_id] = freq

end_time = time.time()
print(f"Inverted index built with {len(inverted_index)} unique terms in {end_time - start_time:.2f} seconds")

# -----------------------
# 5️⃣ Term-at-a-time query
# -----------------------
def term_at_a_time_search(query, top_k=50):
    query_tokens = preprocess(query)
    doc_scores = defaultdict(int)  # post_id -> total score

    for term in query_tokens:
        postings = inverted_index.get(term, {})
        for post_id, freq in postings.items():
            doc_scores[post_id] += freq  # sum frequencies

    ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs[:top_k]

# -----------------------
# 6️⃣ Example usage
# -----------------------
query = "Playstation"
results = term_at_a_time_search(query)
print("Top results (post_id, total_frequency):")
for pid, score in results:
    print(pid, score)
