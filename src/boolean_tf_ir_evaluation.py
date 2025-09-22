# ----------------------------
# boolean_tf_ir_evaluation.py
# ----------------------------
import os
import pandas as pd
import xml.etree.ElementTree as ET
import re, string, time, math
from collections import defaultdict

# ----------------------------
# 0️⃣ Define path for Posts.xml
# ----------------------------
repo_root = os.getcwd()  # Current working directory
posts_path = os.path.join(repo_root, "data/Posts.xml")

if not os.path.exists(posts_path):
    raise FileNotFoundError(f"{posts_path} not found! Current folder: {os.getcwd()}")

# ----------------------------
# 1️⃣ Text normalization (No NLTK)
# ----------------------------
stop_words = set([
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves',
    'he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their',
    'theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an',
    'the','and','but','if','or','because','as','until','while','of','at','by','for','with','about',
    'against','between','into','through','during','before','after','above','below','to','from','up',
    'down','in','out','on','off','over','under','again','further','then','once','here','there','when',
    'where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor',
    'not','only','own','same','so','than','too','very','s','t','can','will','just','don','should','now'
])

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)          # Remove HTML
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    return tokens

# ----------------------------
# 2️⃣ Load posts (subset for speed)
# ----------------------------
def load_posts(file_path, max_rows=None):
    posts = []
    for i, (event, elem) in enumerate(ET.iterparse(file_path, events=("end",))):
        if elem.tag == "row":
            post_id = elem.attrib.get("Id")
            title = elem.attrib.get("Title", "")
            body = elem.attrib.get("Body", "")
            posts.append({"Id": int(post_id), "Text": title + " " + body})
            if max_rows and len(posts) >= max_rows:
                break
        elem.clear()
    return pd.DataFrame(posts)

# ----------------------------
# 3️⃣ Build indexes
# ----------------------------
def build_indexes(df):
    boolean_index = defaultdict(set)
    tf_index = defaultdict(lambda: defaultdict(int))
    doc_text_map = {}
    start_time = time.time()

    for _, row in df.iterrows():
        doc_id = row["Id"]
        tokens = normalize_text(row["Text"])
        doc_text_map[doc_id] = tokens
        for token in tokens:
            boolean_index[token].add(doc_id)
            tf_index[token][doc_id] += 1

    build_time = time.time() - start_time
    return boolean_index, tf_index, doc_text_map, build_time

# ----------------------------
# 4️⃣ Boolean search (AND/OR)
# ----------------------------
def boolean_search(query, boolean_index, operator="OR"):
    tokens = normalize_text(query)
    if not tokens:
        return set()
    sets = [boolean_index.get(t, set()) for t in tokens]
    if operator.upper() == "AND":
        result = set.intersection(*sets) if sets else set()
    else:
        result = set.union(*sets)
    return result

# ----------------------------
# 5️⃣ TF-based ranking (top-k)
# ----------------------------
def tf_ranking(query, tf_index, k=50):
    tokens = normalize_text(query)
    scores = defaultdict(int)
    for t in tokens:
        for doc_id, freq in tf_index.get(t, {}).items():
            scores[doc_id] += freq
    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in ranked_docs[:k]]

# ----------------------------
# 6️⃣ Evaluation metrics
# ----------------------------
def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    hits = sum(1 for doc in retrieved_k if doc in relevant)
    return hits / k

def ndcg_at_k(retrieved, relevant, k):
    dcg = 0
    for i, doc in enumerate(retrieved[:k]):
        rel = 1 if doc in relevant else 0
        dcg += (2**rel - 1) / math.log2(i + 2)
    idcg = sum((2**1 - 1) / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / idcg if idcg > 0 else 0

# ----------------------------
# 7️⃣ Load data and build indexes
# ----------------------------
posts_df = load_posts(posts_path, max_rows=50000)
boolean_index, tf_index, doc_text_map, build_time = build_indexes(posts_df)
print(f"Built indexes with {len(boolean_index)} unique terms in {build_time:.2f}s")

# ----------------------------
# 8️⃣ Query set (20 queries)
# ----------------------------
queries = [
    'Can I download PlayStation 3 games for my PlayStation 4',
    'Playstation payment problem',
    'Downloading games onto a PlayStation 4',
    'Dark Souls 2: Xbox360 vs PlayStation 3 vs Xbox One vs PlayStation 4',
    'GTA V won\'t load online',
    'Trouble downloading GTA5 on XBOX360',
    'Is digital download GTA5 Faster than Physical copy?',
    'Do I need to be online to play story mode?',
    'Do I need to play previous Witchers before Witcher3',
    'Witcher 3 crashes constantly',
    'Skyrim reinstalling it with all the mods',
    'Can I start with Dark souls 2?',
    'Defeating Moonlight Butterfly in Dark Souls',
    'Max Farming in Dark Souls III',
    'What are the different endings of Sekiro?',
    'How precise is the Deflect Mechanic in Sekiro?',
    'Can you upgrade the horse in Elden Ring?',
    'Does dexterity increase weapon art speed in Elden Ring?',
    'What happens if you defeat the Grafted Scion in the beginning of Elden Ring?',
    'I don\'t know where to go next in Elden Ring! How do I find out?',
    'What attributes should a samurai build focus on in Elden Ring?'
]

# ----------------------------
# 9️⃣ Run evaluation for both models
# ----------------------------
all_results = []

for query in queries:
    # Boolean OR search
    start_time = time.time()
    boolean_docs = boolean_search(query, boolean_index, operator="OR")
    boolean_time = time.time() - start_time

    # TF ranking search
    start_time = time.time()
    tf_docs = tf_ranking(query, tf_index, k=10)
    tf_time = time.time() - start_time

    # Simulate relevance (top-3 as relevant)
    relevant_docs = tf_docs[:3]  # assume TF-based ranking relevance
    prec5_bool = precision_at_k(list(boolean_docs), relevant_docs, 5)
    prec10_bool = precision_at_k(list(boolean_docs), relevant_docs, 10)
    ndcg5_bool = ndcg_at_k(list(boolean_docs), relevant_docs, 5)
    ndcg10_bool = ndcg_at_k(list(boolean_docs), relevant_docs, 10)

    prec5_tf = precision_at_k(tf_docs, relevant_docs, 5)
    prec10_tf = precision_at_k(tf_docs, relevant_docs, 10)
    ndcg5_tf = ndcg_at_k(tf_docs, relevant_docs, 5)
    ndcg10_tf = ndcg_at_k(tf_docs, relevant_docs, 10)

    all_results.append({
        "Query": query,
        "BooleanTime": boolean_time,
        "TFTime": tf_time,
        "Prec5_Boolean": prec5_bool,
        "Prec10_Boolean": prec10_bool,
        "nDCG5_Boolean": ndcg5_bool,
        "nDCG10_Boolean": ndcg10_bool,
        "Prec5_TF": prec5_tf,
        "Prec10_TF": prec10_tf,
        "nDCG5_TF": ndcg5_tf,
        "nDCG10_TF": ndcg10_tf
    })

results_df = pd.DataFrame(all_results)

# ----------------------------
# 10️⃣ Compute average metrics across 20 queries
# ----------------------------
avg_metrics = {
    "BooleanTime": results_df["BooleanTime"].mean(),
    "TFTime": results_df["TFTime"].mean(),
    "Prec5_Boolean": results_df["Prec5_Boolean"].mean(),
    "Prec10_Boolean": results_df["Prec10_Boolean"].mean(),
    "nDCG5_Boolean": results_df["nDCG5_Boolean"].mean(),
    "nDCG10_Boolean": results_df["nDCG10_Boolean"].mean(),
    "Prec5_TF": results_df["Prec5_TF"].mean(),
    "Prec10_TF": results_df["Prec10_TF"].mean(),
    "nDCG5_TF": results_df["nDCG5_TF"].mean(),
    "nDCG10_TF": results_df["nDCG10_TF"].mean()
}

print("\n===== Average Metrics Across 20 Queries =====")
for k, v in avg_metrics.items():
    print(f"{k}: {v:.4f}")
