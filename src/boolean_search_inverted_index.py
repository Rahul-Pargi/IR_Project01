import os
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import string
import time

# -----------------------
# 0️⃣ Ensure required files exist
# -----------------------
posts_file = "data/Posts.xml"
if not os.path.exists(posts_file):
    raise FileNotFoundError(f"{posts_file} not found in current directory: {os.getcwd()}")

# -----------------------
# 1️⃣ Download NLTK stopwords
# -----------------------
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# -----------------------
# 2️⃣ Load and clean posts
# -----------------------
tree = ET.parse(posts_file)
root = tree.getroot()

posts = []
for row in root.findall("row"):
    try:
        post_id = int(row.attrib.get("Id"))
    except (ValueError, TypeError):
        continue
    body = row.attrib.get("Body", "")
    # Remove HTML tags
    text = BeautifulSoup(body, "html.parser").get_text()
    if text.strip():
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
# 4️⃣ Build inverted index
# -----------------------
start_time = time.time()
inverted_index = {}

for post_id, text in posts:
    tokens = preprocess(text)
    for token in tokens:
        inverted_index.setdefault(token, set()).add(post_id)

end_time = time.time()
print(f"Inverted index built with {len(inverted_index)} unique terms in {end_time - start_time:.2f} seconds")

# -----------------------
# 5️⃣ Boolean search function
# -----------------------
def boolean_search(query, operator="AND"):
    query_tokens = preprocess(query)
    if not query_tokens:
        return []

    # Get sets for each token
    sets = [inverted_index.get(t, set()) for t in query_tokens]

    if operator.upper() == "AND":
        result_set = set.intersection(*sets) if sets else set()
    elif operator.upper() == "OR":
        result_set = set.union(*sets)
    else:
        raise ValueError("Operator must be AND or OR")

    # Return top 50 post IDs
    return list(result_set)[:50]

# -----------------------
# 6️⃣ Example usage
# -----------------------
query1 = "Playstation"
results_and = boolean_search(query1, operator="AND")
results_or = boolean_search(query1, operator="OR")

print("AND results:", results_and)
print("OR results:", results_or)
