import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import string

# -----------------------
# 1️⃣ Setup
# -----------------------
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -----------------------
# 2️⃣ Load Posts.xml with pandas
# -----------------------
posts_path = "/content/IR_Project01/data/Posts.xml"

try:
    df = pd.read_xml(posts_path, xpath="//row")
except FileNotFoundError:
    raise FileNotFoundError(f"❌ Could not find Posts.xml at {posts_path}")

# Keep only useful columns
df = df[['Id', 'Title', 'Body']].fillna("")

# -----------------------
# 3️⃣ Normalize text function
# -----------------------
def normalize_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [word for word in word_tokenize(text) if word.isalpha() and word not in stop_words]
    return tokens

# -----------------------
# 4️⃣ Precompute normalized text
# -----------------------
df['norm_title'] = df['Title'].apply(normalize_text)
df['norm_body'] = df['Body'].apply(normalize_text)

# Combine for TF-IDF
df['combined_text'] = df['Title'].astype(str) + " " + df['Body'].astype(str)

# -----------------------
# 5️⃣ TF-IDF vectorization
# -----------------------
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['combined_text'])

# -----------------------
# 6️⃣ Nearest neighbors (approximate)
# -----------------------
nn = NearestNeighbors(metric='cosine', algorithm='auto')
nn.fit(X)

batch_size = min(len(df), 5000)  # avoid OOM
distances, indices = nn.kneighbors(X[:batch_size], n_neighbors=6)

# -----------------------
# 7️⃣ Find duplicate pairs
# -----------------------
duplicate_pairs = []
threshold = 0.8  # cosine similarity > 0.8

for i, neighbors in enumerate(indices):
    for j, idx in enumerate(neighbors[1:]):  # skip self
        sim = 1 - distances[i][j+1]
        if sim > threshold:
            pair = tuple(sorted((i, idx)))
            if pair not in duplicate_pairs:
                duplicate_pairs.append(pair)

print(f"Total duplicate question pairs found (subset of {batch_size}): {len(duplicate_pairs)}")

# -----------------------
# 8️⃣ Count common terms
# -----------------------
for i, j in duplicate_pairs[:5]:  # show first 5 examples
    title_i = set(df.loc[i, 'norm_title'])
    title_j = set(df.loc[j, 'norm_title'])
    body_i = set(df.loc[i, 'norm_body'])
    body_j = set(df.loc[j, 'norm_body'])

    common_title = title_i.intersection(title_j)
    common_body = body_i.intersection(body_j)

    print(f"Question {df.loc[i, 'Id']} and Question {df.loc[j, 'Id']} are duplicates")
    print(f"Title common terms ({len(common_title)}): {common_title}")
    print(f"Body common terms ({len(common_body)}): {common_body}")
    print("-" * 80)
