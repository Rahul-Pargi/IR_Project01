import pandas as pd
import xml.etree.ElementTree as ET
import re
from html import unescape
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

# === HELPER FUNCTIONS ===
def clean_html(text):
    if not isinstance(text, str):
        return ''
    text = unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_posts(file_path, max_rows=None):
    rows = []
    context = ET.iterparse(file_path, events=('end',))
    for event, elem in context:
        if elem.tag == 'row':
            rows.append(elem.attrib.copy())
            elem.clear()
        if max_rows and len(rows) >= max_rows:
            break
    return pd.DataFrame(rows)

def count_words_sentences(text):
    words = word_tokenize(text)
    sentences = nltk.sent_tokenize(text)
    return len(words), len(sentences)

# === LOAD DATA ===
posts_df = load_posts("/content/IR_Project01/data/Posts.xml", max_rows=10000)
posts_df['Body'] = posts_df['Body'].fillna('')
posts_df['Title'] = posts_df['Title'].fillna('')
if 'Tags_list' not in posts_df.columns:
    posts_df['Tags_list'] = [[] for _ in range(len(posts_df))]

# === COMPUTE WORD & SENTENCE COUNTS (needed for Q5) ===
posts_df['q_words'], posts_df['q_sentences'] = zip(*posts_df['Body'].astype(str).map(count_words_sentences))
posts_df['t_words'], posts_df['t_sentences'] = zip(*posts_df['Title'].astype(str).map(count_words_sentences))

# === FILTER QUESTIONS ===
questions_df = posts_df[posts_df['PostTypeId'] == '1'].copy()

# === COUNT NUMBER OF ANSWERS PER QUESTION ===
answers_df = posts_df[posts_df['PostTypeId'] == '2'].copy()
answers_count = answers_df.groupby('ParentId').size()
questions_df['num_answers'] = questions_df['Id'].map(answers_count).fillna(0).astype(int)

# === UNANSWERED QUESTIONS SAMPLE ===
unanswered = questions_df[questions_df['num_answers'] == 0]
sample_unanswered = unanswered.sample(min(5, len(unanswered)), random_state=42)

print("Examples of unanswered questions (first 300 characters):\n")
for i, row in sample_unanswered.iterrows():
    print(f"Question ID: {row['Id']}")
    print(row['Body'][:300])
    print("---\n")

# === AVERAGE LENGTHS ===
avg_unanswered_len = unanswered['q_words'].mean()
answered_questions_df = questions_df[questions_df['num_answers'] > 0]
avg_answered_len = answered_questions_df['q_words'].mean()

print(f"Average length (words) of unanswered questions: {avg_unanswered_len:.2f}")
print(f"Average length (words) of answered questions: {avg_answered_len:.2f}")

# Optional: title lengths
avg_unanswered_title_len = unanswered['t_words'].mean()
avg_answered_title_len = answered_questions_df['t_words'].mean()

print(f"Average title length of unanswered questions: {avg_unanswered_title_len:.2f}")
print(f"Average title length of answered questions: {avg_answered_title_len:.2f}")

# Optional: tags
unanswered_tags = unanswered['Tags_list'].apply(len).mean()
answered_tags = answered_questions_df['Tags_list'].apply(len).mean()

print(f"Average number of tags for unanswered questions: {unanswered_tags:.2f}")
print(f"Average number of tags for answered questions: {answered_tags:.2f}")

# === PLOT HISTOGRAMS ===
plt.figure(figsize=(12, 6))
plt.hist(answered_questions_df['q_words'], bins=50, alpha=0.6, label='Answered Questions', color='green')
plt.hist(unanswered['q_words'], bins=50, alpha=0.6, label='Unanswered Questions', color='red')
plt.title('Distribution of Question Lengths (Words) – Answered vs Unanswered')
plt.xlabel('Number of Words in Question Body')
plt.ylabel('Number of Questions')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()
