# ===============================
# Q5: Analyze unanswered questions
# Standalone version
# ===============================

import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize

# -----------------------
# Download NLTK resources
# -----------------------
nltk.download('punkt')

# -----------------------
# Helper: Load Posts.xml safely
# -----------------------
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

# # -----------------------
# # File path to your Posts.xml
# # -----------------------
# posts_path = "/content/IR_Project01/data/Posts.xml"  # adjust path
# posts_df = load_posts(posts_path, max_rows=10000)

# -----------------------
# File path to your Posts.xml
# -----------------------
posts_path = "/content/IR_Project01/data/Posts.xml"  # adjust path
posts_df = load_posts(posts_path, max_rows=None)  # <-- Load all posts


# -----------------------
# Ensure required columns exist
# -----------------------
for col in ['Body', 'Title', 'Tags']:
    if col not in posts_df.columns:
        posts_df[col] = ''

posts_df['Body'] = posts_df['Body'].fillna('')
posts_df['Title'] = posts_df['Title'].fillna('')
posts_df['Tags'] = posts_df['Tags'].fillna('')

# -----------------------
# Compute word counts for Body and Title
# -----------------------
def count_words(text):
    return len(word_tokenize(text))

posts_df['q_words'] = posts_df['Body'].map(count_words)
posts_df['t_words'] = posts_df['Title'].map(count_words)

# -----------------------
# Process Tags: convert pipe-separated string to list
# -----------------------
def extract_tags(tag_string):
    if pd.isna(tag_string) or tag_string == '':
        return []
    return tag_string.split('|')

posts_df['Tags_list'] = posts_df['Tags'].map(extract_tags)

# -----------------------
# Filter only questions
# -----------------------
questions_df = posts_df[posts_df['PostTypeId'] == '1'].copy()

# -----------------------
# Count number of answers per question
# -----------------------
answers_df = posts_df[posts_df['PostTypeId'] == '2'].copy()
answers_count = answers_df.groupby('ParentId').size()
questions_df['num_answers'] = questions_df['Id'].map(answers_count).fillna(0).astype(int)

# -----------------------
# Select unanswered questions
# -----------------------
unanswered = questions_df[questions_df['num_answers'] == 0]

# Random sample of 5 unanswered questions
sample_unanswered = unanswered.sample(5, random_state=42)

print("Examples of unanswered questions (first 300 characters):\n")
for i, row in sample_unanswered.iterrows():
    print(f"Question ID: {row['Id']}")
    print(row['Body'][:300])
    print("---\n")

# -----------------------
# Average lengths
# -----------------------
avg_unanswered_len = unanswered['q_words'].mean()
avg_answered_len = questions_df[questions_df['num_answers'] > 0]['q_words'].mean()
avg_unanswered_title_len = unanswered['t_words'].mean()
avg_answered_title_len = questions_df[questions_df['num_answers'] > 0]['t_words'].mean()

# Average number of tags
avg_unanswered_tags = unanswered['Tags_list'].apply(len).mean()
avg_answered_tags = questions_df[questions_df['num_answers'] > 0]['Tags_list'].apply(len).mean()

print(f"Average length (words) of unanswered questions: {avg_unanswered_len:.2f}")
print(f"Average length (words) of answered questions: {avg_answered_len:.2f}")
print(f"Average title length of unanswered questions: {avg_unanswered_title_len:.2f}")
print(f"Average title length of answered questions: {avg_answered_title_len:.2f}")
print(f"Average number of tags for unanswered questions: {avg_unanswered_tags:.2f}")
print(f"Average number of tags for answered questions: {avg_answered_tags:.2f}")

# -----------------------
# Plot histograms: question body length
# -----------------------
answered_questions = questions_df[questions_df['num_answers'] > 0]
unanswered_questions = questions_df[questions_df['num_answers'] == 0]

plt.figure(figsize=(12, 6))
plt.hist(answered_questions['q_words'], bins=50, alpha=0.6, label='Answered Questions', color='green')
plt.hist(unanswered_questions['q_words'], bins=50, alpha=0.6, label='Unanswered Questions', color='red')
plt.title('Distribution of Question Lengths (Words) – Answered vs Unanswered')
plt.xlabel('Number of Words in Question Body')
plt.ylabel('Number of Questions')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()
