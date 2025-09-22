# === IMPORTS ===
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
import os

# === DOWNLOAD NLTK DATA ===
nltk.download('punkt')

# === HELPER FUNCTION ===
def count_words_sentences(text):
    if not isinstance(text, str):
        text = ''
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    return len(words), len(sentences)

# === LOAD POSTS (ensure Posts.xml is used) ===
data_dir = "/content/IR_Project01/data"
posts_path = os.path.join(data_dir, "Posts.xml")

if not os.path.exists(posts_path):
    raise FileNotFoundError(f"Posts.xml not found in {data_dir}")

# Iterparse loading
import xml.etree.ElementTree as ET
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

posts_df = load_posts(posts_path, max_rows=10000)

# Ensure necessary columns exist
for col in ['Body', 'Title', 'PostTypeId', 'AcceptedAnswerId', 'Id']:
    if col not in posts_df.columns:
        posts_df[col] = None  # fill missing columns

# -----------------------
# 1️⃣ Compute words and sentences for questions
# -----------------------
posts_df['q_words'], posts_df['q_sentences'] = zip(*posts_df['Body'].astype(str).map(count_words_sentences))
posts_df['t_words'], posts_df['t_sentences'] = zip(*posts_df['Title'].astype(str).map(count_words_sentences))

# -----------------------
# 2️⃣ Filter answers (PostTypeId = '2' → answer)
# -----------------------
answers_df = posts_df[posts_df['PostTypeId'] == '2'].copy()
answers_df['a_words'], answers_df['a_sentences'] = zip(*answers_df['Body'].astype(str).map(count_words_sentences))

# -----------------------
# 3️⃣ Compute averages for words & sentences
# -----------------------
print("Average number of words and sentences:")
print("Questions (Body):", round(posts_df['q_words'].mean(), 2), "words,", round(posts_df['q_sentences'].mean(), 2), "sentences")
print("Questions (Title):", round(posts_df['t_words'].mean(), 2), "words,", round(posts_df['t_sentences'].mean(), 2), "sentences")
print("Answers (Body):", round(answers_df['a_words'].mean(), 2), "words,", round(answers_df['a_sentences'].mean(), 2), "sentences")

# -----------------------
# 4️⃣ Average number of answers per question
# -----------------------
answers_per_question = answers_df.groupby('ParentId').size()
avg_answers_per_question = answers_per_question.mean() if not answers_per_question.empty else 0
print("\nAverage number of answers per question:", round(avg_answers_per_question, 2))

# -----------------------
# 5️⃣ Number of questions with no answers
# -----------------------
question_ids = posts_df[posts_df['PostTypeId'] == '1']['Id'].dropna()
questions_with_answers = answers_df['ParentId'].dropna().unique()
questions_no_answers = set(question_ids) - set(questions_with_answers)
print("Number of questions with no answers:", len(questions_no_answers))

# -----------------------
# 6️⃣ Number of questions with accepted answers
# -----------------------
questions_with_accepted_answer = posts_df[posts_df['PostTypeId'] == '1']['AcceptedAnswerId'].dropna()
print("Number of questions with an accepted answer:", len(questions_with_accepted_answer))
