# === IMPORTS ===
import os
import xml.etree.ElementTree as ET
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

# === DOWNLOAD NLTK DATA ===
nltk.download('punkt')

# -----------------------
# Helper: load posts safely from XML
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

# -----------------------
# Load Posts.xml into posts_df
# -----------------------
# data_dir = os.path.join(os.getcwd(), "data")  # adjust if needed
# posts_path = os.path.join(data_dir, "Posts.xml")
# posts_df = load_posts(posts_path, max_rows=10000)

# # Fill missing Body and Title to avoid errors
# posts_df['Body'] = posts_df['Body'].fillna('')
# posts_df['Title'] = posts_df['Title'].fillna('')

# -----------------------
# Load Posts.xml into posts_df
# -----------------------
data_dir = os.path.join(os.getcwd(), "data")  # adjust if needed
posts_path = os.path.join(data_dir, "Posts.xml")
posts_df = load_posts(posts_path)  # Load all posts

# Fill missing Body and Title to avoid errors
posts_df['Body'] = posts_df['Body'].fillna('')
posts_df['Title'] = posts_df['Title'].fillna('')

# -----------------------
# Function to count words & sentences
# -----------------------
def count_words_sentences(text):
    words = word_tokenize(text)
    sentences = nltk.sent_tokenize(text)
    return len(words), len(sentences)

# -----------------------
# 1️⃣ Compute words and sentences for questions
# -----------------------
posts_df['q_words'], posts_df['q_sentences'] = zip(*posts_df['Body'].astype(str).map(count_words_sentences))
posts_df['t_words'], posts_df['t_sentences'] = zip(*posts_df['Title'].astype(str).map(count_words_sentences))

# -----------------------
# 2️⃣ Filter answers
# PostTypeId = 2 → answer
# -----------------------
answers_df = posts_df[posts_df['PostTypeId'] == '2'].copy()
answers_df['a_words'], answers_df['a_sentences'] = zip(*answers_df['Body'].astype(str).map(count_words_sentences))

# -----------------------
# 3️⃣ Compute averages for words & sentences
# -----------------------
print("Average number of words and sentences:")
print("Questions (Body):", posts_df['q_words'].mean(), "words,", posts_df['q_sentences'].mean(), "sentences")
print("Questions (Title):", posts_df['t_words'].mean(), "words,", posts_df['t_sentences'].mean(), "sentences")
print("Answers (Body):", answers_df['a_words'].mean(), "words,", answers_df['a_sentences'].mean(), "sentences")

# -----------------------
# 4️⃣ Average number of answers per question
# -----------------------
answers_per_question = answers_df.groupby('ParentId').size()
avg_answers_per_question = answers_per_question.mean()
print("\nAverage number of answers per question:", avg_answers_per_question)

# -----------------------
# 5️⃣ Number of questions with no answers
# -----------------------
question_ids = posts_df[posts_df['PostTypeId'] == '1']['Id']
questions_with_answers = answers_df['ParentId'].unique()
questions_no_answers = set(question_ids) - set(questions_with_answers)
print("Number of questions with no answers:", len(questions_no_answers))

# -----------------------
# 6️⃣ Number of questions with accepted answers
# -----------------------
questions_with_accepted_answer = posts_df[posts_df['PostTypeId'] == '1']['AcceptedAnswerId'].dropna()
print("Number of questions with an accepted answer:", len(questions_with_accepted_answer))

# # -----------------------
# # 7️⃣ Optional: Examples of unanswered questions (first 300 chars)
# # -----------------------
# unanswered_questions = posts_df[posts_df['Id'].isin(questions_no_answers)]
# print("\nExamples of unanswered questions (first 300 characters):")
# for i, text in enumerate(unanswered_questions['Body'].astype(str).head(5)):
#     print(f"{i+1}. {text[:300]}\n")
