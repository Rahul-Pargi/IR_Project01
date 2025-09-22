# === IMPORTS ===
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd

# === DOWNLOAD NLTK RESOURCES ===
nltk.download('punkt')

# -----------------------
# Function to count words & sentences
# -----------------------
def count_words_sentences(text):
    text = str(text)  # ensure it's a string
    words = word_tokenize(text)
    sentences = nltk.sent_tokenize(text)
    return len(words), len(sentences)

# -----------------------
# 1️⃣ Compute words and sentences for questions
# -----------------------
posts_df['Body'] = posts_df['Body'].fillna('')
posts_df['Title'] = posts_df['Title'].fillna('')

posts_df['q_words'], posts_df['q_sentences'] = zip(*posts_df['Body'].map(count_words_sentences))
posts_df['t_words'], posts_df['t_sentences'] = zip(*posts_df['Title'].map(count_words_sentences))

# -----------------------
# 2️⃣ Filter answers
# PostTypeId = 2 → answer
# -----------------------
answers_df = posts_df[posts_df['PostTypeId'] == '2'].copy()
answers_df['a_words'], answers_df['a_sentences'] = zip(*answers_df['Body'].map(count_words_sentences))

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
answers_per_question = answers_df.groupby('ParentId').size()  # ParentId points to question Id
avg_answers_per_question = answers_per_question.mean()
print("\nAverage number of answers per question:", avg_answers_per_question)

# -----------------------
# 5️⃣ Number of questions with no answers
# -----------------------
question_ids = posts_df[posts_df['PostTypeId'] == '1']['Id']  # all question IDs
questions_with_answers = answers_df['ParentId'].unique()
questions_no_answers = set(question_ids) - set(questions_with_answers)
print("Number of questions with no answers:", len(questions_no_answers))

# -----------------------
# 6️⃣ Number of questions with accepted answers
# -----------------------
questions_with_accepted_answer = posts_df[posts_df['PostTypeId'] == '1']['AcceptedAnswerId'].dropna()
print("Number of questions with an accepted answer:", len(questions_with_accepted_answer))
