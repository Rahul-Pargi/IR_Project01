import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# 0️⃣ Ensure 'Body' and 'Title' are filled
# -----------------------
posts_df['Body'] = posts_df['Body'].fillna('')
posts_df['Title'] = posts_df['Title'].fillna('')

# -----------------------
# 1️⃣ Ensure Tags_list exists
# -----------------------
if 'Tags_list' not in posts_df.columns:
    def extract_tags(tag_string):
        if pd.isna(tag_string):
            return []
        return [t for t in str(tag_string).split("|") if t]
    posts_df['Tags_list'] = posts_df['Tags'].apply(extract_tags)

# -----------------------
# 2️⃣ Ensure word counts exist
# -----------------------
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

def count_words(text):
    return len(word_tokenize(text))

if 'q_words' not in posts_df.columns:
    posts_df['q_words'] = posts_df['Body'].astype(str).map(count_words)
if 't_words' not in posts_df.columns:
    posts_df['t_words'] = posts_df['Title'].astype(str).map(count_words)

# -----------------------
# 3️⃣ Filter questions
# -----------------------
questions_df = posts_df[posts_df['PostTypeId'] == '1'].copy()

# -----------------------
# 4️⃣ Count number of answers per question
# -----------------------
answers_df = posts_df[posts_df['PostTypeId'] == '2'].copy()
answers_count = answers_df.groupby('ParentId').size()
questions_df['num_answers'] = questions_df['Id'].map(answers_count).fillna(0).astype(int)

# -----------------------
# 5️⃣ Examples of unanswered questions
# -----------------------
unanswered = questions_df[questions_df['num_answers'] == 0]
sample_unanswered = unanswered.sample(min(5, len(unanswered)), random_state=42)

print("Examples of unanswered questions (first 300 characters):\n")
for i, row in sample_unanswered.iterrows():
    print(f"Question ID: {row['Id']}")
    print(row['Body'][:300])
    print("---\n")

# -----------------------
# 6️⃣ Average lengths
# -----------------------
avg_unanswered_len = unanswered['q_words'].mean()
avg_answered_len = questions_df[questions_df['num_answers'] > 0]['q_words'].mean()

avg_unanswered_title_len = unanswered['t_words'].mean()
avg_answered_title_len = questions_df[questions_df['num_answers'] > 0]['t_words'].mean()

print(f"Average length (words) of unanswered questions: {avg_unanswered_len:.2f}")
print(f"Average length (words) of answered questions: {avg_answered_len:.2f}")
print(f"Average title length of unanswered questions: {avg_unanswered_title_len:.2f}")
print(f"Average title length of answered questions: {avg_answered_title_len:.2f}")

# -----------------------
# 7️⃣ Average number of tags
# -----------------------
avg_unanswered_tags = unanswered['Tags_list'].apply(len).mean()
avg_answered_tags = questions_df[questions_df['num_answers'] > 0]['Tags_list'].apply(len).mean()

print(f"Average number of tags for unanswered questions: {avg_unanswered_tags:.2f}")
print(f"Average number of tags for answered questions: {avg_answered_tags:.2f}")

# -----------------------
# 8️⃣ Plot histograms for question lengths
# -----------------------
answered_questions = questions_df[questions_df['num_answers'] > 0]

plt.figure(figsize=(12, 6))
plt.hist(answered_questions['q_words'], bins=50, alpha=0.6, label='Answered Questions', color='green')
plt.hist(unanswered['q_words'], bins=50, alpha=0.6, label='Unanswered Questions', color='red')
plt.title('Distribution of Question Lengths (Words) – Answered vs Unanswered')
plt.xlabel('Number of Words in Question Body')
plt.ylabel('Number of Questions')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()
