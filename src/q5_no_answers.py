# === IMPORTS ===
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Ensure required columns exist
# -----------------------
# Assuming posts_df already exists and contains 'PostTypeId', 'Body', 'Title', 'Tags_list', 'q_words', 't_words'
posts_df['Body'] = posts_df['Body'].fillna('')
posts_df['Title'] = posts_df['Title'].fillna('')
if 'Tags_list' not in posts_df.columns:
    posts_df['Tags_list'] = [[] for _ in range(len(posts_df))]  # empty list if not present

# -----------------------
# Filter questions
# -----------------------
questions_df = posts_df[posts_df['PostTypeId'] == '1'].copy()

# -----------------------
# 1️⃣ Count number of answers per question
# -----------------------
answers_df = posts_df[posts_df['PostTypeId'] == '2'].copy()
answers_count = answers_df.groupby('ParentId').size()
questions_df['num_answers'] = questions_df['Id'].map(answers_count).fillna(0).astype(int)

# -----------------------
# 2️⃣ Select examples of unanswered questions
# -----------------------
unanswered = questions_df[questions_df['num_answers'] == 0]

# Take a random sample of 5 for inspection
sample_unanswered = unanswered.sample(min(5, len(unanswered)), random_state=42)

print("Examples of unanswered questions (first 300 characters):\n")
for i, row in sample_unanswered.iterrows():
    print(f"Question ID: {row['Id']}")
    print(row['Body'][:300])  # show first 300 characters
    print("---\n")

# -----------------------
# 3️⃣ Compare average lengths
# -----------------------
avg_unanswered_len = unanswered['q_words'].mean()
answered_questions_df = questions_df[questions_df['num_answers'] > 0]
avg_answered_len = answered_questions_df['q_words'].mean()

print(f"Average length (words) of unanswered questions: {avg_unanswered_len:.2f}")
print(f"Average length (words) of answered questions: {avg_answered_len:.2f}")

# -----------------------
# 4️⃣ Optional: Inspect title length
# -----------------------
avg_unanswered_title_len = unanswered['t_words'].mean()
avg_answered_title_len = answered_questions_df['t_words'].mean()

print(f"Average title length of unanswered questions: {avg_unanswered_title_len:.2f}")
print(f"Average title length of answered questions: {avg_answered_title_len:.2f}")

# -----------------------
# 5️⃣ Optional: Check tags
# -----------------------
unanswered_tags = unanswered['Tags_list'].apply(len).mean()
answered_tags = answered_questions_df['Tags_list'].apply(len).mean()

print(f"Average number of tags for unanswered questions: {unanswered_tags:.2f}")
print(f"Average number of tags for answered questions: {answered_tags:.2f}")

# -----------------------
# 6️⃣ Plot distributions
# -----------------------
plt.figure(figsize=(12, 6))
plt.hist(answered_questions_df['q_words'], bins=50, alpha=0.6, label='Answered Questions', color='green')
plt.hist(unanswered['q_words'], bins=50, alpha=0.6, label='Unanswered Questions', color='red')
plt.title('Distribution of Question Lengths (Words) – Answered vs Unanswered')
plt.xlabel('Number of Words in Question Body')
plt.ylabel('Number of Questions')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()
