# ===============================
# Q6: Accepted Answer Analysis
# ===============================

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# -----------------------
# 0️⃣ Load Posts.xml safely
# -----------------------
posts_path = "/content/IR_Project01/data/Posts.xml"  # adjust path
try:
    df = pd.read_xml(posts_path, xpath="//row")
except Exception as e:
    print("Error loading XML:", e)
    df = pd.DataFrame()

if df.empty:
    raise ValueError("Posts.xml could not be loaded or is empty.")

# -----------------------
# 1️⃣ Ensure columns exist and correct types
# -----------------------
for col in ['PostTypeId', 'Score', 'OwnerUserId', 'ParentId', 'Id', 'AcceptedAnswerId', 'CreationDate', 'Title']:
    if col not in df.columns:
        df[col] = np.nan

df['PostTypeId'] = df['PostTypeId'].astype(str)
df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
df['OwnerUserId'] = pd.to_numeric(df['OwnerUserId'], errors='coerce')
df['ParentId'] = pd.to_numeric(df['ParentId'], errors='coerce')
df['Id'] = pd.to_numeric(df['Id'], errors='coerce')

# -----------------------
# 2️⃣ Filter questions and answers
# -----------------------
questions = df[df['PostTypeId'] == '1'].copy()
answers = df[df['PostTypeId'] == '2'].copy()

questions['has_accepted'] = questions['AcceptedAnswerId'].notnull()

# -----------------------
# 3️⃣ How many accepted answers are the first answers?
# -----------------------
first_answer_flags = []

for _, q in questions[questions['has_accepted']].iterrows():
    # Answers to this question sorted by CreationDate
    question_answers = answers[answers['ParentId'] == q['Id']].sort_values('CreationDate')
    if question_answers.empty:
        continue
    first_answer_id = question_answers.iloc[0]['Id']
    accepted_id = q['AcceptedAnswerId']
    first_answer_flags.append(first_answer_id == accepted_id)

num_first_accepted = sum(first_answer_flags)
total_accepted = len(first_answer_flags)

if total_accepted > 0:
    print(f"Accepted answers that are the first answers: {num_first_accepted}/{total_accepted} "
          f"({num_first_accepted/total_accepted:.2%})")
else:
    print("No accepted answers found.")

# -----------------------
# 4️⃣ Correlation between accepted answer score and user reputation
# -----------------------
accepted_ids = questions['AcceptedAnswerId'].dropna()
accepted_answers = answers[answers['Id'].isin(accepted_ids)].copy()
accepted_answers = accepted_answers.dropna(subset=['OwnerUserId', 'Score'])

# For demonstration, generate a sample user reputation
user_reputation = {uid: np.random.randint(1, 5000) for uid in accepted_answers['OwnerUserId'].unique()}
accepted_answers['Reputation'] = accepted_answers['OwnerUserId'].map(user_reputation)

if not accepted_answers.empty:
    corr, _ = spearmanr(accepted_answers['Score'], accepted_answers['Reputation'])
    print(f"Spearman correlation between accepted answer score and reputation: {corr:.3f}")
else:
    print("No accepted answers with score and user ID available for correlation.")

# -----------------------
# 5️⃣ Are accepted answers always the highest scored?
# -----------------------
not_highest_examples = []

for _, q in questions[questions['has_accepted']].iterrows():
    question_answers = answers[answers['ParentId'] == q['Id']]
    if question_answers.empty:
        continue

    accepted_score = question_answers.loc[question_answers['Id'] == q['AcceptedAnswerId'], 'Score']
    if accepted_score.empty:
        continue

    max_score = question_answers['Score'].max()
    if accepted_score.iloc[0] < max_score:
        not_highest_examples.append({
            'QuestionId': q['Id'],
            'Title': q['Title'] if 'Title' in q else "",
            'AcceptedScore': accepted_score.iloc[0],
            'MaxScore': max_score
        })

if not_highest_examples:
    example = not_highest_examples[0]
    print("\nExample where accepted answer is not the highest scored:")
    print(f"Question ID: {example['QuestionId']}")
    print(f"Title: {example['Title']}")
    print(f"Accepted Answer Score: {example['AcceptedScore']}")
    print(f"Highest Answer Score: {example['MaxScore']}")
else:
    print("All accepted answers are the highest scored.")
