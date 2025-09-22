import os
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import textstat
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# 0️⃣ Define file path
# -----------------------
posts_file = "data/Posts.xml"

# Check if file exists
if not os.path.exists(posts_file):
    raise FileNotFoundError(f"{posts_file} not found in current directory: {os.getcwd()}")

# -----------------------
# 1️⃣ Load Posts.xml
# -----------------------
df = pd.read_xml(posts_file, parser="lxml")

# Filter questions and answers
questions_df = df[df['PostTypeId'] == 1].copy()
answers_df = df[df['PostTypeId'] == 2].copy()

# Convert to datetime
questions_df['CreationDate'] = pd.to_datetime(questions_df['CreationDate'])
answers_df['CreationDate'] = pd.to_datetime(answers_df['CreationDate'])

# -----------------------
# 2️⃣ First answer time
# -----------------------
first_answers = answers_df.groupby('ParentId')['CreationDate'].min().reset_index()
first_answers.rename(columns={'CreationDate': 'first_answer_time'}, inplace=True)

# Merge with questions
questions_df = questions_df.merge(first_answers, left_on='Id', right_on='ParentId', how='left')

# Compute time to first answer in hours
questions_df['time_to_answer'] = (
    (questions_df['first_answer_time'] - questions_df['CreationDate'])
    .dt.total_seconds() / 3600
)

# -----------------------
# 3️⃣ Compute features for analysis
# -----------------------

# Clean HTML from body
def clean_text(html):
    if pd.isna(html):
        return ""
    return BeautifulSoup(str(html), 'html.parser').get_text()

questions_df['Body_text'] = questions_df['Body'].apply(clean_text)

# Question length (words)
questions_df['body_word_count'] = questions_df['Body_text'].apply(lambda x: len(x.split()))
questions_df['title_word_count'] = questions_df['Title'].astype(str).apply(lambda x: len(x.split()))

# Number of tags
def count_tags(tags):
    if pd.isna(tags):
        return 0
    return len(str(tags).split('|'))

questions_df['num_tags'] = questions_df['Tags'].apply(count_tags)

# Readability
questions_df['readability'] = questions_df['Body_text'].apply(lambda x: textstat.flesch_reading_ease(x) if x else np.nan)

# -----------------------
# 4️⃣ Analysis: Correlation with time to first answer
# -----------------------
features = ['body_word_count', 'title_word_count', 'num_tags', 'readability']
correlations = questions_df[features + ['time_to_answer']].corr()['time_to_answer'].drop('time_to_answer')

print("Correlation of question features with time to first answer (hours):")
print(correlations)

# -----------------------
# 5️⃣ Optional: Visualize patterns
# -----------------------
plt.figure(figsize=(16, 10))
for i, feature in enumerate(features, 1):
    plt.subplot(2, 2, i)
    sns.scatterplot(x=questions_df[feature], y=questions_df['time_to_answer'])
    plt.title(f'{feature} vs time to first answer')
    plt.xlabel(feature)
    plt.ylabel('Time to first answer (hours)')
plt.tight_layout()
plt.show()
