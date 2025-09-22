# ===============================
# Q7: Readability Analysis
# ===============================

import pandas as pd
from bs4 import BeautifulSoup
import textstat
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# -----------------------
# 0️⃣ Ensure df exists
# -----------------------
try:
    df
except NameError:
    raise NameError("DataFrame 'df' is not defined. Make sure Posts.xml is loaded into 'df'.")

# -----------------------
# 1️⃣ Ensure required columns exist
# -----------------------
for col in ['Body', 'AnswerCount']:
    if col not in df.columns:
        df[col] = 0 if col == 'AnswerCount' else ""

# -----------------------
# 2️⃣ Clean HTML from Body
# -----------------------
def clean_text(x):
    if pd.isna(x):
        return ""
    return BeautifulSoup(str(x), 'html.parser').get_text()

df['Body_text'] = df['Body'].apply(clean_text)

# -----------------------
# 3️⃣ Compute readability (Flesch Reading Ease)
# -----------------------
df['readability'] = df['Body_text'].apply(lambda x: textstat.flesch_reading_ease(x) if x else 0)

# -----------------------
# 4️⃣ Ensure AnswerCount is numeric
# -----------------------
df['AnswerCount'] = pd.to_numeric(df['AnswerCount'], errors='coerce').fillna(0)

# -----------------------
# 5️⃣ Separate answered and unanswered questions
# -----------------------
answered = df[df['AnswerCount'] > 0]
unanswered = df[df['AnswerCount'] == 0]

# -----------------------
# 6️⃣ Average readability for answered vs unanswered
# -----------------------
print("Average readability of answered questions:", answered['readability'].mean())
print("Average readability of unanswered questions:", unanswered['readability'].mean())

# -----------------------
# 7️⃣ Pearson correlation between readability and number of answers
# -----------------------
if len(df) > 1 and df['readability'].nunique() > 1:
    corr, p_value = pearsonr(df['readability'], df['AnswerCount'])
    print("Pearson correlation (readability vs AnswerCount):", corr)
    print("p-value:", p_value)
else:
    print("Not enough data or variation to compute correlation.")

# -----------------------
# 8️⃣ Scatter plot for visualization
# -----------------------
plt.figure(figsize=(10, 5))
plt.scatter(df['readability'], df['AnswerCount'], alpha=0.3, color='purple')
plt.xlabel('Flesch Reading Ease (Readability)')
plt.ylabel('Number of Answers')
plt.title('Question Readability vs Number of Answers')
plt.grid(True, alpha=0.3)
plt.show()

