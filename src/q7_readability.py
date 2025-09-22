import pandas as pd
from bs4 import BeautifulSoup
import textstat
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# -----------------------
# 0️⃣ Load data (ensure df is available)
# -----------------------
try:
    df
except NameError:
    df = pd.read_xml("Posts.xml", xpath="//row")

# -----------------------
# 1️⃣ Clean HTML from Body
# -----------------------
def clean_text(x):
    if pd.isna(x):
        return ""
    return BeautifulSoup(str(x), 'html.parser').get_text()

df['Body_text'] = df['Body'].apply(clean_text)

# -----------------------
# 2️⃣ Compute readability (Flesch Reading Ease)
# -----------------------
df['readability'] = df['Body_text'].apply(textstat.flesch_reading_ease)

# -----------------------
# 3️⃣ Separate answered and unanswered questions
# -----------------------
answered = df[df['AnswerCount'] > 0]
unanswered = df[df['AnswerCount'] == 0]

# -----------------------
# 4️⃣ Average readability for answered vs unanswered
# -----------------------
print("Average readability of answered questions:", answered['readability'].mean())
print("Average readability of unanswered questions:", unanswered['readability'].mean())

# -----------------------
# 5️⃣ Correlation between readability and number of answers
# -----------------------
df['AnswerCount'] = pd.to_numeric(df['AnswerCount'], errors='coerce').fillna(0)

corr, p_value = pearsonr(df['readability'], df['AnswerCount'])
print("Pearson correlation (readability vs AnswerCount):", corr)
print("p-value:", p_value)

# -----------------------
# 6️⃣ Scatter plot for visualization
# -----------------------
plt.figure(figsize=(10,5))
plt.scatter(df['readability'], df['AnswerCount'], alpha=0.3, color='purple')
plt.xlabel('Flesch Reading Ease (Readability)')
plt.ylabel('Number of Answers')
plt.title('Question Readability vs Number of Answers')
plt.grid(True, alpha=0.3)
plt.show()
