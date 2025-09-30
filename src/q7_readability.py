import pandas as pd
from bs4 import BeautifulSoup
import textstat
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# -----------------------
# 1️⃣ Load Posts.xml (absolute path for Colab)
# -----------------------
posts_path = "/content/IR_Project01/data/Posts.xml"

try:
    df = pd.read_xml(posts_path, xpath="//row")
except FileNotFoundError:
    raise FileNotFoundError(f"❌ Could not find Posts.xml at {posts_path}")

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
df['readability'] = df['Body_text'].apply(textstat.flesch_reading_ease)

# -----------------------
# 4️⃣ Separate answered and unanswered questions
# -----------------------
df['AnswerCount'] = pd.to_numeric(df['AnswerCount'], errors='coerce').fillna(0)
answered = df[df['AnswerCount'] > 0]
unanswered = df[df['AnswerCount'] == 0]

# -----------------------
# 5️⃣ Average readability for answered vs unanswered
# -----------------------
print("Average readability of answered questions:", answered['readability'].mean())
print("Average readability of unanswered questions:", unanswered['readability'].mean())

# -----------------------
# 6️⃣ Correlation between readability and number of answers
# -----------------------
corr, p_value = pearsonr(df['readability'], df['AnswerCount'])
print("Pearson correlation (readability vs AnswerCount):", corr)
print("p-value:", p_value)

# -----------------------
# 7️⃣ Scatter plot
# -----------------------
plt.figure(figsize=(10,5))
plt.scatter(df['readability'], df['AnswerCount'], alpha=0.3, color='purple')
plt.xlabel('Flesch Reading Ease (Readability)')
plt.ylabel('Number of Answers')
plt.title('Question Readability vs Number of Answers')
plt.grid(True, alpha=0.3)
plt.show()
