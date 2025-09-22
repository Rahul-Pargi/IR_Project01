import os
import pandas as pd
import xml.etree.ElementTree as ET
import re

# -----------------------
# 0️⃣ Define file paths
# -----------------------
posts_file = "data/Posts.xml"
comments_file = "data/Comments.xml"

# Check if files exist
if not os.path.exists(posts_file):
    raise FileNotFoundError(f"{posts_file} not found in current directory: {os.getcwd()}")
if not os.path.exists(comments_file):
    raise FileNotFoundError(f"{comments_file} not found in current directory: {os.getcwd()}")

# -----------------------
# 1️⃣ Parse Posts
# -----------------------
posts_tree = ET.parse(posts_file)
posts_root = posts_tree.getroot()

posts_data = []
for row in posts_root.findall("row"):
    post_id = row.attrib.get("Id")
    title = row.attrib.get("Title", "")
    body = row.attrib.get("Body", "")
    posts_data.append({"Id": post_id, "Title": title, "Body": body})

df = pd.DataFrame(posts_data)
df['Id'] = df['Id'].astype(int)

# -----------------------
# 2️⃣ Parse Comments
# -----------------------
comments_tree = ET.parse(comments_file)
comments_root = comments_tree.getroot()

comments_data = []
for row in comments_root.findall("row"):
    cid = row.attrib.get("Id")
    pid = row.attrib.get("PostId")
    text = row.attrib.get("Text", "")
    comments_data.append({"CommentId": cid, "PostId": pid, "Text": text})

comments_df = pd.DataFrame(comments_data)
comments_df['PostId'] = comments_df['PostId'].astype(int)

# -----------------------
# 3️⃣ Merge Posts with Comments
# -----------------------
merged_df = df.merge(comments_df, left_on="Id", right_on="PostId", how="left")
posts_with_comments = merged_df[merged_df['Text'].notnull()]

# -----------------------
# 4️⃣ Sample 5 posts
# -----------------------
sample_posts = posts_with_comments[['Id', 'Title', 'Body', 'Text']].sample(5, random_state=42)

# -----------------------
# 5️⃣ Simple comment analysis function
# -----------------------
def analyze_comment(comment):
    comment_lower = comment.lower()

    # Askers clarifying question
    if re.search(r'\b(can|could|how|what|why|is it|does it|should)\b', comment_lower):
        return "Clarification request"
    # Adding details / edits
    elif re.search(r'\b(update|edit|added|also|more info|details|clarify)\b', comment_lower):
        return "Asker adds details"
    # Suggestions / hints / answers
    elif re.search(r'\b(try|use|you can|solution|answer|suggest|recommend)\b', comment_lower):
        return "Suggestion / hint / answer"
    else:
        return "Other / general comment"

# -----------------------
# 6️⃣ Print examples with analysis
# -----------------------
for post_id, group in sample_posts.groupby('Id'):
    title = group['Title'].iloc[0]
    body = group['Body'].iloc[0]
    print(f"\nPost ID: {post_id}, Title: {title}\nBody (first 200 chars): {body[:200]}...\n")

    for comment in group['Text']:
        analysis = analyze_comment(comment)
        print(f"Comment: {comment}")
        print(f"Analysis: {analysis}\n")
    print("-" * 100)

