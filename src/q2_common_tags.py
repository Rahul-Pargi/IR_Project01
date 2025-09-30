# === IMPORTS ===
from collections import Counter
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import os
from google.colab import files  # for downloading files

# === HELPER: load posts safely from XML ===
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

# === FILE PATHS (Colab-friendly) ===
project_dir = "/content/IR_Project01"  # adjust if your repo is elsewhere
data_dir = os.path.join(project_dir, "data")
posts_path = os.path.join(data_dir, "Posts.xml")

# === LOAD POSTS ===
print(f"Loading posts from {posts_path} ...")
posts_df = load_posts(posts_path)  # Load all posts
print("Loaded posts:", posts_df.shape)

if 'Tags' not in posts_df.columns:
    raise RuntimeError("No 'Tags' column found in Posts.xml rows. Check the XML structure.")

# === FUNCTION TO SPLIT TAGS ===
def extract_tags(tag_string):
    if pd.isna(tag_string):
        return []
    return [t for t in str(tag_string).split("|") if t]

# === APPLY FUNCTION ===
posts_df['Tags_list'] = posts_df['Tags'].apply(extract_tags)

# === FLATTEN ALL TAGS ===
all_tags = [tag for tags in posts_df['Tags_list'] for tag in tags]

# === COUNT FREQUENCY OF TAGS ===
tag_freq = Counter(all_tags)

# === TOP-10 MOST COMMON TAGS ===
top_10_tags = tag_freq.most_common(10)
print("Top-10 most common question tags:")
for i, (tag, count) in enumerate(top_10_tags, 1):
    print(f"{i}. {tag}: {count} occurrences")

# === TOP-20 FOR DISTRIBUTION + 'Other' CATEGORY ===
top_20_tags = tag_freq.most_common(20)
tags, counts = zip(*top_20_tags)

# Count "Other" tags
total_all_tags = sum(tag_freq.values())
count_top_20 = sum(counts)
count_other = total_all_tags - count_top_20

# Add "Other" category
tags_with_other = list(tags) + ["Other"]
counts_with_other = list(counts) + [count_other]

# === PLOT DISTRIBUTION ===
plt.figure(figsize=(12, 6))
plt.bar(tags_with_other, counts_with_other, color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.title("Distribution of Top-20 Question Tags (Others combined)")
plt.ylabel("Number of Questions")
plt.tight_layout()

# === SAVE FIGURE ===
output_path = "tag_distribution.png"
plt.savefig(output_path, dpi=300)
plt.show()

# === DOWNLOAD FIGURE ===
files.download(output_path)


# # === IMPORTS ===
# from collections import Counter
# import xml.etree.ElementTree as ET
# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# # === HELPER: load posts safely from XML ===
# def load_posts(file_path, max_rows=None):
#     rows = []
#     context = ET.iterparse(file_path, events=('end',))
#     for event, elem in context:
#         if elem.tag == 'row':
#             rows.append(elem.attrib.copy())
#             elem.clear()
#         if max_rows and len(rows) >= max_rows:
#             break
#     return pd.DataFrame(rows)

# # === FILE PATHS (Colab-friendly) ===
# project_dir = "/content/IR_Project01"  # adjust if your repo is elsewhere
# data_dir = os.path.join(project_dir, "data")
# posts_path = os.path.join(data_dir, "Posts.xml")

# # === LOAD POSTS ===
# # print(f"Loading posts from {posts_path} ...")
# # posts_df = load_posts(posts_path, max_rows=10000)
# # print("Loaded posts:", posts_df.shape)

# # === LOAD POSTS ===
# print(f"Loading posts from {posts_path} ...")
# posts_df = load_posts(posts_path)  # Load all posts
# print("Loaded posts:", posts_df.shape)

# if 'Tags' not in posts_df.columns:
#     raise RuntimeError("No 'Tags' column found in Posts.xml rows. Check the XML structure.")

# # === FUNCTION TO SPLIT TAGS ===
# def extract_tags(tag_string):
#     if pd.isna(tag_string):
#         return []
#     return [t for t in str(tag_string).split("|") if t]

# # === APPLY FUNCTION ===
# posts_df['Tags_list'] = posts_df['Tags'].apply(extract_tags)

# # === FLATTEN ALL TAGS ===
# all_tags = [tag for tags in posts_df['Tags_list'] for tag in tags]

# # === COUNT FREQUENCY OF TAGS ===
# tag_freq = Counter(all_tags)

# # === TOP-10 MOST COMMON TAGS ===
# top_10_tags = tag_freq.most_common(10)
# print("Top-10 most common question tags:")
# for i, (tag, count) in enumerate(top_10_tags, 1):
#     print(f"{i}. {tag}: {count} occurrences")

# # === TOP-20 FOR DISTRIBUTION + 'Other' CATEGORY ===
# top_20_tags = tag_freq.most_common(20)
# tags, counts = zip(*top_20_tags)

# # Count "Other" tags
# total_all_tags = sum(tag_freq.values())
# count_top_20 = sum(counts)
# count_other = total_all_tags - count_top_20

# # Add "Other" category
# tags_with_other = list(tags) + ["Other"]
# counts_with_other = list(counts) + [count_other]

# # === PLOT DISTRIBUTION ===
# plt.figure(figsize=(12, 6))
# plt.bar(tags_with_other, counts_with_other, color='skyblue')
# plt.xticks(rotation=45, ha='right')
# plt.title("Distribution of Top-20 Question Tags (Others combined)")
# plt.ylabel("Number of Questions")
# plt.tight_layout()
# plt.show()



