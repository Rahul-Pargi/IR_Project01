# Information Retrieval System on Gaming StackExchange

## 📌 Project Overview
This project focuses on building and evaluating an **Information Retrieval (IR) System** using data from the `https://gaming.stackexchange.com/`. The system processes, indexes, and retrieves relevant questions and answers to simulate real-world information-seeking scenarios.  

The project demonstrates practical applications of IR concepts such as **indexing, querying, ranking, and evaluation**. The final evaluation uses a **set of 20 manually selected queries**, categorized by topics and query lengths, to measure system performance.

---

## ⚙️ Development Environment and Project Setup

- **Programming Language**: Python 3.x  
- **Libraries/Dependencies**:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - `lxml`
  - `sqlalchemy`
- **Dataset**: Gaming StackExchange data dump (`.7z` archives)  

## Getting Started  

Follow these steps to set up and run the project in **Google Colab** or a local environment.

### Step 1: Clone the Repository  
Run the following command to clone the repository:  
`!git clone https://github.com/Rahul-Pargi/IR_Project01`
### Step 2: Setup Environment & Install Dependencies:
Run the setup script to install required libraries and prepare the dataset (Posts.xml, Comments.xml):
`!python /content/IR_Project01/collab_setup.py`
### Step 3: Change Directory
Move into the project directory:
`Step 3: Change Directory`

## Running Analysis Scripts:

### Question 1: WordCloud & Zipf’s Law
`%run /content/IR_Project01/src/q1_wordcloud_zipf.py`
### Question 2: Most Common Tags
`!python /content/IR_Project01/src/q2_common_tags.py`
### Question 3 & 4: Average Question Length & Tags
`%run /content/IR_Project01/src/q3_q4_avg.py`
### Question 5: Unanswered Questions Analysis
`%run /content/IR_Project01/src/q5_no_answers.py`
### Question 6: Accepted Answer Positions
`%run /content/IR_Project01/src/q6_accepted_answers.py`
### Question 7: Readability of Questions
`%run /content/IR_Project01/src/q7_readability.py`
### Question 8: Duplicate Questions Detection
`%run /content/IR_Project01/src/q8_duplicate_ques.py`
### Question 9: Impact of Comments
`%run /content/IR_Project01/src/q9_comments.py`
### Question 10: Self-Analysis of IR Project
`%run /content/IR_Project01/src/q10_self_analysis.py`

## Retrieval Models
### Boolean Search System
`%run /content/IR_Project01/src/boolean_search_inverted_index.py`
### Term-Frequency Inverted Index Retrieval
`%run /content/IR_Project01/src/term_frequency_inverted_index.py`
### Evaluation of Boolean vs Term-Frequency Models
`%run /content/IR_Project01/src/boolean_tf_ir_evaluation.py`


