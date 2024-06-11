import pandas as pd
from rank_bm25 import BM25Okapi
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('scratch/n11357738/cleaned_subtopic_and_content.csv')

# Define the categories
categories = {
    1: "Requests/suggestions for editing coordination",
    2: "Requests for information",
    3: "References to vandalism",
    4: "References to Wikipedia guidelines and policies",
    5: "References to internal Wikipedia resources",
    6: "Off-topic remarks",
    7: "Polls",
    8: "Requests for peer review",
    9: "Information boxes",
    10: "Images",
    11: "Other"
}

# List of category names as queries
category_queries = list(categories.values())

# Tokenize discussions and categories
tokenized_discussions = [doc.split() for doc in df['relevant_content'].tolist()]
tokenized_categories = [query.split() for query in category_queries]

# Initialize BM25
bm25 = BM25Okapi(tokenized_discussions)

# Calculate BM25 scores and assign categories
category_assignments = []
for discussion in tokenized_discussions:
    scores = [bm25.get_scores(query) for query in tokenized_categories]
    scores = [score[discussion_idx] for score, discussion_idx in zip(scores, range(len(scores)))]
    best_category_index = scores.index(max(scores))
    category_assignments.append(category_queries[best_category_index])

df['assigned_category'] = category_assignments

# Count the number of discussions in each category
category_counts = df['assigned_category'].value_counts(normalize=True) * 100

# Save the categorized discussions to a new CSV file
df.to_csv('scratch/n11357738/categorized_discussions_bm25.csv', index=False)

# Visualize the results
plt.figure(figsize=(10, 6))
category_counts.plot(kind='bar', color='skyblue')
plt.title('Percentage of Discussions in Each Category')
plt.xlabel('Category')
plt.ylabel('Percentage')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('scratch/n11357738/category_distribution_bm25.png')
plt.show()
