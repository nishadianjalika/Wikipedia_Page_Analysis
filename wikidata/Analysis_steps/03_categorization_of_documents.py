import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

# Prepare text for vectorization
all_texts = df['relevant_content'].tolist() + category_queries
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)

# Split the TF-IDF matrix into discussions and categories
discussion_tfidf = tfidf_matrix[:len(df)]
category_tfidf = tfidf_matrix[len(df):]

# Calculate cosine similarity
cosine_similarities = cosine_similarity(discussion_tfidf, category_tfidf)

# Assign each discussion to the most similar category
category_indices = cosine_similarities.argmax(axis=1)
category_assignments = [category_queries[idx] for idx in category_indices]
df['assigned_category'] = category_assignments

# Count the number of discussions in each category
category_counts = df['assigned_category'].value_counts(normalize=True) * 100

# Save the categorized discussions to a new CSV file
df.to_csv('scratch/n11357738/categorized_discussions.csv', index=False)

# Visualize the results
plt.figure(figsize=(10, 6))
category_counts.plot(kind='bar', color='skyblue')
plt.title('Percentage of Discussions in Each Category')
plt.xlabel('Category')
plt.ylabel('Percentage')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('scratch/n11357738/category_distribution.png')
plt.show()
