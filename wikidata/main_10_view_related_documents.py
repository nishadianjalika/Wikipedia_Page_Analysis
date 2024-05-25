import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import csv

nltk.download('punkt')

# Load your DataFrame
df = pd.read_csv('scratch/n11357738/cleaned_subtopic_and_content.csv')

# Drop rows with empty content
df.drop(df.query("relevant_content == ''").index, inplace=True)
df.drop(df.query("relevant_content == '--'").index, inplace=True)

# Define custom stop words
custom_stopwords = set(['ect', 'hou', 'com', 'recipient', 'archive'])
stopwords = ENGLISH_STOP_WORDS.union(custom_stopwords)

# Convert the stopwords set to a list
stopwords = list(stopwords)

# Define a function to apply stemming
def stem_text(text):
    stemmer = PorterStemmer()
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

# Apply stemming to the 'relevant_content' column
# df['relevant_content'] = df['relevant_content'].apply(stem_text)

# Remove specific patterns from the text
patterns_to_remove = ['</small>', '::', ':', ' -- ', '--', ':--']
for pattern in patterns_to_remove:
    df['relevant_content'] = df['relevant_content'].str.replace(pattern, '', regex=False)

# Create TfidfVectorizer with n-gram range
vect = TfidfVectorizer(analyzer='word', stop_words=stopwords, ngram_range=(1, 2), max_df=0.2, min_df=2)

# Fit and transform the relevant content column
X = vect.fit_transform(df['relevant_content'])
features = vect.get_feature_names_out()

# Apply KMeans clustering
n_clusters = 3
random_seed = 42
clf = KMeans(n_clusters=n_clusters, max_iter=100, init='k-means++', n_init=10, random_state=random_seed)
labels = clf.fit_predict(X)

# Add the cluster labels to the DataFrame
df['cluster'] = labels

# Define queries for structural and content changes
structural_queries = [
    "section", "merge", "template", "infobox", "layout", "reference", "formatting", "citation",
    "sourcecheck", "instructions template"
]

content_queries = [
    "update", "addition", "remove", "edit", "correction", "expand", "accuracy", "content",
     "discussion","links", "details"
]

# Function to find related discussions based on queries
def find_related_discussions(queries, df, vectorizer, X, min_score=0):
    results = []
    for query in queries:
        query_tfidf = vectorizer.transform([query])
        similarity_scores = cosine_similarity(query_tfidf, X).flatten()
        
        for idx, score in enumerate(similarity_scores):
            if score > min_score:
                results.append({
                    "query": query,
                    "document_index": idx,
                    "similarity_score": score,
                    "content": df.iloc[idx]['relevant_content'],
                    "cluster": df.iloc[idx]['cluster']
                })
    return results

# Find related discussions for structural and content queries with score > 0
structural_results = find_related_discussions(structural_queries, df, vect, X, min_score=0)
content_results = find_related_discussions(content_queries, df, vect, X, min_score=0)

# Combine results
all_results = structural_results + content_results

# Save results to a file
with open('related_discussions.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['query', 'document_index', 'similarity_score', 'content', 'cluster']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in all_results:
        writer.writerow(row)

# Count the number of discussions per query
query_counts = pd.DataFrame(all_results).groupby('query').size().reset_index(name='counts')

# Separate structural and content queries
query_counts['type'] = query_counts['query'].apply(lambda x: 'Structural' if x in structural_queries else 'Content')

# Plot the results
fig, ax = plt.subplots(figsize=(16, 15))
for query_type, data in query_counts.groupby('type'):
    ax.bar(data['query'], data['counts'], label=query_type)

ax.set_xlabel('Queries')
ax.set_ylabel('Number of Discussions')
ax.set_title('Number of Discussions Related to Structural and Content Changes')
ax.legend()
plt.xticks(rotation=20)
plt.show()
