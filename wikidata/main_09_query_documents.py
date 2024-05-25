import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

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

# Download nltk punkt tokenizer model
nltk.download('punkt')

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

# Example query
query = "changes"

# Transform the query using the same TF-IDF vectorizer
query_tfidf = vect.transform([query])

# Compute cosine similarity between the query and all documents
similarity_scores = cosine_similarity(query_tfidf, X).flatten()

# Number of top documents to retrieve
top_n = 10

# Get the indices of the top N most similar documents
top_n_indices = similarity_scores.argsort()[-top_n:][::-1]

# Print the top N most similar documents
for idx in top_n_indices:
    print(f"Document Index: {idx}, Similarity Score: {similarity_scores[idx]}")
    print(f"Content: {df.iloc[idx]['relevant_content']}")
    print(f"Subtopic: {df.iloc[idx]['subtopic']}")
    print(f"Title: {df.iloc[idx]['title']}\n")
