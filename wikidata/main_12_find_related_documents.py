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
from wordcloud import WordCloud
import csv

nltk.download('punkt')

# Load your DataFrame
df = pd.read_csv('scratch/n11357738/cleaned_subtopic_and_content.csv')

# Drop rows with empty content
df.drop(df.query("relevant_content == ''").index, inplace=True)
df.drop(df.query("relevant_content == '--'").index, inplace=True)

# Define custom stop words
custom_stopwords = set(['ect', 'hou', 'com', 'recipient', 'archive', 'wikipedia', 'redirects', 'redirect', 'wikipediaredirects'])
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

query_titles = ["Requests, suggestions for editing coordination"
                
                ]

# Function to find documents with related terms using cosine similarity
def find_related_documents(terms, df, vectorizer, X, threshold=0.2):
    related_docs = set()
    for term in terms:
        term_tfidf = vectorizer.transform([term])
        similarities = cosine_similarity(term_tfidf, X).flatten()
        related_docs.update(np.where(similarities > threshold)[0])
    return related_docs

# Find documents for each type of discussion
query_docs = find_related_documents(query_titles, df, vect, X)
print(query_docs)
