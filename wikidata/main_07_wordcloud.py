import matplotlib.pyplot as plt
from wordcloud import WordCloud

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

nltk.download('punkt')

# Load your DataFrame
df = pd.read_csv('scratch/n11357738/cleaned_subtopic_and_content.csv')

# Drop rows with empty content
df.drop(df.query("relevant_content == ''").index, inplace=True)
df.drop(df.query("relevant_content == '--'").index, inplace=True)

# Define custom stop words
custom_stopwords = set(['ect', 'hou', 'com', 'recipient', 'archive', 'archiv', ''])
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

# Create word clouds for each cluster
def generate_wordcloud(data, title=None):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(data)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    if title:
        plt.title(title, size=20)
    plt.show()

# Concatenate text for each cluster
cluster_texts = df.groupby('cluster')['relevant_content'].apply(lambda x: ' '.join(x))

# Generate word clouds for each cluster
for cluster, text in cluster_texts.items():
    generate_wordcloud(text, title=f'Cluster {cluster} Word Cloud')
