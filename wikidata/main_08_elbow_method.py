# Elbow method to find best cluster number

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from helpers import *  
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

df = pd.read_csv('scratch/n11357738/cleaned_subtopic_and_content.csv') 

# Define custom stop words
custom_stopwords = set(['ect', 'hou', 'com', 'recipient', 'archive', 'wikipedia', 'redirects', 'redirect', 'wikipediaredirects'])
stopwords = ENGLISH_STOP_WORDS.union(custom_stopwords)

# Convert the stopwords set to a list
stopwords = list(stopwords)

# # Download nltk punkt tokenizer model
# nltk.download('punkt')

# Define a function to apply stemming
def stem_text(text):
    stemmer = PorterStemmer()
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

# Apply stemming to the 'relevant_content' column
df['relevant_content'] = df['relevant_content'].apply(stem_text)

# Create TfidfVectorizer
# max_df=0.5 means "ignore all terms that appear in more then 50% of the documents"
# min_df=2 means "ignore all terms that appear in less then 2 documents"
# vect = TfidfVectorizer(analyzer='word', stop_words=stopwords, max_df=0.3, min_df=2)

# Create TfidfVectorizer with n-gram range
# vect = TfidfVectorizer(analyzer='word', stop_words=stopwords, ngram_range=(1, 2), max_df=0.3, min_df=2)
vect = TfidfVectorizer(analyzer='word', stop_words=stopwords, ngram_range=(2,2), max_df=0.2, min_df=2)


# Fit and transform the relevant content column
X = vect.fit_transform(df['relevant_content'])

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, max_iter=100, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()