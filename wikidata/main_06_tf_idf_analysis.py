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
from wordcloud import WordCloud

random_seed = 42

# Load your DataFrame
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
# df['relevant_content'] = df['relevant_content'].apply(stem_text)

# Create TfidfVectorizer
# max_df=0.5 means "ignore all terms that appear in more then 50% of the documents"
# min_df=2 means "ignore all terms that appear in less then 2 documents"
# vect = TfidfVectorizer(analyzer='word', stop_words=stopwords, max_df=0.3, min_df=2)

# Create TfidfVectorizer with n-gram range
# vect = TfidfVectorizer(analyzer='word', stop_words=stopwords, ngram_range=(1, 2), max_df=0.3, min_df=2)
vect = TfidfVectorizer(analyzer='word', stop_words=stopwords, ngram_range=(1,2), max_df=0.1, min_df=2)


# Fit and transform the relevant content column
X = vect.fit_transform(df['relevant_content'])
features = vect.get_feature_names_out()

# Print the top 10 terms in document 1
print(top_feats_in_doc(X, features, 1, 100))
print("top_mean_feats")
# Print the top terms across all documents
print(top_mean_feats(X, features, None, 0.1, 100))

# Apply KMeans clustering
n_clusters = 3
clf = KMeans(n_clusters=n_clusters, max_iter=100, init='k-means++', n_init=1, random_state=random_seed)
labels = clf.fit_predict(X)

# Visualize the clusters using PCA
X_dense = X.todense()
X_dense_array = np.asarray(X_dense)
pca = PCA(n_components=2, random_state=random_seed).fit(X_dense_array)
coords = pca.transform(X_dense_array)

# Assign colors to clusters
label_colors = ["#2AB0E9", "#2BAF74", "#D7665E", "#CCCCCC", "#D2CA0D", "#522A64", "#A3DB05", "#FC6514"]
colors = [label_colors[i] for i in labels]

# Plot the clusters
plt.scatter(coords[:, 0], coords[:, 1], c=colors)
centroids = clf.cluster_centers_
centroid_coords = pca.transform(np.asarray(centroids))
plt.scatter(centroid_coords[:, 0], centroid_coords[:, 1], marker='X', s=200, linewidths=2, c='#444d60')
plt.show()

# Plot the top TF-IDF features per cluster
dfs = top_feats_per_cluster(X, labels, features, 0.1, 50)
plot_tfidf_classfeats_h(dfs)

# Save the TF-IDF features to a file
filename='tfidf_content_features.txt'
save_tfidf_features_to_file(dfs, filename)

#draw word cloud
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