import pandas as pd
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Display topics
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

df = pd.read_csv("scratch/n11357738/cleaned_subtopic_and_content.csv")

# Topic modeling using Latent Dirichlet Allocation (LDA)
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, stop_words='english')
# tfidf = tfidf_vectorizer.fit_transform(df['relevant_content'])
tfidf = tfidf_vectorizer.fit_transform(df['subtopic'])
lda_model = LatentDirichletAllocation(n_components=20, random_state=42)
lda_output = lda_model.fit_transform(tfidf)

no_top_words = 20
feature_names = tfidf_vectorizer.get_feature_names_out()
display_topics(lda_model, feature_names, no_top_words)


