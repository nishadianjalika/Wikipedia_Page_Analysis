from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd

# Display the top words in each topic
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))


def topic_modelling(df):
    # Combine all discussions into a single document for each talk page
    df['combined_content'] = df.groupby('title')['relevant_content'].transform(lambda x: ' '.join(x))
    df_unique = df[['title', 'combined_content']].drop_duplicates()

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(df_unique['combined_content'])

    # LDA Model
    lda = LatentDirichletAllocation(n_components=10, random_state=42)
    lda.fit(tfidf_matrix)

    no_top_words = 10
    display_topics(lda, vectorizer.get_feature_names_out(), no_top_words)

def topic_modelling_individual_discussions(df):
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(df['relevant_content'])

    # LDA Model
    lda = LatentDirichletAllocation(n_components=20, random_state=42)
    lda.fit(tfidf_matrix)

    no_top_words = 20
    display_topics(lda, vectorizer.get_feature_names_out(), no_top_words)

if __name__ == "__main__":
    df = pd.read_csv('scratch/n11357738/cleaned_subtopic_and_content.csv')
    topic_modelling(df)
    # print("===============================================")
    # topic_modelling_individual_discussions(df)