import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load your DataFrame
df = pd.read_csv('scratch/n11357738/cleaned_subtopic_and_content.csv')

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(df['relevant_content'])

# Apply LDA
lda = LatentDirichletAllocation(n_components=10, random_state=0)
topics = lda.fit_transform(tfidf)

# Assign dominant topics
dominant_topics = topics.argmax(axis=1)
df['Dominant_Topic'] = dominant_topics

# Load spaCy for semantic similarity
nlp = spacy.load('en_core_web_md')

# Define category descriptions based on the research paper
category_descriptions = {
    "Editing Coordination": "Requests and suggestions for editing and article coordination.",
    "Information Requests": "Queries for additional information or data about topics.",
    "Vandalism": "Discussions about identifying and managing vandalism.",
    "Policy Discussion": "References to Wikipedia guidelines and policies.",
    "Internal Resources": "Links and references to internal Wikipedia resources.",
    "Off-topic Remarks": "Non-relevant discussions and personal comments.",
    "Polls": "Voting and opinion polls about article content.",
    "Peer Review": "Requests for article reviews and quality checks.",
    "Information Boxes": "Notices and metadata related discussions.",
    "Images": "Discussions about the use and modifications of images."
    # "Other": "General discussions that do not fit into other categories."
}

# Pre-compute category vectors
category_vectors = {cat: nlp(desc) for cat, desc in category_descriptions.items()}

# Function to find closest category based on semantic similarity
def find_closest_category(topic_index):
    # Retrieve the top words for the topic
    topic_text = " ".join(tfidf_vectorizer.get_feature_names_out()[i] for i in lda.components_[topic_index].argsort()[:-10 - 1:-1])
    topic_vector = nlp(topic_text)
    # Calculate similarities and return the category with the highest similarity
    similarities = {category: topic_vector.similarity(vector) for category, vector in category_vectors.items()}
    return max(similarities, key=similarities.get)

# Map each topic to the closest category
df['topic_assigned_category'] = df['Dominant_Topic'].apply(find_closest_category)


# Function to find closest category based on semantic similarity
def find_closest_category_by_content(content):
    content_vector = nlp(content)
    similarities = {category: content_vector.similarity(vector) for category, vector in category_vectors.items()}
    return max(similarities, key=similarities.get)

# Map each discussion content to the closest category
df['Content_Assigned_Category'] = df['relevant_content'].apply(find_closest_category_by_content)


# Save the DataFrame with topics and categories
df.to_csv('scratch/n11357738/enhanced_discussions_with_categories.csv', index=False)
