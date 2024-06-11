import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load your DataFrame
df = pd.read_csv('scratch/n11357738/cleaned_subtopic_and_content.csv')

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')

# Define category descriptions
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
}

# Vectorize the relevant_content of discussions
discussion_vectors = tfidf_vectorizer.fit_transform(df['relevant_content'])

# Vectorize category descriptions
category_texts = list(category_descriptions.values())
category_vectors = tfidf_vectorizer.transform(category_texts)

# Function to find closest category based on cosine similarity
def find_closest_category_by_cosine(content_vector):
    # Calculate cosine similarity between the content vector and all category vectors
    similarities = cosine_similarity(content_vector, category_vectors)
    # Find the index of the maximum similarity
    closest_category_index = similarities.argmax()
    # Map the index to category name
    return list(category_descriptions.keys())[closest_category_index]

# Map each discussion content to the closest category based on cosine similarity
df['Cosine_Assigned_Category'] = [find_closest_category_by_cosine(v) for v in discussion_vectors]

# Save the DataFrame with topics and categories
df.to_csv('scratch/n11357738/enhanced_discussions_with_categories_cosine.csv', index=False)
