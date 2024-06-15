import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import wordnet as wn
from nltk import word_tokenize
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')

# Load your DataFrame
df = pd.read_csv('scratch/n11357738/cleaned_subtopic_and_content.csv')

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
discussion_vectors = tfidf_vectorizer.fit_transform(df['relevant_content'])

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

# Vectorize category descriptions
category_texts = list(category_descriptions.values())
category_vectors = tfidf_vectorizer.transform(category_texts)

def wordnet_similarity(doc1, doc2):
    words1 = word_tokenize(doc1)
    words2 = word_tokenize(doc2)
    synsets1 = [wn.synsets(word)[0] for word in words1 if wn.synsets(word)]
    synsets2 = [wn.synsets(word)[0] for word in words2 if wn.synsets(word)]
    max_score = 0
    for syn1 in synsets1:
        for syn2 in synsets2:
            score = wn.path_similarity(syn1, syn2)
            if score and score > max_score:
                max_score = score
    return max_score

# Applying all similarity measures
results = {
    'Cosine_Similarity': [],
    'WordNet_Similarity': []
}

for i, text_vector in enumerate(discussion_vectors):
    text = ' '.join(tfidf_vectorizer.inverse_transform(text_vector)[0])
    
    # Cosine Similarity
    cosine_sim = cosine_similarity(text_vector, category_vectors)
    closest_cat_cosine = list(category_descriptions.keys())[cosine_sim.argmax()]

    # WordNet Similarity
    closest_cat_wordnet = max(category_descriptions.keys(), key=lambda cat: wordnet_similarity(text, category_descriptions[cat]))

    results['Cosine_Similarity'].append(closest_cat_cosine)
    results['WordNet_Similarity'].append(closest_cat_wordnet)

# Adding results to the DataFrame
df['Cosine_Assigned_Category'] = results['Cosine_Similarity']
df['WordNet_Assigned_Category'] = results['WordNet_Similarity']

# Save the DataFrame with topics and categories
df.to_csv('scratch/n11357738/enhanced_discussions_with_all_similarities.csv', index=False)
