import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_suptopic_data(df, cleaned_file_path):
    text_to_remove = 'listed at [[Wikipedia:Redirects for discussion|Redirects for discussion]]'

    # df_cleaned = df.dropna()
    # Remove rows where 'relevant_content' column is empty or has NaN values
    df_cleaned = df[df['relevant_content'].notna() & (df['relevant_content'].str.strip() != '')]

    # df_cleaned['subtopic'] = df_cleaned['subtopic'].str.replace(text_to_remove, '', regex=False).str.strip()
    df_cleaned.loc[:, 'subtopic'] = df_cleaned['subtopic'].str.replace(text_to_remove, '', regex=False).str.strip()

    # Save the cleaned DataFrame to a new CSV file
    df_cleaned.to_csv(cleaned_file_path, index=False)

    print(f"Cleaned data saved to {cleaned_file_path}")   

def subtopic_frequency_analysis(df):

    # Frequency analysis of subtopics for each title
    subtopic_freq = df.groupby(['title', 'subtopic']).size().reset_index(name='count')
    subtopic_freq = subtopic_freq.sort_values(by=['title', 'count'], ascending=[True, False])

    # Display the top subtopics for each title
    top_subtopics = subtopic_freq.groupby('title').head(10)
    # print(top_subtopics)

    # Save the frequency analysis to a CSV file
    top_subtopics.to_csv('scratch/n11357738/subtopic_frequency_analysis.csv', index=False)

    plot_frequency(top_subtopics)
     
def plot_frequency(top_subtopics):
    # Plot the top subtopics for each title
    for title in top_subtopics['title'].unique():
        data = top_subtopics[top_subtopics['title'] == title]
        plt.figure(figsize=(10, len(data) * 0.5))  # Increase figure size based on the number of subtopics
        sns.barplot(data=data, x='count', y='subtopic')
        plt.title(f'Top Subtopics for {title}')
        plt.xlabel('Count')
        plt.ylabel('Subtopic')
        plt.tight_layout()  # Adjust layout to make room for y-axis labels
        plt.yticks(rotation=0, ha='right') 
        # plt.show()
        file_name = f"images/{title.replace(':', '_')}_subtopic_count.png"  
        plt.savefig(file_name)

# Function to get sentiment polarity
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

def sentiment_analysis(df):
    # Calculate sentiment for each subtopic discussion
    df['sentiment'] = df['relevant_content'].apply(get_sentiment)

    # Average sentiment per subtopic for each title
    sentiment_analysis = df.groupby(['title', 'subtopic'])['sentiment'].mean().reset_index()
    sentiment_analysis = sentiment_analysis.sort_values(by=['title', 'sentiment'], ascending=[True, False])

    # Display the sentiment analysis
    print(sentiment_analysis)

    # Save the sentiment analysis to a CSV file
    sentiment_analysis.to_csv('scratch/n11357738/subtopic_sentiment_analysis.csv', index=False)
    plot_sentiment(sentiment_analysis)

def plot_sentiment(sentiment_analysis):
    # Plot the sentiment analysis for each title with adjustments for long y-axis labels and label rotation
    for title in sentiment_analysis['title'].unique():
        data = sentiment_analysis[sentiment_analysis['title'] == title]
        plt.figure(figsize=(10, len(data) * 0.5))  # Increase figure size based on the number of subtopics
        sns.barplot(data=data, x='sentiment', y='subtopic')
        plt.title(f'Sentiment Analysis for {title}')
        plt.xlabel('Sentiment')
        plt.ylabel('Subtopic')
        plt.tight_layout()  # Adjust layout to make room for y-axis labels
        plt.yticks(rotation=0, ha='right')  # Rotate y-axis labels if needed and adjust alignment
        # plt.show()
        file_name = f"images/{title.replace(':', '_')}_sentiment.png"  
        plt.savefig(file_name)

# Function to get keywords
def get_keywords(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return tokens

def keyword_analysis(df):
    # Calculate keywords for each subtopic
    df['keywords'] = df['relevant_content'].apply(get_keywords)

    # Get the most common keywords for each title
    keyword_analysis = df.groupby('title')['keywords'].apply(lambda x: Counter([item for sublist in x for item in sublist]).most_common(10)).reset_index()

    # Display the keyword analysis
    print(keyword_analysis)

    # Save the keyword analysis to a CSV file
    keyword_analysis.to_csv('scratch/n11357738/subtopic_keyword_analysis.csv', index=False)

def main():

    subtopics_output_file = 'scratch/n11357738/subtopic_data.csv'
    cleaned_file_path = 'scratch/n11357738/cleaned_subtopic_data.csv'
    df = pd.read_csv(subtopics_output_file)
    # clean_suptopic_data(df, cleaned_file_path)

    cleaned_df = pd.read_csv(cleaned_file_path)
    # subtopic_frequency_analysis(cleaned_df)

    # sentiment_analysis(cleaned_df)

    keyword_analysis(cleaned_df)

if __name__ == "__main__":
    main()