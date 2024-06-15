from extract_talk_page_info import extract_data_from_xml
from extract_talk_page_info import save_to_csv
from edit_frequency_analysis import analyse_edits_over_the_years
from edit_frequency_analysis import total_edit_counts
from edit_frequency_analysis import total_edit_counts_per_industry
from edit_frequency_analysis import edit_count_per_title
from extract_sub_topics import extract_sub_topics
from extract_sub_topics import extract_sub_topics_with_dates
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import glob

output_csv_file = 'scratch/n11357738/output_file_full.csv'

def convert_xml_to_csv(r_path):
    
    # Initialize an empty list to store data from all XML files
    all_data = []

    for file in glob.glob(r_path + "/*.xml"):
        xml_data = extract_data_from_xml(file)
        # print(xml_data)
        all_data.extend(xml_data)
    
    print('Collecting data to csv completed')
    if all_data:
        # Save the combined data to a single CSV file
        save_to_csv(all_data, output_csv_file)
        print("Data saved to CSV file:")
    else:
        print("No data extracted from XML files.")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if pd.isna(text):  # Check if text is NaN
        return ''       # Return an empty string for missing values
    tokens = word_tokenize(str(text).lower())  # Convert to string before lowercasing
    tokens = [token for token in tokens if token.isalnum()]
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

def load_and_process_csv(per_sector_csv_file):

    # Load the CSV file into a pandas DataFrame
    sector_df = pd.read_csv(per_sector_csv_file)
    sector_df['edited_date'] = pd.to_datetime(sector_df['edited_date'])
    # print(sector_df['edited_date'])
    # print("=================================")
    # sector_df['cleaned_content'] = sector_df['content'].apply(preprocess_text)

    return sector_df

def main():

    r_path = 'scratch/n11357738/'
    # convert_xml_to_csv(r_path) #First, read and convert xml files to csv format to save page title, user, edited_date and edited_content
    
    sector_df = load_and_process_csv(output_csv_file) #Load saved csv file and pre process

    total_edit_counts(sector_df)
    total_edit_counts_per_industry(sector_df)
    edit_count_per_title(sector_df) 

    subtopics_output_file = 'scratch/n11357738/subtopic_data.csv'
    extract_sub_topics(sector_df, subtopics_output_file) #this is done to get the title subtopic and its latest full content

    #this is to further devide the full content for each subtopic based on dates
    df = pd.read_csv(subtopics_output_file)
    extract_sub_topics_with_dates(df, 'scratch/n11357738/processed_subtopic_data.csv') 
    
    
    # todo:
    # apply topic modellling for these extracted contents 
    # check frequent topics
    # check highly used words



if __name__ == "__main__":
    main()