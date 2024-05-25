import pandas as pd
import re
from bs4 import BeautifulSoup

def clean_text(text):
    # Remove user patterns like [[user:jeff watts|jw]]
    text = re.sub(r'\[\[user:[^\]]+\|[^\]]+\]\]', '', text)
    # Remove signatures patterns like [[wikipedia:signatures|unsigned]] comment added by
    text = re.sub(r'\[\[wikipedia:signatures\|unsigned\]\] comment added by [^\]]+\]', '', text)
    # Remove other patterns like [[special:contributions/...|...]] ([[user talk:...|...]])
    text = re.sub(r'\[\[special:contributions/[^\]]+\|[^\]]+\]\] \(\[\[user talk:[^\]]+\|[^\]]+\]\]\)', '', text)
    # Remove patterns like ([[user talk:internetarchivebot|report bug]])
    text = re.sub(r'\(\[\[user talk:[^\]]+\|[^\]]+\]\]\)', '', text)
    # Remove patterns like [[user:internetarchivebot|'''internetarchivebot''']]
    text = re.sub(r'\[\[user:[^\]]+\|[^\]]+\]\]', '', text)
    # Remove remaining square bracketed items
    text = re.sub(r'\[\[[^\]]+\]\]', '', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    return text

def clean_relevant_content(content):
    patterns = [
        r'\[\[user:.*?\]\] \(\[\[user talk:.*?\|talk\]\]\)', # Matches [[User:RichardFry|RichardFry]] ([[User talk:RichardFry|talk]])
        r'\[\[user talk:.*?\]\]',                    # Matches [[user talk:kvng|kvng]]
        r'\(\[\[user talk:.*?\|talk\]\]\)',          # Matches (([[user talk:paul w|talk]]))
        r'\[\[user:.*?\]\]',                        # Matches [[user:paul w]]
        r'\(\[\[user_talk:.*?\|talk\]\]\)',          # Matches (([[user_talk:finnusertop|talk]] &#124;  &#124; ))
        # r'\(\[\[.*?\)',                          # Matches (([[user_talk:finnusertop|talk]] &#124;  &#124; ))
        r'\[\[user_talk:.*?\]\]'                     # Matches [[user_talk:northamerica1000|1000]]
    ]
    # Remove HTML tags
    content = clean_html_tags(content)

    # for pattern in patterns:
    #     content = re.sub(pattern, '', content, flags=re.IGNORECASE)
    
    return content

def clean_html_tags(content):
    # Remove HTML tags
    soup = BeautifulSoup(content, 'html.parser')
    text = soup.get_text()
    return text

# Load your DataFrame
df = pd.read_csv('scratch/n11357738/processed_subtopic_data.csv') 

# Drop rows with empty content
df.drop(df.query("relevant_content == ''").index, inplace=True)
df['relevant_content'] = df['relevant_content'].apply(clean_text)
df['relevant_content'] = df['relevant_content'].apply(clean_relevant_content)

df['relevant_content'] = df['relevant_content'].astype(str).str.strip()

# Drop rows with empty content and those with specific unwanted values
unwanted_values = ["", "--", "it's", ":", "hffffÅ“", ":--", "—", "*"]
df = df[~df['relevant_content'].isin(unwanted_values)]

unwanted_patterns = ['/small>', '*', '::', ':', "==", "="]
for pattern in unwanted_patterns:
    df['relevant_content'] = df['relevant_content'].str.replace(pattern, '', regex=False)
    df['subtopic'] = df['subtopic'].str.replace(pattern, '', regex=False)



df.to_csv("scratch/n11357738/cleaned_subtopic_and_content.csv", index=False)


