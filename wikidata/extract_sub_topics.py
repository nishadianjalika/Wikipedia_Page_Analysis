import pandas as pd
import re
from bs4 import BeautifulSoup

# Function to extract subtopics and relevant content
def extract_subtopics(content):
    subtopics = []
    relevant_content = []
    lines = content.split('\n')
    current_subtopic = None
    current_content = []

    for line in lines:
        if line.startswith("==") and line.endswith("=="):
            if current_subtopic is not None:
                # Add the previous subtopic and its content
                subtopics.append(current_subtopic)
                relevant_content.append(' '.join(current_content).strip())
            current_subtopic = line[2:-2].strip()
            current_content = []
        else:
            current_content.append(line.strip())

    if current_subtopic is not None:
        # Add the last subtopic and its content
        subtopics.append(current_subtopic)
        relevant_content.append(' '.join(current_content).strip())

    return subtopics, relevant_content

def extract_sub_topics(df, subtopics_output_file):
    # Group by 'title' and get the latest date's row for each title
    latest_rows = df.sort_values(by=['edited_date']).groupby('title').last().reset_index()

    # Initialize a list to store the new rows
    new_rows = []

    # Iterate over each row in the DataFrame
    for index, row in latest_rows.iterrows():
        # Extract subtopics and relevant content
        subtopics, relevant_content = extract_subtopics(row['content'])

        # Create a new row for each subtopic
        for subtopic, content in zip(subtopics, relevant_content):
            new_row = {
                'title': row['title'],
                'subtopic': subtopic,
                'relevant_content': content
            }
            new_rows.append(new_row)

    # Create a new DataFrame from the list of new rows
    new_df = pd.DataFrame(new_rows)
    new_df = new_df[new_df['relevant_content'].notna() & (new_df['relevant_content'].str.strip() != '')]
    new_df = new_df.dropna()

    # Save to CSV
    new_df.to_csv(subtopics_output_file, index=False)

# def split_discussions_with_dates(content):
#     discussions = []
#     dates = []
#     # Convert content to lowercase to handle case insensitivity
#     content = content.lower()
#     date_pattern = re.compile(r'(\d{2}:\d{2}, \d{1,2} \w+ \d{4} \(utc\))')
    
#     last_end = 0
#     for match in date_pattern.finditer(content):
#         start = match.start()
#         end = match.end()

#         if last_end != 0:
#             discussions.append(content[last_end:start].strip())
#             dates.append(match.group(1).strip())  # group(1) captures the date part
#         last_end = end

#     if last_end < len(content):
#         discussions.append(content[last_end:].strip())
#         dates.append(dates[-1] if dates else '')

#     return discussions, dates

def split_discussions_with_dates(content):
    discussions = []
    dates = []
    # Convert content to lowercase to handle case insensitivity
    content = content.lower()
    date_pattern = re.compile(r'(\d{2}:\d{2}, \d{1,2} \w+ \d{4} \(utc\))')
    
    last_end = 0
    for match in date_pattern.finditer(content):
        start = match.start()
        end = match.end()

        if last_end == 0:
            discussions.append(content[last_end:start].strip())
            dates.append(match.group(1).strip())  # group(1) captures the date part

        if last_end != 0:
            discussions.append(content[(last_end+1):start].strip())
            dates.append(match.group(1).strip())  # group(1) captures the date part

        last_end = end
    # if last_end < len(content):
    #     discussions.append(content[last_end:].strip())
    #     dates.append(dates[-1] if dates else '')

    return discussions, dates

def extract_sub_topics_with_dates(df, processed_subtopics_output_file):
    new_rows = []

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Extract discussions and dates
        # print(row['title'], row['subtopic'])
        # if (row['title'] == 'Talk:Film industry') and (row['subtopic'] == 'Ranked lists based on incomplete statistics'):
        #     discussions, dates = split_discussions_with_dates(row['relevant_content'])

        discussions, dates = split_discussions_with_dates(row['relevant_content'])
        # Create a new row for each discussion
    
        for discussion, date in zip(discussions, dates):
            new_row = {
                'title': row['title'],
                'subtopic': row['subtopic'],
                'relevant_content': discussion,
                'discussion_date': date
            }
            new_rows.append(new_row)
    
    new_df = pd.DataFrame(new_rows)
    
        # Filter out rows with specific undesired content
    undesired_patterns = [
        r'\}\}',             # Matches }}
        r'^\|\}$',           # Matches |}
        r'^.$',              # Matches single character
        r'^:hello$',         # Matches :hello
        r'^:hi$',            # Matches :hi
        r'^::::$',           # Matches ::::
        r'^:$',              # Matches :
        r'^::btw,$',         # Matches ::btw,
        r'^djz$',            # Matches djz
        r'^\)$',             # Matches )
        r'^\:\[\[wp:sofixit\]\]--$',  # Matches :[[wp:sofixit]]--
        r'^for example:$'    # Matches for example:
    ]

    # for pattern in undesired_patterns:
        # new_df = new_df[~new_df['relevant_content'].str.match(pattern, na=False, case=False)]

    # Create a new DataFrame from the list of new rows
    
    new_df['relevant_content'] = new_df['relevant_content'].apply(clean_relevant_content)
    # new_df = new_df[~new_df['relevant_content'].str.contains(r'\}\}', regex=True)]
    new_df = new_df[new_df['relevant_content'].notna() & (new_df['relevant_content'].str.strip() != '')]
    new_df = new_df.dropna()
    new_df.loc[:, 'discussion_date'] = new_df['discussion_date'].str.replace('(utc)', '', regex=False).str.strip()
    new_df['relevant_content'] = new_df['relevant_content'].apply(clean_relevant_content)

    # Save to CSV
    new_df.to_csv(processed_subtopics_output_file, index=False)

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