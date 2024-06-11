import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def analyse_edits_over_the_years(df, title, save_path):
    #plot published news yearly
    ax = df.groupby(df.edited_date.dt.year)['cleaned_content'].count().plot(kind='bar', figsize=(12, 8))
    ax.set(xlabel='Year', ylabel='Count of Edits', title=f"Trend of Talk Pages edits Over the last decade for " + title )
    # plt.show()

    # Save the plot as an image if save_path is provided
    if save_path:
        plt.savefig(save_path)
        plt.close() 

def edit_count_per_title(df):
    edit_counts = df.groupby('title').size()

    # Iterate over each title and plot edit count variation in separate plots
    for title, count in edit_counts.items():
        title_df = df[df['title'] == title]
        # title_df['edited_date'] = pd.to_datetime(title_df['edited_date'])  # Convert edited_date to datetime if it's not already
        
        # Plot edit count variation
        plt.figure(figsize=(10, 6))
        title_df.groupby(title_df['edited_date'].dt.year)['edited_date'].count().plot(kind='bar')
        plt.title(f"Edit Count Variation for '{title}'")
        plt.xlabel('Year')
        plt.ylabel('Edit Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        # plt.show()

        file_name = f"images/{title.replace(':', '_')}_edit_count.png"  
        plt.savefig(file_name)
        plt.close()

def total_edit_counts(df):
    # Extract the year portion from the 'date' column
    df['year'] = df['edited_date'].dt.year

    # Group by Year and Title, then count the number of edits for each title in each year
    edit_counts = df.groupby(['year', 'title']).size().reset_index(name='edit_count')

    # Aggregate edit counts to get the total edit count for each year
    total_edit_counts = edit_counts.groupby('year')['edit_count'].sum().reset_index()

    # Plot the total edit counts over time
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=total_edit_counts, x='year', y='edit_count')
    plt.title('Total Discussion Counts Over Years')
    plt.xlabel('Year')
    plt.ylabel('Total Discussion Count')
    plt.xticks(rotation=45)
    # plt.show()
    plt.savefig('images/Total_Discussion_Counts_Over_Years.png')
    plt.close() 

def total_edit_counts_per_industry(df):
    # Extract the year portion from the 'date' column
    df['year'] = df['edited_date'].dt.year

    # Group by Year and Title, then count the number of edits for each title in each year
    edit_counts = df.groupby(['year', 'title']).size().reset_index(name='edit_count')

    # Plot the edit counts for each title
    plt.figure(figsize=(20, 12))
    sns.lineplot(data=edit_counts, x='year', y='edit_count', hue='title')
    plt.title('Discussions for Each Industry Over Years')
    plt.xlabel('Year')
    plt.ylabel('Edit Count')
    plt.xticks(rotation=45)
    plt.legend(title='Title', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    # plt.show()
    plt.savefig('images/Total_Discussion_Counts_Over_Years_Per_Industry.png')
    plt.close() 