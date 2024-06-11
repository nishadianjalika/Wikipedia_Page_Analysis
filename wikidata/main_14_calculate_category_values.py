import pandas as pd
import matplotlib.pyplot as plt

def calculate_percentage(df, column):
    """
    Calculate the percentage of each category in the given DataFrame column.
    
    Args:
    df (DataFrame): The DataFrame containing the discussions.
    column (str): The column name where the categories are stored.
    
    Returns:
    DataFrame: A DataFrame containing the categories and their corresponding percentages.
    """
    category_counts = df[column].value_counts()
    category_percentages = (category_counts / category_counts.sum()) * 100
    return category_percentages

def plot_category_distribution(category_percentages, title):
    """
    Plot a bar graph of the category percentages.
    
    Args:
    category_percentages (Series): Pandas Series containing the category percentages.
    title (str): The title of the plot.
    """
    plt.figure(figsize=(10, 8))
    category_percentages.plot(kind='bar', color='skyblue')
    plt.title(title)
    plt.ylabel('Percentage')
    plt.xticks(rotation=45)
    plt.show()

def compare_content_structural_changes(df, content_categories, structural_categories, column):
    """
    Compare the percentages of content-related and structural-related changes.
    
    Args:
    df (DataFrame): The DataFrame containing the discussions.
    content_categories (list): List of content-related categories.
    structural_categories (list): List of structural-related categories.
    column (str): The column name where the categories are stored.
    
    Returns:
    None: This function plots the comparison directly.
    """
    content_df = df[df[column].isin(content_categories)]
    structural_df = df[df[column].isin(structural_categories)]
    
    content_percentage = calculate_percentage(content_df, column)
    structural_percentage = calculate_percentage(structural_df, column)
    
    combined = pd.concat([content_percentage, structural_percentage], axis=1)
    combined.columns = ['Content', 'Structural']
    combined.plot(kind='bar', color=['blue', 'green'])
    plt.title('Comparison of Content vs. Structural Changes')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45)
    plt.show()

# Example usage
# Assume df is your DataFrame loaded from the CSV with 'Assigned_Category' column for categories
content_categories = ["Editing Coordination", "Information Requests", "References to Vandalism", "Policy Discussion", "Peer Review", "Images"]
structural_categories = ["Internal Resources", "Off-topic Remarks", "Polls", "Information Boxes"]

df = pd.read_csv('scratch/n11357738/enhanced_discussions_with_categories_cosine.csv')

# # Calculating percentages for a given category column
# category_percentages = calculate_percentage(df, 'Content_Assigned_Category')
# plot_category_distribution(category_percentages, 'Distribution of Categories')

# # Comparing content vs structural changes
# compare_content_structural_changes(df, content_categories, structural_categories, 'Content_Assigned_Category')

# #for topic
# category_percentages = calculate_percentage(df, 'topic_assigned_category')
# plot_category_distribution(category_percentages, 'Distribution of Categories')

# # Comparing content vs structural changes
# compare_content_structural_changes(df, content_categories, structural_categories, 'topic_assigned_category')


# Cosine_Assigned_Category
category_percentages = calculate_percentage(df, 'Cosine_Assigned_Category')
plot_category_distribution(category_percentages, 'Distribution of Categories')

# Comparing content vs structural changes
compare_content_structural_changes(df, content_categories, structural_categories, 'Cosine_Assigned_Category')
