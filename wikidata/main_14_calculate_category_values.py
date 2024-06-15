import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV file
df = pd.read_csv('scratch/n11357738/enhanced_discussions_with_all_similarities.csv')

# Function to calculate and plot the distribution of categories
def plot_category_distribution(column, title):
    # Count the frequency of each category
    category_counts = df[column].value_counts()
    
    # Convert counts to percentages
    category_percentages = (category_counts / category_counts.sum()) * 100

    # Plotting
    plt.figure(figsize=(10, 6))
    ax = category_percentages.plot(kind='bar', color='skyblue')
    plt.title(title)
    plt.ylabel('Percentage')
    plt.xlabel('Categories')
    plt.xticks(rotation=25, ha='right', fontsize = 8)
    # Add percentage labels above each bar
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')


    plt.show()

    return category_counts, category_percentages

# Plot for Manual Categories
manual_counts, manual_percentages = plot_category_distribution('Manual_Category', 'Distribution of Manual Categories')

# Plot for WordNet Similarity Categories
wordnet_counts, wordnet_percentages = plot_category_distribution('WordNet_Assigned_Category', 'Distribution of WordNet Similarity Categories')

# If you need to compare them in a single plot or do any additional analysis, you can easily extend from here.
