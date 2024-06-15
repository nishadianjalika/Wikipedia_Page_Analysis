from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
df = pd.read_csv('scratch/n11357738/enhanced_discussions_with_all_similarities.csv')

# Unique Labels from categories
labels = df['Manual_Category'].unique()

# Cosine Similarity Metrics and F1-Scores
cosine_metrics = classification_report(df['Manual_Category'], df['Cosine_Assigned_Category'], target_names=labels, output_dict=True)
print("Cosine Similarity Metrics:")
print(classification_report(df['Manual_Category'], df['Cosine_Assigned_Category'], target_names=labels))

# Overall F1-score for Cosine Similarity
overall_f1_cosine = cosine_metrics['weighted avg']['f1-score']

# Plot Confusion Matrix for Cosine Similarity
cm_cosine = confusion_matrix(df['Manual_Category'], df['Cosine_Assigned_Category'], labels=labels)
plt.figure(figsize=(14, 14))
sns.heatmap(cm_cosine, fmt='d', xticklabels=labels, yticklabels=labels)
plt.title(f'Confusion Matrix for Cosine Similarity\nOverall F1-score: {overall_f1_cosine:.2f}')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=25, ha='right',fontsize = 8)
plt.yticks(rotation=0, fontsize = 8)
plt.show()

# WordNet Similarity Metrics and F1-Scores
wordnet_metrics = classification_report(df['Manual_Category'], df['WordNet_Assigned_Category'], target_names=labels, output_dict=True)
print("WordNet Similarity Metrics:")
print(classification_report(df['Manual_Category'], df['WordNet_Assigned_Category'], target_names=labels))

# Overall F1-score for WordNet Similarity
overall_f1_wordnet = wordnet_metrics['weighted avg']['f1-score']

# Plot Confusion Matrix for WordNet Similarity
cm_wordnet = confusion_matrix(df['Manual_Category'], df['WordNet_Assigned_Category'], labels=labels)
plt.figure(figsize=(14, 14))
sns.heatmap(cm_wordnet, xticklabels=labels, yticklabels=labels)
plt.title(f'Confusion Matrix for WordNet Similarity\nOverall F1-score: {overall_f1_wordnet:.2f}')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=25, ha='right',fontsize = 8)
plt.yticks(rotation=0,fontsize = 8)
plt.show()
