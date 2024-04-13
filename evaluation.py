import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Script to evaluate data and present data visualization in graph format

# Load data
data = pd.read_csv("./QuestionnaireDiss.csv")

# Extract the questions based on the analysis needs
relevant_columns = [
    "What age range would you say you are in?",
    "What country are you currently situated in?",
    "On a scale of 1-10, 1 being the worst and 10 being the best, how would you rate model 1?",
    "On a scale of 1-10, 1 being the worst and 10 being the best, how would you rate the model 2?",
    "On a scale of 1-10, 1 being the worst and 10 being the best, how would you rate the model 3?",
    "On a scale of 1-10, 1 being the worst and 10 being the best, how would you rate the model 4?"
]

# Extract only the relevant data
analysis_data = data[relevant_columns]

# Rename columns for ease of analysis of data
analysis_data.columns = ['Age_Range', 'Country', 'Model1_Rating', 'Model2_Rating', 'Model3_Rating', 'Model4_Rating']

# Correcting country names for consistency
analysis_data['Country'] = analysis_data['Country'].replace('United Arab Emirates', 'UAE')

# Calculate the average rating for each model
model_averages = analysis_data[['Model1_Rating', 'Model2_Rating', 'Model3_Rating', 'Model4_Rating']].mean()

# Plotting the average ratings for each model
plt.figure(figsize=(10, 6))
model_averages.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Average Ratings of Anime Recommendation Models')
plt.xlabel('Model')
plt.ylabel('Average Rating')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Create a column for the preferred model based on highest rating
preferred_model_column = analysis_data[['Model1_Rating', 'Model2_Rating', 'Model3_Rating', 'Model4_Rating']].idxmax(axis=1)
analysis_data['Preferred_Model'] = preferred_model_column.str.extract('(\d)').astype(int)

# Grouped bar graph for distribution of preferred models by country
country_model_counts = analysis_data.groupby('Country')['Preferred_Model'].value_counts().unstack().fillna(0)
country_model_counts.plot(kind='bar', figsize=(14, 7), width=0.8)
plt.title('Preferred Model Distribution by Country')
plt.xlabel('Country')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Preferred Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Grouped bar graph for distribution of preferred models by age range
age_range_model_counts = analysis_data.groupby('Age_Range')['Preferred_Model'].value_counts().unstack().fillna(0)
age_range_model_counts.plot(kind='bar', figsize=(14, 7), width=0.8)
plt.title('Preferred Model Distribution by Age Range')
plt.xlabel('Age Range')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Preferred Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
