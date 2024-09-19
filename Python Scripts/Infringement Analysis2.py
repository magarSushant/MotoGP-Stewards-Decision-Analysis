#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd

# Define the file path
file_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Final Versions\Infraction classification-Final.xlsx'

# Load the specific sheet 'Infringement Master', skipping the first 8 rows
df = pd.read_excel(file_path, sheet_name='Infringement Master', skiprows=8)

# Correct column name with leading space
correct_column_name = ' Sanction Classification'  # Notice the leading space

# Define the severe sanctions keywords
severe_keywords = ["Grid Penalties", "Performance Penalties", "Fine", "Disqualified", "Double"]

# Categorize sanctions as 'Severe' or 'Mild'
df['Sanction Severity'] = df[correct_column_name].apply(
    lambda x: 'Severe' if any(kw in str(x) for kw in severe_keywords) else 'Mild'
)

# Identify rows with multiple sanctions (consider them as severe)
df['Multiple Sanctions'] = df[correct_column_name].apply(lambda x: True if ',' in str(x) else False)

# If multiple sanctions, overwrite the severity as 'Severe'
df.loc[df['Multiple Sanctions'] == True, 'Sanction Severity'] = 'Severe'

# Display the categorized data
print(df[[correct_column_name, 'Sanction Severity']])

# Save the categorized data to a new Excel file if needed
output_file_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Final Versions\categorized_sanctions.xlsx'
df.to_excel(output_file_path, index=False)

print(f"Categorized sanctions have been saved to {output_file_path}")


# In[21]:


# Display all cases where the sanction is classified as Severe
severe_sanctions = df[df['Sanction Severity'] == 'Severe']
print(severe_sanctions[['Infraction ID', 'Infringement Type', 'Sanction Severity', 'Aggravating factor']])



# In[23]:


# Analyze sentiments for the identified severe sanctions
sentiment_analysis = severe_sanctions[
    ['Infraction ID', 'Infringement Type', 'Sanction Severity', 'PUBLIC SENTIMENT', 'RIDER SENTIMENT', 'TEAM SENTIMENT', 'MEDIA SENTIMENT']
]

# Display the sentiment data
print(sentiment_analysis)


# In[25]:


# Identify potentially contentious decisions with multiple negative sentiments
contentious_cases = sentiment_analysis[
    (sentiment_analysis['PUBLIC SENTIMENT'] == 'Negative') |
    (sentiment_analysis['RIDER SENTIMENT'] == 'Negative') |
    (sentiment_analysis['TEAM SENTIMENT'] == 'Negative') |
    (sentiment_analysis['MEDIA SENTIMENT'] == 'Negative')
]

print(contentious_cases)


# In[27]:


# Analyze the specifics of the most contentious cases
contentious_case_details = df[df['Infraction ID'].isin([
    '2022-CGP-002', '2022-CGP-004', '2022-CGP-006', '2022-CGP-007',
    '2022-CGP-0011', '2022-C3-0013', '2022-CGP-0014', '2022-CGP-0016',
    '2022-CGP-0017', '2022-C3-0023', '2022-C3-0024'
])][['Infraction ID', 'Infringement Type', 'Aggravating factor', 'SANCTION']]

print(contentious_case_details)


# In[29]:


# Correlate sentiment with severity and aggravating factors
contentious_cases_sentiment = df[df['Infraction ID'].isin([
    '2022-CGP-002', '2022-CGP-004', '2022-CGP-006', '2022-CGP-007',
    '2022-CGP-0011', '2022-C3-0013', '2022-CGP-0014', '2022-CGP-0016',
    '2022-CGP-0017', '2022-C3-0023', '2022-C3-0024'
])][['Infraction ID', 'Infringement Type', 'Aggravating factor', 'SANCTION', 'PUBLIC SENTIMENT', 'RIDER SENTIMENT', 'TEAM SENTIMENT', 'MEDIA SENTIMENT']]

print(contentious_cases_sentiment)


# In[31]:


# Focus on all severe sanctions with any aggravating factors
contentious_cases_all_factors = contentious_cases_sentiment[
    contentious_cases_sentiment['Aggravating factor'].apply(lambda x: x != '-')
]

print(contentious_cases_all_factors)


# In[35]:


# Identify cases with predominantly negative sentiments
highly_contentious = contentious_cases_all_factors[
    contentious_cases_all_factors[['PUBLIC SENTIMENT', 'RIDER SENTIMENT', 'TEAM SENTIMENT', 'MEDIA SENTIMENT']].apply(lambda x: (x == 'Negative').sum(), axis=1) > 1
]

print(highly_contentious)


# In[37]:


# Detailed view of the most contentious cases
most_contentious_cases = contentious_cases_all_factors[
    contentious_cases_all_factors[['PUBLIC SENTIMENT', 'RIDER SENTIMENT', 'TEAM SENTIMENT', 'MEDIA SENTIMENT']].apply(lambda x: (x == 'Negative').sum() > 1, axis=1)
]

print(most_contentious_cases)


# In[39]:


import matplotlib.pyplot as plt

# Visualize sentiment distribution for the highly contentious cases
sentiment_distribution = most_contentious_cases[['PUBLIC SENTIMENT', 'RIDER SENTIMENT', 'TEAM SENTIMENT', 'MEDIA SENTIMENT']].apply(pd.Series.value_counts)
sentiment_distribution.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.title('Sentiment Distribution for Highly Contentious Cases')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()


# In[41]:


# Analyze common patterns among contentious cases
pattern_analysis = most_contentious_cases.groupby(['Infringement Type', 'Aggravating factor', 'SANCTION']).size().reset_index(name='Counts')
print(pattern_analysis)


# In[43]:


# Compare with less contentious cases
less_contentious_cases = contentious_cases_all_factors[
    ~contentious_cases_all_factors['Infraction ID'].isin(most_contentious_cases['Infraction ID'])
]

print(less_contentious_cases)


# In[45]:


#Diagnostic analysis

import pandas as pd

# Group by infraction type, sanction severity, and stakeholder sentiment
grouped_data = df.groupby(['Infringement Type', 'Sanction Severity', 'PUBLIC SENTIMENT', 'RIDER SENTIMENT', 'TEAM SENTIMENT', 'MEDIA SENTIMENT']).size().reset_index(name='Counts')

# Focus on contentious cases with divergent sentiments
contentious_cases_divergent = df[
    (df['PUBLIC SENTIMENT'] == 'Negative') &
    ((df['RIDER SENTIMENT'] != 'Negative') | (df['TEAM SENTIMENT'] != 'Negative') | (df['MEDIA SENTIMENT'] != 'Negative'))
]

# Display the details of contentious cases with divergent sentiments
print(contentious_cases_divergent[['Infraction ID', 'Infringement Type', 'Aggravating factor', 'SANCTION', 'PUBLIC SENTIMENT', 'RIDER SENTIMENT', 'TEAM SENTIMENT', 'MEDIA SENTIMENT']])

# Compare these contentious cases with other less contentious cases
less_contentious_cases = df[
    (df['Sanction Severity'] == 'Severe') & 
    (df['Infraction ID'].isin(contentious_cases_divergent['Infraction ID']) == False)
]

# Display less contentious cases for comparison
print(less_contentious_cases[['Infraction ID', 'Infringement Type', 'Aggravating factor', 'SANCTION', 'PUBLIC SENTIMENT', 'RIDER SENTIMENT', 'TEAM SENTIMENT', 'MEDIA SENTIMENT']])


# In[48]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Example data (replace with your actual dataset)
# df = <your DataFrame>

# 1. Bar Chart: Sentiment Distribution Across Stakeholders for Contentious Cases
contentious_cases_sentiment = df[['Infraction ID', 'PUBLIC SENTIMENT', 'RIDER SENTIMENT', 'TEAM SENTIMENT', 'MEDIA SENTIMENT']].set_index('Infraction ID')
contentious_cases_sentiment.apply(pd.Series.value_counts).T.plot(kind='bar', stacked=True, figsize=(10, 7))
plt.title('Sentiment Distribution Across Stakeholders for Contentious Cases')
plt.ylabel('Number of Cases')
plt.show()

# 2. Heatmap: Correlation Between Aggravating Factors and Negative Sentiment
aggravating_factors = ['Caused Contact', 'Caused Crash', 'Disturbing Other(s)', 'Caused Danger/Safety Concern', 'Repeat Offence']
correlation_matrix = df[aggravating_factors + ['PUBLIC SENTIMENT', 'RIDER SENTIMENT', 'TEAM SENTIMENT', 'MEDIA SENTIMENT']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Aggravating Factors and Negative Sentiment')
plt.show()

# 3. Comparative Bar Chart: Contentious vs. Less Contentious Cases
contentious_counts = contentious_cases_sentiment.apply(lambda x: (x == 'Negative').sum(), axis=1)
less_contentious_counts = df[df['Infraction ID'].isin(contentious_cases_sentiment.index) == False]['Sanction Severity'].value_counts()

plt.figure(figsize=(10, 6))
plt.bar(contentious_counts.index, contentious_counts.values, label='Contentious Cases')
plt.bar(less_contentious_counts.index, less_contentious_counts.values, label='Less Contentious Cases', alpha=0.7)
plt.title('Comparison of Contentious vs. Less Contentious Cases')
plt.ylabel('Number of Cases')
plt.legend()
plt.show()

# 4. Line Graph: Sentiment Trend Over Time (if applicable)
# If there's a date column or similar time-based data
# df['Date'] = pd.to_datetime(df['DATE'])
# sentiment_trend = df.groupby('Date')[['PUBLIC SENTIMENT', 'RIDER SENTIMENT', 'TEAM SENTIMENT', 'MEDIA SENTIMENT']].mean()
# sentiment_trend.plot(figsize=(12, 8))
# plt.title('Sentiment Trend Over Time')
# plt.ylabel('Average Sentiment')
# plt.show()


# In[50]:


# Print the columns of the DataFrame to inspect the exact names
print(df.columns)


# In[52]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is your DataFrame

# List of aggravating factors we're interested in
aggravating_factors = ['Caused Contact', 'Caused Crash', 'Disturbing Other(s)', 'Caused Danger/Safety Concern', 'Repeat Offence']

# Create binary columns for each aggravating factor
for factor in aggravating_factors:
    df[factor] = df['Aggravating factor'].apply(lambda x: 1 if factor in str(x).split('\n') else 0)

# Convert sentiment columns to binary (1 for Negative, 0 for others)
for sentiment_column in ['PUBLIC SENTIMENT', 'RIDER SENTIMENT', 'TEAM SENTIMENT', 'MEDIA SENTIMENT']:
    df[sentiment_column] = df[sentiment_column].apply(lambda x: 1 if x == 'Negative' else 0)

# Select relevant columns for the heatmap
selected_columns = aggravating_factors + ['PUBLIC SENTIMENT', 'RIDER SENTIMENT', 'TEAM SENTIMENT', 'MEDIA SENTIMENT']

# Calculate the correlation matrix
correlation_matrix = df[selected_columns].corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Aggravating Factors and Negative Sentiment')
plt.show()


# In[54]:


import pandas as pd

# Assuming df is your DataFrame

# List of aggravating factors we're interested in
aggravating_factors = ['Caused Contact', 'Caused Crash', 'Disturbing Other(s)', 'Caused Danger/Safety Concern', 'Repeat Offence']

# Create binary columns for each aggravating factor
for factor in aggravating_factors:
    df[factor] = df['Aggravating factor'].apply(lambda x: 1 if factor in str(x).split('\n') else 0)

# Convert sentiment columns to binary (1 for Negative, 0 for others)
for sentiment_column in ['PUBLIC SENTIMENT', 'RIDER SENTIMENT', 'TEAM SENTIMENT', 'MEDIA SENTIMENT']:
    df[sentiment_column] = df[sentiment_column].apply(lambda x: 1 if x == 'Negative' else 0)

# Select relevant columns for analysis
selected_columns = aggravating_factors + ['PUBLIC SENTIMENT', 'RIDER SENTIMENT', 'TEAM SENTIMENT', 'MEDIA SENTIMENT']

# Calculate the correlation matrix
correlation_matrix = df[selected_columns].corr()

# Generate textual output of the correlations
textual_output = ""

for factor in aggravating_factors:
    textual_output += f"\nAggravating Factor: {factor}\n"
    for sentiment in ['PUBLIC SENTIMENT', 'RIDER SENTIMENT', 'TEAM SENTIMENT', 'MEDIA SENTIMENT']:
        correlation_value = correlation_matrix.loc[factor, sentiment]
        textual_output += f"  Correlation with {sentiment}: {correlation_value:.2f}\n"

# Display the textual output
print(textual_output)


# In[56]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is your DataFrame

# Focus on relevant columns for this analysis
sanction_data = df[['Sanction Severity', ' Sanction Classification', 'PUBLIC SENTIMENT', 'RIDER SENTIMENT', 'TEAM SENTIMENT', 'MEDIA SENTIMENT']]

# Convert sentiment columns to binary (1 for Negative, 0 for others)
for sentiment_column in ['PUBLIC SENTIMENT', 'RIDER SENTIMENT', 'TEAM SENTIMENT', 'MEDIA SENTIMENT']:
    sanction_data[sentiment_column] = sanction_data[sentiment_column].apply(lambda x: 1 if x == 'Negative' else 0)

# Analyze the distribution of sanctions by type and severity
sanction_type_severity = sanction_data.groupby([' Sanction Classification', 'Sanction Severity']).mean()

# Visualize the correlation between sanction type, severity, and sentiments
plt.figure(figsize=(12, 8))
sns.heatmap(sanction_type_severity, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Sanction Type, Severity, and Negative Sentiments')
plt.show()

# Summarize the findings in text
sanction_correlation_summary = ""
for sanction_type in sanction_type_severity.index.levels[0]:
    sanction_correlation_summary += f"\nSanction Type: {sanction_type}\n"
    for severity in sanction_type_severity.loc[sanction_type].index:
        sanction_correlation_summary += f"  Severity: {severity}\n"
        for sentiment in ['PUBLIC SENTIMENT', 'RIDER SENTIMENT', 'TEAM SENTIMENT', 'MEDIA SENTIMENT']:
            correlation_value = sanction_type_severity.loc[(sanction_type, severity), sentiment]
            sanction_correlation_summary += f"    Correlation with {sentiment}: {correlation_value:.2f}\n"

# Display the textual summary
print(sanction_correlation_summary)


# In[114]:


import pandas as pd

# Load your dataset
file_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Final Versions\Infraction classification-Final.xlsx'
df = pd.read_excel(file_path, sheet_name='Infringement Master', skiprows=8)

# Select the most contentious cases
contentious_cases = df[df['Infraction ID'].isin([
    '2022-CGP-004', '2022-CGP-006', '2022-C3-0013', 
    '2022-CGP-0016', '2022-CGP-0017', '2022-C3-0024'
])]

# Extract detailed information for each case
contentious_case_details = contentious_cases[
    ['Infraction ID', 'Infringement Type', 'Aggravating factor', 'SANCTION', 
     'PUBLIC SENTIMENT', 'RIDER SENTIMENT', 'TEAM SENTIMENT', 'MEDIA SENTIMENT']
]

# Display the contentious case details
print(contentious_case_details)

# Save these details to a new Excel file for reference
output_file_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Final Versions\contentious_case_studies.xlsx'
contentious_case_details.to_excel(output_file_path, index=False)


# In[140]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'heatmap_data' is your data prepared for the heatmap
plt.figure(figsize=(20, 15))  # Significantly increase the figure size

# Create the heatmap with rotated x-axis labels and smaller font size
sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="Blues", cbar=True,
            annot_kws={"size": 8},  # Smaller annotation font size
            linewidths=0.5, linecolor='gray')

plt.xticks(rotation=45, ha='right', fontsize=12)  # Rotate x-axis labels
plt.yticks(fontsize=12)  # Adjust y-axis label size
plt.title("Heatmap for Irresponsible Riding Penalty Consistency Study", fontsize=16)  # Add a title for context
# Set titles and labels

plt.xlabel("Sanction Types", fontsize=16, labelpad=10)
plt.ylabel("Irresponsible Riding Type with Aggravating Factor", fontsize=16, labelpad=10)


plt.tight_layout()  # Adjust layout to make space for labels
plt.show()


# In[128]:


print(df.columns)


# In[132]:


import pandas as pd

# Use the correct column names from your DataFrame
sub_class_column = 'Sub Classification'
agg_factor_column = 'Aggravating factor'
sanction_column = ' Sanction Classification'  # Note the leading space

# Filter cases for 'Irresponsible Riding'
irresponsible_riding_cases = df[df['Infringement Type'] == 'Irresponsible Riding']

# Summarize the findings by sub-classifications and aggravating factors
sub_class_agg_summary = irresponsible_riding_cases.groupby([sub_class_column, agg_factor_column, sanction_column]).size().reset_index(name='Count')

# Print the summarized findings
for index, row in sub_class_agg_summary.iterrows():
    sub_class = row[sub_class_column] if row[sub_class_column] else 'No specific sub-classification'
    agg_factor = row[agg_factor_column] if row[agg_factor_column] else 'No aggravating factor'
    sanction = row[sanction_column]
    count = row['Count']
    
    print(f"Sub-classification: {sub_class}")
    print(f"Aggravating factor: {agg_factor}")
    print(f"Sanction: {sanction}")
    print(f"Number of occurrences: {count}")
    print("------")





# In[142]:


import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'sub_class_agg_summary' DataFrame has columns ['Sub Classification', 'Aggravating factor', ' Sanction Classification', 'Count']

# Concatenate Sub Classification and Aggravating Factor to create a new column for y-axis
sub_class_agg_summary['Sub Classification + Aggravating Factor'] = sub_class_agg_summary['Sub Classification'] + ' - ' + sub_class_agg_summary['Aggravating factor']

# Plot the bubble chart
plt.figure(figsize=(12, 8))

# Use a scatter plot where the size of the bubble represents the frequency (Count)
plt.scatter(
    x=sub_class_agg_summary[' Sanction Classification'],
    y=sub_class_agg_summary['Sub Classification + Aggravating Factor'],
    s=sub_class_agg_summary['Count'] * 100,  # Multiply by 100 to scale bubble sizes
    alpha=0.6
)

# Add labels and title
plt.title("Sanctions for Irresponsible Riding: Sub-classifications and Aggravating Factors", fontsize=16)
plt.xlabel("Sanction Type", fontsize=14)
plt.ylabel("Irresponsible Riding Type with Aggravating Factor", fontsize=14)

# Improve readability
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Show the plot
plt.show()


# In[146]:


print(df.columns)


# In[148]:


import pandas as pd

# Assuming your data is already loaded into the DataFrame `df`

# Use the correct column names from your DataFrame
sub_class_column = 'Sub Classification'
agg_factor_column = 'Aggravating factor'
sanction_column = ' Sanction Classification'  # Note the leading space

# Filter cases for 'Irresponsible Riding'
irresponsible_riding_cases = df[df['Infringement Type'] == 'Irresponsible Riding']

# Summarize the findings by sub-classifications and aggravating factors
sub_class_agg_summary = irresponsible_riding_cases.groupby([sub_class_column, agg_factor_column, sanction_column]).size().reset_index(name='Count')

# Print the summarized findings
for index, row in sub_class_agg_summary.iterrows():
    sub_class = row[sub_class_column] if row[sub_class_column] else 'No specific sub-classification'
    agg_factor = row[agg_factor_column] if row[agg_factor_column] else 'No aggravating factor'
    sanction = row[sanction_column]
    count = row['Count']
    
    print(f"Sub-classification: {sub_class}")
    print(f"Aggravating factor: {agg_factor}")
    print(f"Sanction: {sanction}")
    print(f"Number of occurrences: {count}")
    print("------")


# In[174]:


import matplotlib.pyplot as plt
import pandas as pd

# Assuming your DataFrame 'df' is already loaded and contains the sentiment columns

# Calculate sentiment distribution for each stakeholder group
sentiment_distribution = {
    'Rider Sentiment': df['RIDER SENTIMENT'].value_counts(normalize=True) * 100,
    'Team Sentiment': df['TEAM SENTIMENT'].value_counts(normalize=True) * 100,
    'Media Sentiment': df['MEDIA SENTIMENT'].value_counts(normalize=True) * 100,
    'Public Sentiment': df['PUBLIC SENTIMENT'].value_counts(normalize=True) * 100
}

# Convert the dictionary to a DataFrame
sentiment_distribution_df = pd.DataFrame(sentiment_distribution)

# Plot a grouped bar chart
fig, ax = plt.subplots(figsize=(12, 8))

# Define the number of bars and their positions
bar_width = 0.2
positions = range(len(sentiment_distribution_df))

# Plotting each sentiment category for all stakeholder groups
bars1 = ax.bar([p - 1.5 * bar_width for p in positions], sentiment_distribution_df['Rider Sentiment'], 
               width=bar_width, label='Rider Sentiment', color='b')
bars2 = ax.bar([p - 0.5 * bar_width for p in positions], sentiment_distribution_df['Team Sentiment'], 
               width=bar_width, label='Team Sentiment', color='g')
bars3 = ax.bar([p + 0.5 * bar_width for p in positions], sentiment_distribution_df['Media Sentiment'], 
               width=bar_width, label='Media Sentiment', color='r')
bars4 = ax.bar([p + 1.5 * bar_width for p in positions], sentiment_distribution_df['Public Sentiment'], 
               width=bar_width, label='Public Sentiment', color='orange')

# Adding labels, title, and legend
ax.set_xlabel('Sentiment')
ax.set_ylabel('Percentage')
ax.set_title('Sentiment Distribution Across Stakeholder Groups')
ax.set_xticks(positions)
ax.set_xticklabels(sentiment_distribution_df.index)
ax.legend()

# Annotate the bars with the actual values
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', 
                    xy=(bar.get_x() + bar.get_width() / 2, height), 
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points", 
                    ha='center', va='bottom', 
                    fontsize=10)

plt.tight_layout()
plt.show()

# Print the distribution data
print(sentiment_distribution_df)


# In[158]:


import pandas as pd

# Load the categorized_sanctions dataset from the Excel file
file_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Final Versions\categorized_sanctions.xlsx'
categorized_sanctions_df = pd.read_excel(file_path)

# Display the first few rows to inspect the dataset
print(categorized_sanctions_df.head())


# In[164]:


# Check the exact column names
print(categorized_sanctions_df.columns)


# In[166]:


# Use the correct column name with the leading space
sanction_type_vs_sentiment = categorized_sanctions_df.groupby(' Sanction Classification').agg({
    'Rider Sentiment Score': 'mean',
    'Team Sentiment Score': 'mean',
    'Media Sentiment Score': 'mean'
}).reset_index()

# Display the results
print("\nSentiment Scores by Sanction Type:")
print(sanction_type_vs_sentiment)


# In[168]:


# Strip leading space from the column name and group by it
sanction_type_vs_sentiment = categorized_sanctions_df.groupby(categorized_sanctions_df[' Sanction Classification'].str.strip()).agg({
    'Rider Sentiment Score': 'mean',
    'Team Sentiment Score': 'mean',
    'Media Sentiment Score': 'mean'
}).reset_index()

# Display the results
print("\nSentiment Scores by Sanction Type:")
print(sanction_type_vs_sentiment)




# In[184]:


import pandas as pd

# Load the categorized_sanctions dataset from the Excel file
file_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Final Versions\categorized_sanctions.xlsx'
categorized_sanctions_df = pd.read_excel(file_path)

# Display the first few rows to inspect the dataset
print(categorized_sanctions_df.head())

# Display the column names to confirm the exact structure of the DataFrame
print(categorized_sanctions_df.columns)


# In[186]:


import matplotlib.pyplot as plt

# Step 1: Filter the data for negative sentiments
negative_sentiments_df = categorized_sanctions_df[
    (categorized_sanctions_df['RIDER SENTIMENT'] == 'Negative') |
    (categorized_sanctions_df['TEAM SENTIMENT'] == 'Negative') |
    (categorized_sanctions_df['MEDIA SENTIMENT'] == 'Negative') |
    (categorized_sanctions_df['PUBLIC SENTIMENT'] == 'Negative')
]

# Step 2: Group by 'Sanction Severity' and calculate the proportion of negative sentiments
severity_vs_negative = (negative_sentiments_df.groupby('Sanction Severity').size() /
                        categorized_sanctions_df.groupby('Sanction Severity').size()) * 100

# Step 3: Plot the results
severity_vs_negative.plot(kind='bar', color=['red', 'orange'], figsize=(10, 6))
plt.title('Proportion of Negative Sentiments by Sanction Severity')
plt.ylabel('Percentage of Negative Sentiments')
plt.xlabel('Sanction Severity')
plt.xticks(rotation=0)

# Annotate the bars with percentage values
for i, v in enumerate(severity_vs_negative):
    plt.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.show()

# Display the proportions as a DataFrame
severity_vs_negative_df = severity_vs_negative.reset_index(name='Negative Sentiment Percentage')
print(severity_vs_negative_df)


# In[200]:


import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Calculate sentiment proportions for each severity level
sentiment_severity = categorized_sanctions_df.groupby('Sanction Severity').agg({
    'RIDER SENTIMENT': lambda x: (x == 'Negative').sum() / len(x) * 100,
    'TEAM SENTIMENT': lambda x: (x == 'Negative').sum() / len(x) * 100,
    'MEDIA SENTIMENT': lambda x: (x == 'Negative').sum() / len(x) * 100,
    'PUBLIC SENTIMENT': lambda x: (x == 'Negative').sum() / len(x) * 100,
}).mean(axis=1).reset_index(name='Negative')

sentiment_severity['Neutral'] = categorized_sanctions_df.groupby('Sanction Severity').agg({
    'RIDER SENTIMENT': lambda x: (x == 'Neutral').sum() / len(x) * 100,
    'TEAM SENTIMENT': lambda x: (x == 'Neutral').sum() / len(x) * 100,
    'MEDIA SENTIMENT': lambda x: (x == 'Neutral').sum() / len(x) * 100,
    'PUBLIC SENTIMENT': lambda x: (x == 'Neutral').sum() / len(x) * 100,
}).mean(axis=1).values

sentiment_severity['Positive'] = categorized_sanctions_df.groupby('Sanction Severity').agg({
    'RIDER SENTIMENT': lambda x: (x == 'Positive').sum() / len(x) * 100,
    'TEAM SENTIMENT': lambda x: (x == 'Positive').sum() / len(x) * 100,
    'MEDIA SENTIMENT': lambda x: (x == 'Positive').sum() / len(x) * 100,
    'PUBLIC SENTIMENT': lambda x: (x == 'Positive').sum() / len(x) * 100,
}).mean(axis=1).values

# Step 2: Plot the results without annotations
sentiment_severity.set_index('Sanction Severity').plot(kind='bar', stacked=True, color=['red', 'yellow', 'green'], figsize=(10, 6))
plt.title('Sentiment Distribution by Sanction Severity')
plt.ylabel('Percentage')
plt.xlabel('Sanction Severity')
plt.xticks(rotation=0)
plt.legend(title='Sentiment')
plt.tight_layout()

plt.show()

# Display the sentiment proportions for further inspection
print(sentiment_severity)


# In[216]:


import pandas as pd

# Load the dataset
file_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Final Versions\Combined_Media_Sentiment.xlsx'
df = pd.read_excel(file_path)

# Relevant columns
relevant_columns = ['Infraction ID', 'Media Source', 'Media Comments Around the Decision', 'Automated BERT Sentiment', 'Final Sentiment']

# Filter the DataFrame to only include the relevant columns
df = df[relevant_columns]

# Count the number of media sources that are positive, negative, or neutral around each infraction
sentiment_count = df.groupby(['Infraction ID', 'Final Sentiment'])['Media Source'].nunique().unstack(fill_value=0)

# Add totals
sentiment_count['Total'] = sentiment_count.sum(axis=1)

# Display the results
print(sentiment_count)

# Optionally, to see the names of the media sources for each sentiment
media_sources_by_sentiment = df.groupby(['Infraction ID', 'Final Sentiment'])['Media Source'].apply(list)

# Display the results
print(media_sources_by_sentiment)


# In[222]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Final Versions\Combined_Media_Sentiment.xlsx'
df = pd.read_excel(file_path)

# Replace 'No Sentiment' and 'No sentiment' with 'Neutral'
df['Final Sentiment'] = df['Final Sentiment'].replace(['No Sentiment', 'No sentiment'], 'Neutral')

# Remove all rows with 'Neutral' sentiment
df = df[df['Final Sentiment'] != 'Neutral']

# Filter out rows where 'Media Source' or 'Media Comments Around the Decision' contains 'Press'
df = df[~df['Media Source'].str.contains('Press', case=False)]
df = df[~df['Media Comments Around the Decision'].str.contains('Press', case=False)]

# Count the number of media sources for each sentiment
sentiment_count = df.groupby(['Media Source', 'Final Sentiment']).size().unstack(fill_value=0)

# Separate the positive and negative sentiments
positive_sentiment = sentiment_count['Positive'].sort_values(ascending=False).head(5)
negative_sentiment = sentiment_count['Negative'].sort_values(ascending=False).head(5)

# Plot the top 5 positive media sources
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
positive_sentiment.plot(kind='bar', color='green')
plt.title('Top 5 Positive Media Sources')
plt.xlabel('Media Source')
plt.ylabel('Number of Positive Mentions')

# Plot the top 5 negative media sources
plt.subplot(1, 2, 2)
negative_sentiment.plot(kind='bar', color='red')
plt.title('Top 5 Negative Media Sources')
plt.xlabel('Media Source')
plt.ylabel('Number of Negative Mentions')

plt.tight_layout()
plt.show()


# In[224]:


import pandas as pd

# Load the dataset
file_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Final Versions\Combined_Media_Sentiment.xlsx'
df = pd.read_excel(file_path)

# Replace 'No Sentiment' and 'No sentiment' with 'Neutral'
df['Final Sentiment'] = df['Final Sentiment'].replace(['No Sentiment', 'No sentiment'], 'Neutral')

# Filter out rows where 'Media Source' or 'Media Comments Around the Decision' contains 'Press'
df = df[~df['Media Source'].str.contains('Press', case=False)]
df = df[~df['Media Comments Around the Decision'].str.contains('Press', case=False)]

# Filter for 'Neutral' sentiments
neutral_df = df[df['Final Sentiment'] == 'Neutral']

# Count the number of neutral mentions
neutral_count = neutral_df['Media Source'].nunique()

# Get the names of the media sources associated with neutral mentions
neutral_sources = neutral_df['Media Source'].unique()

# Print the results
print(f"Number of neutral mentions: {neutral_count}")
print("Names of media sources with neutral mentions:")
for source in neutral_sources:
    print(source)


# In[226]:


import pandas as pd

# Load the Excel file to inspect the columns
file_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Final Versions\Sentiment_Data_Riders_and_Rider_Other_Updated.xlsx'
df = pd.read_excel(file_path)

# Print the columns in the dataset
print("Columns in the dataset:")
print(df.columns.tolist())


# In[228]:


import pandas as pd

# Load the dataset
file_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Final Versions\Sentiment_Data_Riders_and_Rider_Other_Updated.xlsx'
df = pd.read_excel(file_path)

# Filter for 'Rider' in the 'Stake Holder Type' column
rider_df = df[df['Stake Holder Type'].str.contains('Rider', case=False)]

# Group by 'Reaction Type' and 'Sentiment', then count the unique riders
rider_sentiment_count = rider_df.groupby(['Reaction Type', 'Sentiment'])['Stake Holder Name'].nunique().unstack(fill_value=0)

# Print the results
print("Count of riders by Reaction Type and Sentiment:")
print(rider_sentiment_count)


# In[230]:


import pandas as pd

# Load the dataset
file_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Final Versions\Sentiment_Data_Riders_and_Rider_Other_Updated.xlsx'
df = pd.read_excel(file_path)

# Remove rows with 'No Response' in the 'Sentiment' column
df = df[df['Sentiment'] != 'No Response']

# Replace 'No Sentiment' with 'Neutral'
df['Sentiment'] = df['Sentiment'].replace('No Sentiment', 'Neutral')

# Filter for 'Rider' in the 'Stake Holder Type' column
rider_df = df[df['Stake Holder Type'].str.contains('Rider', case=False)]

# Group by 'Reaction Type' and 'Sentiment', then count the unique riders
rider_sentiment_count = rider_df.groupby(['Reaction Type', 'Sentiment'])['Stake Holder Name'].nunique().unstack(fill_value=0)

# Print the cleaned results
print("Cleaned count of riders by Reaction Type and Sentiment:")
print(rider_sentiment_count)


# In[232]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Final Versions\Sentiment_Data_Riders_and_Rider_Other_Updated.xlsx'
df = pd.read_excel(file_path)

# Remove rows with 'No Response' in the 'Sentiment' column
df = df[df['Sentiment'] != 'No Response']

# Replace 'No Sentiment' with 'Neutral'
df['Sentiment'] = df['Sentiment'].replace('No Sentiment', 'Neutral')

# Filter for 'Rider' in the 'Stake Holder Type' column
rider_df = df[df['Stake Holder Type'].str.contains('Rider', case=False)]

# Group by 'Reaction Type' and 'Sentiment', then aggregate the rider names
rider_sentiment_grouped = rider_df.groupby(['Reaction Type', 'Sentiment'])['Stake Holder Name'].apply(lambda x: ', '.join(x.unique())).reset_index()

# Count the number of riders by reaction type and sentiment
rider_count = rider_df.groupby(['Reaction Type', 'Sentiment'])['Stake Holder Name'].nunique().unstack(fill_value=0)

# Print the detailed rider names for each group
print("Detailed rider names by Reaction Type and Sentiment:")
print(rider_sentiment_grouped)

# Visualization
rider_count.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='tab20c')
plt.title('Rider Sentiment by Reaction Type')
plt.xlabel('Reaction Type')
plt.ylabel('Number of Riders')
plt.legend(title='Sentiment')
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()


# In[234]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Final Versions\Sentiment_Data_Riders_and_Rider_Other_Updated.xlsx'
df = pd.read_excel(file_path)

# Remove rows with 'No Response' in the 'Sentiment' column
df = df[df['Sentiment'] != 'No Response']

# Replace 'No Sentiment' with 'Neutral'
df['Sentiment'] = df['Sentiment'].replace('No Sentiment', 'Neutral')

# Filter for 'Rider' in the 'Stake Holder Type' column
rider_df = df[df['Stake Holder Type'].str.contains('Rider', case=False)]

# Further filter for only negative sentiments
negative_rider_df = rider_df[rider_df['Sentiment'] == 'Negative']

# Group by 'Reaction Type' and aggregate the rider names
negative_rider_grouped = negative_rider_df.groupby('Reaction Type')['Stake Holder Name'].apply(lambda x: ', '.join(x.unique())).reset_index()

# Count the number of negative riders by reaction type
negative_rider_count = negative_rider_df.groupby('Reaction Type')['Stake Holder Name'].nunique()

# Print the detailed rider names for each reaction type
print("Rider names with Negative Sentiment by Reaction Type:")
print(negative_rider_grouped)

# Visualization
negative_rider_count.plot(kind='bar', color='red', figsize=(10, 6))
plt.title('Number of Riders with Negative Sentiment by Reaction Type')
plt.xlabel('Reaction Type')
plt.ylabel('Number of Riders')
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()


# In[240]:


import pandas as pd

# Load the dataset
file_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Final Versions\Sentiment_Data_Riders_and_Rider_Other_Updated.xlsx'
df = pd.read_excel(file_path)

# Remove rows with 'No Response' in the 'Sentiment' column
df = df[df['Sentiment'] != 'No Response']

# Replace 'No Sentiment' with 'Neutral'
df['Sentiment'] = df['Sentiment'].replace('No Sentiment', 'Neutral')

# Filter for 'Rider' in the 'Stake Holder Type' column
rider_df = df[df['Stake Holder Type'].str.contains('Rider', case=False)]

# Further filter for only negative sentiments
negative_rider_df = rider_df[rider_df['Sentiment'] == 'Negative']

# Group by 'Reaction Type' and aggregate the rider names
negative_rider_grouped = negative_rider_df.groupby('Reaction Type')['Stake Holder Name'].apply(lambda x: ', '.join(x.unique())).reset_index()

# Count the number of negative riders by reaction type
negative_rider_count = negative_rider_df.groupby('Reaction Type')['Stake Holder Name'].nunique()

# Print the detailed rider names for each reaction type
print("Textual Summary: Riders with Negative Sentiment by Reaction Type\n")
for index, row in negative_rider_grouped.iterrows():
    reaction_type = row['Reaction Type']
    rider_names = row['Stake Holder Name']
    count = negative_rider_count[reaction_type]
    print(f"Reaction Type: {reaction_type}")
    print(f"Number of Riders with Negative Sentiment: {count}")
    print(f"Riders: {rider_names}")
    print("-" * 50)

# Example of how it would appear:
# Reaction Type: Media Article
# Number of Riders with Negative Sentiment: 7
# Riders: Rider1, Rider2, Rider3
# --------------------------------------------------


# In[242]:


import pandas as pd

# Load the dataset
file_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Final Versions\Sentiment_Data_Riders_and_Rider_Other_Updated.xlsx'
df = pd.read_excel(file_path)

# Remove rows with 'No Response' in the 'Sentiment' column
df = df[df['Sentiment'] != 'No Response']

# Replace 'No Sentiment' with 'Neutral'
df['Sentiment'] = df['Sentiment'].replace('No Sentiment', 'Neutral')

# Filter for 'Rider' in the 'Stake Holder Type' column
rider_df = df[df['Stake Holder Type'].str.contains('Rider', case=False)]

# Further filter for only negative sentiments
negative_rider_df = rider_df[rider_df['Sentiment'] == 'Negative']

# Count the number of negative sentiments per rider
negative_rider_count = negative_rider_df['Stake Holder Name'].value_counts()

# Print the names of the riders and their respective counts of negative sentiments
print("Riders and their respective counts of Negative Sentiments:\n")
print(negative_rider_count)


# In[244]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Final Versions\Sentiment_Data_Riders_and_Rider_Other_Updated.xlsx'
df = pd.read_excel(file_path)

# Remove rows with 'No Response' in the 'Sentiment' column
df = df[df['Sentiment'] != 'No Response']

# Replace 'No Sentiment' with 'Neutral'
df['Sentiment'] = df['Sentiment'].replace('No Sentiment', 'Neutral')

# Filter for 'Rider' in the 'Stake Holder Type' column
rider_df = df[df['Stake Holder Type'].str.contains('Rider', case=False)]

# Further filter for only negative sentiments
negative_rider_df = rider_df[rider_df['Sentiment'] == 'Negative']

# Count the number of negative sentiments per rider
negative_rider_count = negative_rider_df['Stake Holder Name'].value_counts()

# Plotting the results
plt.figure(figsize=(12, 8))
negative_rider_count.plot(kind='bar', color='red')
plt.title('Number of Negative Sentiments Expressed by Riders')
plt.xlabel('Rider Name')
plt.ylabel('Count of Negative Sentiments')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Show the plot
plt.show()


# In[17]:


import pandas as pd

# Load the dataset from Excel
file_path = r"C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Final Versions\Sentiment_Data_Team_Filtered.xlsx"
df = pd.read_excel(file_path)

# Convert "No Sentiment" to "Neutral"
df['Adjusted Sentiment'] = df['Adjusted Sentiment'].replace('No Sentiment', 'Neutral')

# Filter out rows where "Adjusted Sentiment" is "No Response"
df_filtered = df[df['Adjusted Sentiment'] != 'No Response']

# Calculate the frequency of each sentiment for each team
sentiment_counts = df_filtered.groupby(['Stake Holder Name', 'Adjusted Sentiment']).size().unstack(fill_value=0)

# Display the text output
print("Frequency of sentiments for each team:")
print(sentiment_counts)

# Display teams with each sentiment type
teams_with_negative = sentiment_counts[sentiment_counts.get('Negative', 0) > 0].index.tolist()
teams_with_neutral = sentiment_counts[sentiment_counts.get('Neutral', 0) > 0].index.tolist()

print("\nTeams with Negative Sentiments:")
print(teams_with_negative)

print("\nTeams with Neutral Sentiments:")
print(teams_with_neutral)


# In[15]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset from Excel
file_path = r"C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Final Versions\Sentiment_Data_Team_Filtered.xlsx"
df = pd.read_excel(file_path)

# Convert "No Sentiment" to "Neutral"
df['Adjusted Sentiment'] = df['Adjusted Sentiment'].replace('No Sentiment', 'Neutral')

# Filter out rows where "Adjusted Sentiment" is "No Response"
df_filtered = df[df['Adjusted Sentiment'] != 'No Response']

# Calculate the frequency of each sentiment for each team
sentiment_counts = df_filtered.groupby(['Stake Holder Name', 'Adjusted Sentiment']).size().unstack(fill_value=0)

# Visualization of Sentiment Distribution by Team
sentiment_counts.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.title('Sentiment Distribution by Team')
plt.xlabel('Team')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Sentiment')
plt.tight_layout()
plt.show()

# Visualization of Total Sentiments by Team
sentiment_counts.sum(axis=1).plot(kind='bar', figsize=(12, 8))
plt.title('Total Sentiments by Team')
plt.xlabel('Team')
plt.ylabel('Total Sentiments')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[ ]:




