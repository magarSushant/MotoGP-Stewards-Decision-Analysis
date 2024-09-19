#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

# Load the dataset
file_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Final Versions\categorized_sanctions.xlsx'
df = pd.read_excel(file_path)

# Display the first few rows of the dataset
print(df.head())


# In[5]:


df['Infringement Group'] = df['Infringement Type'] + ' - ' + df['Sub Classification']


# In[9]:


import pandas as pd

# Load the dataset
file_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Final Versions\categorized_sanctions.xlsx'
df = pd.read_excel(file_path)

# Display the first few rows to understand the data
print(df.head())

# Check data types
print(df.dtypes)

# Map 'Sanction Severity' from categorical to numeric
severity_mapping = {'Mild': 1, 'Severe': 2}
df['Sanction Severity Numeric'] = df['Sanction Severity'].map(severity_mapping)

# Combine Infringement Type and Sub Classification to create a new column
df['Infringement Group'] = df['Infringement Type'] + ' - ' + df['Sub Classification']

# Calculate average sanction severity for each infringement group
grouped_infringements = df.groupby('Infringement Group')['Sanction Severity Numeric'].mean().reset_index()

# Display the results
print(grouped_infringements)


# In[11]:


# Group by 'Infringement Group' to check consistency in severity
severity_by_infringement = df.groupby('Infringement Group')['Sanction Severity Numeric'].mean().reset_index()

# Display the results
print(severity_by_infringement)

# Group by 'PANEL MEMBERS' and 'Infringement Group' to check severity consistency by panel
severity_by_panel = df.groupby(['PANEL MEMBERS', 'Infringement Group'])['Sanction Severity Numeric'].mean().reset_index()

# Display the results
print(severity_by_panel)


# In[13]:


# Calculate standard deviation of sanction severity for each infringement group
std_dev_by_infringement = df.groupby('Infringement Group')['Sanction Severity Numeric'].std().reset_index()
std_dev_by_infringement.rename(columns={'Sanction Severity Numeric': 'Severity Std Dev'}, inplace=True)

# Display the results
print(std_dev_by_infringement)


# In[17]:


import pandas as pd

# Load the dataset
file_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Final Versions\categorized_sanctions.xlsx'
df = pd.read_excel(file_path)

# Display column names
print("Column Names in DataFrame:")
print(df.columns)


# In[25]:


import pandas as pd

# Load the dataset
file_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Final Versions\categorized_sanctions.xlsx'
df = pd.read_excel(file_path)

# Convert 'Sanction Severity' to numeric
df['Sanction Severity Numeric'] = pd.to_numeric(df['Sanction Severity'], errors='coerce')

# Check for missing values
missing_values = df[['Infringement Type', 'Sanction Severity Numeric']].isna().sum()
print("\nMissing values in key columns:")
print(missing_values)

# Option 1: Remove rows with missing 'Sanction Severity Numeric'
df_clean = df.dropna(subset=['Sanction Severity Numeric'])

# Option 2: Impute missing values with the median (or another statistic)
# median_severity = df['Sanction Severity Numeric'].median()
# df['Sanction Severity Numeric'].fillna(median_severity, inplace=True)

# Recalculate standard deviation of sanction severity for each infringement type
std_dev_by_infringement = df_clean.groupby('Infringement Type')['Sanction Severity Numeric'].std().reset_index()
std_dev_by_infringement.rename(columns={'Sanction Severity Numeric': 'Severity Std Dev'}, inplace=True)

# Recalculate mean severity for comparison
mean_severity_by_infringement = df_clean.groupby('Infringement Type')['Sanction Severity Numeric'].mean().reset_index()
mean_severity_by_infringement.rename(columns={'Sanction Severity Numeric': 'Average Severity'}, inplace=True)

# Merge mean and std deviation data
severity_stats = pd.merge(mean_severity_by_infringement, std_dev_by_infringement, on='Infringement Type')

# Display results
print("\nStandard Deviation and Mean Severity by Infringement Type:")
print(severity_stats)

# Calculate standard deviation by panel members and infringement type
std_dev_by_panel_member = df_clean.groupby(['PANEL MEMBERS', 'Infringement Type'])['Sanction Severity Numeric'].std().reset_index()
std_dev_by_panel_member.rename(columns={'Sanction Severity Numeric': 'Severity Std Dev'}, inplace=True)

# Display standard deviation by panel members
print("\nStandard Deviation by Panel Members and Infringement Type:")
print(std_dev_by_panel_member)


# In[27]:


import pandas as pd

# Load the dataset
file_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Final Versions\categorized_sanctions.xlsx'
df = pd.read_excel(file_path)

# Convert 'Sanction Severity' to numeric, forcing errors to NaN
df['Sanction Severity Numeric'] = pd.to_numeric(df['Sanction Severity'], errors='coerce')

# Check for missing values
missing_values = df[['Infringement Type', 'Sanction Severity Numeric']].isna().sum()
print("\nMissing values in key columns:")
print(missing_values)

# Inspect the first few rows after handling missing values
print("\nData after converting to numeric and handling missing values:")
print(df.head(10))

# Check how many rows remain after dropping NaNs
print(f"\nNumber of rows before handling missing values: {len(df)}")
df_clean = df.dropna(subset=['Sanction Severity Numeric'])
print(f"Number of rows after dropping NaNs: {len(df_clean)}")


# In[29]:


import matplotlib.pyplot as plt
import pandas as pd

# Group by panel members and sentiment categories
panel_sentiment_distribution = df.groupby(['PANEL MEMBERS', 'PUBLIC SENTIMENT', 'RIDER SENTIMENT', 'TEAM SENTIMENT', 'MEDIA SENTIMENT']).size().unstack(fill_value=0)

# Plot setup
fig, ax = plt.subplots(figsize=(12, 8))

# Plot each sentiment category
panel_sentiment_distribution.plot(kind='bar', stacked=True, ax=ax, cmap='tab10')

# Add labels and title
ax.set_title("Sentiment Distribution by Panel Group", fontsize=16)
ax.set_xlabel("Panel Group", fontsize=12)
ax.set_ylabel("Count", fontsize=12)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add legend
plt.legend(title="Sentiment", bbox_to_anchor=(1.05, 1), loc='upper left')

# Show plot with tight layout
plt.tight_layout()
plt.show()


# In[ ]:




