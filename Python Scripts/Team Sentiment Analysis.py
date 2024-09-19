#!/usr/bin/env python
# coding: utf-8

# In[115]:


import pandas as pd

# Define the file path
file_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Final Versions\Sentiment_Data_Team_Filtered.xlsx'

# Load the Excel file
df_team = pd.read_excel(file_path)


# Create a new column 'Adjusted Sentiment' based on 'Sentiment' column
df_team['Adjusted Sentiment'] = df_team['Sentiment'].replace({'No Response': 'Neutral', 'No Sentiment': 'Neutral'})

# Function to determine the overall sentiment for each Infraction ID
def determine_overall_sentiment(sentiments):
    sentiments_list = list(sentiments)
    if 'Negative' in sentiments_list:
        return 'Negative'
    elif 'Positive' in sentiments_list:
        return 'Positive'
    else:
        return 'Neutral'

# Group by 'Infraction ID' and determine the overall sentiment
df_overall = df_team.groupby('Infraction ID').agg({
    'Adjusted Sentiment': lambda x: determine_overall_sentiment(x),
    'DETAILS OF INFRACTION': 'first',
    'Stake Holder Type': 'first',
    'Stake Holder Name': 'first',
    'Reaction Type': 'first',
    'Context (Post/Quotations in Article/Comments)': 'first'
}).reset_index()

# Save the result to a new sheet called 'Final' in the same Excel file
with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
    df_overall.to_excel(writer, sheet_name='Final', index=False)

print(f"Overall sentiment per Infraction ID saved to the 'Final' sheet in: {file_path}")


# In[ ]:




