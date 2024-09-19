#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd

# Define the file paths
input_file_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Final Versions\Sentiment_Data_Riders_and_Rider_Other.xlsx'
output_file_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Final Versions\Sentiment_Data_Riders_and_Rider_Other_Updated.xlsx'

# Load the Excel file
df_riders = pd.read_excel(input_file_path)

# Drop the 'Source (Instagram URL)' and 'Unnamed: 8' columns
df_riders = df_riders.drop(columns=['Source (Instagram URL)', 'Unnamed: 8'])

# Create a new column 'Adjusted Sentiment' based on 'Sentiment' column
df_riders['Adjusted Sentiment'] = df_riders['Sentiment'].replace({'No Response': 'Neutral', 'No Sentiment': 'Neutral'})

# Save the modified DataFrame to a new Excel file
df_riders.to_excel(output_file_path, sheet_name='Riders and Rider Other', index=False)

print(f"Updated file saved to: {output_file_path}")


# In[19]:


import pandas as pd

# Define the file paths
input_file_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Final Versions\Sentiment_Data_Riders_and_Rider_Other_Updated.xlsx'
output_file_path = input_file_path  # Saving in the same file

# Load the Excel file
df_riders = pd.read_excel(input_file_path)

# Function to determine the final sentiment for each Infraction ID
def determine_final_sentiment(sentiments):
    sentiments_set = set(sentiments)
    if 'Negative' in sentiments_set and 'Positive' in sentiments_set:
        return 'Manual Decision'
    elif 'Negative' in sentiments_set:
        return 'Negative'
    elif 'Positive' in sentiments_set:
        return 'Positive'
    else:
        return 'Neutral'

# Group by 'Infraction ID' and determine the overall sentiment
df_final = df_riders.groupby('Infraction ID').agg({
    'Adjusted Sentiment': lambda x: determine_final_sentiment(x),
    'DETAILS OF INFRACTION': 'first',  # Keep the first occurrence of DETAILS OF INFRACTION
    'Stake Holder Type': 'first',      # Keep the first occurrence of Stake Holder Type
    'Stake Holder Name': 'first',      # Keep the first occurrence of Stake Holder Name
    'Reaction Type': 'first',          # Keep the first occurrence of Reaction Type
    'Context (Post/Quotations in Article/Comments)': 'first'
}).reset_index()

# Rename 'Adjusted Sentiment' to 'Final Sentiment'
df_final.rename(columns={'Adjusted Sentiment': 'Final Sentiment'}, inplace=True)

# Save the result to a new sheet called 'Final' in the same Excel file
with pd.ExcelWriter(output_file_path, engine='openpyxl', mode='a') as writer:
    df_final.to_excel(writer, sheet_name='Final', index=False)

print(f"Final sentiment per Infraction ID saved to the 'Final' sheet in: {output_file_path}")


# In[ ]:




