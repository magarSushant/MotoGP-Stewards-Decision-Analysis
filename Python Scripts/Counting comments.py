#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

# Define the file path
file_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Sentiment Data-Public.xlsx'

# Load the Excel file
excel_file = pd.ExcelFile(file_path)

# Initialize a counter for the total number of comments
total_comments = 0

# Iterate through each sheet in the Excel file
for sheet_name in excel_file.sheet_names:
    # Load the sheet into a DataFrame
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # Check if the 'Comment' column exists in the sheet
    if 'Comment' in df.columns:
        # Count the number of non-null entries in the 'Comment' column
        sheet_comment_count = df['Comment'].dropna().shape[0]
        total_comments += sheet_comment_count

# Print the total number of comments across all sheets
print(f"Total number of comments (excluding blanks) across all sheets: {total_comments}")


# In[ ]:




