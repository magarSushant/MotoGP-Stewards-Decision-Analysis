#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from urllib.parse import urlparse

# Load the Excel file
file_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Media Sentiment v001.xlsx'
xls = pd.ExcelFile(file_path)

# Initialize an empty DataFrame
combined_data = pd.DataFrame(columns=['Infraction', 'Media Source', 'Media Narrative', 'Rider Narrative'])

# Iterate through each sheet
for sheet_name in xls.sheet_names:
    # Read the sheet into a DataFrame
    df = pd.read_excel(xls, sheet_name=sheet_name)

    # Extract Media Source from the URL (Assuming it's in the first column)
    df['Media Source'] = df.iloc[:, 0].apply(lambda x: urlparse(x).netloc if pd.notnull(x) else '')

    # Combine the sheet name (infraction) with the DataFrame
    df['Infraction'] = sheet_name

    # Reorder the columns as required
    df = df[['Infraction', 'Media Source', df.columns[0], df.columns[1], df.columns[2]]]
    df.columns = ['Infraction', 'Media Source', 'Original Media Source', 'Media Narrative', 'Rider Narrative']

    # Drop the Original Media Source column as it's now redundant
    df = df.drop(columns=['Original Media Source'])

    # Append the processed DataFrame to the combined DataFrame
    combined_data = pd.concat([combined_data, df], ignore_index=True)

# Save the combined DataFrame to a new Excel file
output_file_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Combined_Media_Sentiment.xlsx'
combined_data.to_excel(output_file_path, index=False)

print(f"Combined data saved to {output_file_path}")



# In[ ]:




