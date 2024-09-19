#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd


# In[40]:


# Full paths to your files
sentiment_file_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Sentiment Data-Public.xlsx'
consolidated_accounts_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Consolidated accounts list.xlsx'

# Load the Excel files using the full path
sentiment_file = pd.ExcelFile(sentiment_file_path)
consolidated_accounts = pd.read_excel(consolidated_accounts_path)






# In[41]:


print(consolidated_accounts.columns)



# In[43]:


#Verifying stakeholders comments in the dataset exported using export comment
# Ensure the Instagram Handle column is unique
consolidated_accounts_unique = consolidated_accounts.drop_duplicates(subset='Instagram Handle')

# Extract Instagram handles and their corresponding types
consolidated_handles = consolidated_accounts_unique['Instagram Handle'].str.lower()
consolidated_types = consolidated_accounts_unique.set_index('Instagram Handle')['Type']

# Create a DataFrame to store results
cross_verification_results = []

# Iterate over each sheet in the sentiment data file
for sheet_name in sentiment_file.sheet_names:
    sentiment_data = sentiment_file.parse(sheet_name)
    
    # Check if 'Username' column exists in the current sheet
    if 'Username' in sentiment_data.columns:
        # Cross-verify with the consolidated account handles
        matched_comments = sentiment_data[sentiment_data['Username'].str.lower().isin(consolidated_handles)].copy()
        
        if not matched_comments.empty:
            # Add the sheet name to the matched comments
            matched_comments['Sheet Name'] = sheet_name
            
            # Add the 'Type' column based on the matching Instagram handle
            matched_comments['Type'] = matched_comments['Username'].str.lower().map(consolidated_types)
            
            # Append the results to the final list
            cross_verification_results.append(matched_comments)
    else:
        print(f"'Username' column not found in sheet: {sheet_name}. Skipping this sheet.")

# Combine all matched results into a single DataFrame
if cross_verification_results:
    final_results = pd.concat(cross_verification_results, ignore_index=True)
    
    # Drop any unnecessary columns and sort by Sheet Name
    final_results = final_results.drop(columns=['Unnamed: 0', 'Unnamed: 1'], errors='ignore').sort_values(by='Sheet Name')
    
    # Save the final cleaned-up results to an Excel file
    output_path = "C:/Users/magar/OneDrive/Desktop/College Project/Moto GP/The Project/Work Data Set/For Analysis/Cleaned_Cross_Verification_Results_With_Type.xlsx"
    final_results.to_excel(output_path, index=False)
    
    print(f"Cross-verification completed and results saved to {output_path}")
else:
    print("No matches found or no sheets contained the 'Username' column.")


# In[45]:


#Verifying the results once again.
# Load the two files using the provided path
cross_verification_results = pd.read_excel(r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Cleaned_Cross_Verification_Results_With_Type.xlsx')
consolidated_accounts = pd.read_excel(r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Consolidated accounts list.xlsx')

# Convert both columns to lowercase for consistent comparison
usernames = cross_verification_results['Username'].str.lower()
instagram_handles = consolidated_accounts['Instagram Handle'].str.lower()

# Check which usernames are not present in the instagram_handles
missing_usernames = usernames[~usernames.isin(instagram_handles)]

# Display the missing usernames
if missing_usernames.empty:
    print("All usernames are present in the Instagram Handle column.")
else:
    print("The following usernames are not present in the Instagram Handle column:")
    print(missing_usernames)


# In[46]:


# Define file paths
original_data_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Sentiment Data-Public.xlsx'
cross_verified_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Cleaned_Cross_Verification_Results_With_Type.xlsx'

# Load the original public sentiment data
original_data = pd.ExcelFile(original_data_path)

# Load the cross-verified data
cross_verified_data = pd.read_excel(cross_verified_path)

# Create a dictionary to store cleaned sheets
cleaned_sheets = {}

# Iterate through each sheet in the original data
for sheet_name in original_data.sheet_names:
    sheet_data = original_data.parse(sheet_name)
    
    # Check the column names to ensure 'Username' exists
    print(f"Sheet: {sheet_name} - Columns: {sheet_data.columns}")
    
    if 'Username' in sheet_data.columns:
        # Remove rows from the original data that are present in the cross-verified data
        cleaned_sheet = sheet_data[~sheet_data['Username'].isin(cross_verified_data['Username'])]
        # Store the cleaned sheet in the dictionary
        cleaned_sheets[sheet_name] = cleaned_sheet
    else:
        print(f"'Username' column not found in sheet: {sheet_name}. Skipping this sheet.")

# Save the cleaned dataset with multiple sheets to a new Excel file
with pd.ExcelWriter(r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Cleaned_Sentiment_Data_Public_MultiSheet.xlsx') as writer:
    for sheet_name, cleaned_sheet in cleaned_sheets.items():
        cleaned_sheet.to_excel(writer, sheet_name=sheet_name, index=False)

print("Cleaning completed and results saved with multiple sheets.")


# In[47]:


# Define file paths
sentiment_data_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Sentiment Data-Public.xlsx'
cleaned_sentiment_data_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Cleaned_Sentiment_Data_Public_MultiSheet.xlsx'
cross_verified_data_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Cleaned_Cross_Verification_Results_With_Type.xlsx'

# Function to count total rows across all sheets in a file
def count_total_comments(file_path):
    excel_file = pd.ExcelFile(file_path)
    total_comments = 0
    for sheet_name in excel_file.sheet_names:
        sheet_data = excel_file.parse(sheet_name)
        total_comments += len(sheet_data) - 1  # Subtract 1 for the header row
    return total_comments

# Count comments in each file
total_comments_original = count_total_comments(sentiment_data_path)
total_comments_cleaned = count_total_comments(cleaned_sentiment_data_path)

# For the cross-verified data, it’s a single sheet, so we count directly
cross_verified_data = pd.read_excel(cross_verified_data_path)
total_comments_cross_verified = len(cross_verified_data) - 1  # Subtract 1 for the header row

# Display the results
print(f"Total number of comments in 'Sentiment Data-Public': {total_comments_original}")
print(f"Total number of comments in 'Cleaned_Sentiment_Data_Public_MultiSheet': {total_comments_cleaned}")
print(f"Total number of comments in 'Cleaned_Cross_Verification_Results_With_Type': {total_comments_cross_verified}")

# Check if the difference matches
difference = total_comments_original - total_comments_cleaned
print(f"Difference between original and cleaned: {difference}")

# Verify if the difference equals the cross-verified count
if difference == total_comments_cross_verified:
    print("The difference between the original and cleaned data matches the cross-verified data.")
else:
    print("There is a discrepancy between the original, cleaned, and cross-verified data counts.")


# In[ ]:




