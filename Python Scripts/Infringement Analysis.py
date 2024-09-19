#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd

# Load the data from the specific sheet and skip the first 8 rows
file_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Final Versions\Infraction classification-Final.xlsx'
sheet_name = 'Infringement Master'

# Load the sheet, skipping the first 8 rows
df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=8)

# Display the first few rows of the dataframe
print(df.head())

# Check the shape of the dataset
print(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

# Check the data types of each column
print(df.dtypes)


# In[9]:


# Calculate the frequency of different types of infringements
infringement_frequency = df['Infringement Type'].value_counts()

# Display the results
print("Frequency of Different Types of Infringements:")
print(infringement_frequency)


# In[11]:


import matplotlib.pyplot as plt

# Plot bar chart for the frequency of different infringement types
infringement_frequency.plot(kind='bar', figsize=(10, 6))
plt.title('Frequency of Different Infringement Types')
plt.xlabel('Infringement Type')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()


# In[13]:


# Calculate the frequency of different Sanction Classifications
sanction_classification_frequency = df[' Sanction Classification'].value_counts()

# Display the results
print("Frequency of Different Sanction Classifications:")
print(sanction_classification_frequency)


# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the frequency of different sanction classifications
sanction_frequency = df[' Sanction Classification'].value_counts()

# Set up the plot size with more height to accommodate the labels
plt.figure(figsize=(12, 12))  # Increase the height

# Create a horizontal bar plot with more space between the bars and adjusted bar width
sns.barplot(x=sanction_frequency.values, 
            y=sanction_frequency.index, 
            palette="viridis", 
            errorbar=None,  # Replaces 'ci=None' as per the new API
            hue=sanction_frequency.index,  # To avoid the palette warning
            dodge=False,  # Ensures the bars are not dodged (stacked side by side)
            orient='h', 
            height=0.5)  # Adjusts the height of each bar

# Add annotations to each bar
for index, value in enumerate(sanction_frequency.values):
    plt.text(value + 0.1, index, str(value), va='center', ha='left', fontsize=10)

# Add title and labels
plt.title("Frequency of Different Sanction Classifications in 2022 MotoGP Season", fontsize=16)
plt.xlabel("Frequency", fontsize=12)
plt.ylabel("Sanction Classification", fontsize=12)

# Adjust spacing between the bars by setting the margin and bar height
plt.gca().margins(y=0.1)  # Adds more space between bars

# Show the plot
plt.show()






# In[17]:


# Filter out the '-' values
filtered_df = df[df['Aggravating factor'] != '-']

# Calculate the frequency of different aggravating factors
aggravating_factor_frequency = filtered_df['Aggravating factor'].value_counts()

# Print the frequency results
print("Frequency of Different Aggravating Factors (excluding null values):")
print(aggravating_factor_frequency)

# Set up the plot size
plt.figure(figsize=(12, 8))

# Create a horizontal bar plot with more space between the bars
sns.barplot(x=aggravating_factor_frequency.values, y=aggravating_factor_frequency.index, errorbar=None, palette="magma")

# Add annotations to each bar
for index, value in enumerate(aggravating_factor_frequency.values):
    plt.text(value + 0.1, index, str(value), va='center', ha='left', fontsize=10)

# Add title and labels
plt.title("Distribution of Aggravating Factors in MotoGP Infringements for 2022 Season", fontsize=16)

plt.xlabel("Frequency", fontsize=12)
plt.ylabel("Aggravating Factor", fontsize=12)

# Show the plot
plt.show()




# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns

# Filter the DataFrame for 'Irresponsible Riding' only
irresponsible_riding_df = df[df['Infringement Type'] == 'Irresponsible Riding']

# Filter out the '-' values within 'Irresponsible Riding'
filtered_irresponsible_df = irresponsible_riding_df[irresponsible_riding_df['Sub Classification'] != '-']

# Calculate the frequency of different sub-classifications within 'Irresponsible Riding'
sub_classification_frequency = filtered_irresponsible_df['Sub Classification'].value_counts()

# Print the frequency results
print("Frequency of Different Sub-Classifications (Irresponsible Riding only):")
print(sub_classification_frequency)

# Set up the plot size
plt.figure(figsize=(12, 8))

# Create a horizontal bar plot with more space between the bars
sns.barplot(x=sub_classification_frequency.values, y=sub_classification_frequency.index, errorbar=None, color="lightcoral")

# Add annotations to each bar
for index, value in enumerate(sub_classification_frequency.values):
    plt.text(value + 0.1, index, str(value), va='center', ha='left', fontsize=10)

# Add title and labels
plt.title("Distribution of Sub-Classifications in MotoGP Infringements for 'Irresponsible Riding' (2022 Season)", fontsize=16)

plt.xlabel("Frequency", fontsize=12)
plt.ylabel("Sub Classification", fontsize=12)

# Show the plot
plt.show()


# In[21]:


# Group by Infringement Type and Sanction Classification
summary_stats = df.groupby('Infringement Type')[' Sanction Classification'].describe()

# Print the summary statistics
print("Summary Statistics for Sanction Classifications per Infringement Type:")
print(summary_stats)


# In[97]:


import matplotlib.pyplot as plt
import seaborn as sns

# Use the original dataframe without filtering out the '-'
# Set up the plot size
plt.figure(figsize=(14, 8))

# Create a box plot for the sanctions across different infringement types
sns.boxplot(
    x='Infringement Type', 
    y=' Sanction Classification', 
    data=df,  # Use the original dataframe
    palette="coolwarm"
)

# Add title and labels
plt.title("Distribution of Sanctions for Different Infringement Types in 2022 MotoGP Season", fontsize=16)
plt.xlabel("Infringement Type", fontsize=12)
plt.ylabel("Sanction Classification", fontsize=12)

# Rotate the x labels if they overlap
plt.xticks(rotation=45)

# Show the plot
plt.show()


# In[23]:


import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table

# Your summary statistics as a DataFrame
summary_stats = pd.DataFrame({
    "Infringement Type": ["Flag Violations", "Irresponsible Riding", "Jump, Incorrect and Practice Starts", "Others", "Personal Aggression", "Technical Infringement"],
    "Count": [7, 40, 3, 1, 4, 4],
    "Unique Sanctions": [3, 11, 2, 1, 3, 2],
    "Most Common Sanction": ["Long Lap Penalties", "Long Lap Penalties", "Grid Penalties", "Grid Penalties", "Grid Penalties\nLong Lap Penalties in Race\nFine", "Performance Penalties"],
    "Frequency of Top Sanction": [5, 13, 2, 1, 2, 3]
})

# Set up the plot size
plt.figure(figsize=(10, 3))  # Adjust the size to fit the table

# Remove the axis
plt.axis('off')

# Create the table plot
tbl = table(plt.gca(), summary_stats, loc='center', cellLoc='center', colWidths=[0.15]*len(summary_stats.columns))

# Style the table
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.2)

# Show the table
plt.show()


# In[25]:


# Filter the data for "Irresponsible Riding"
irresponsible_riding_df = df[df['Infringement Type'] == 'Irresponsible Riding']

# Calculate the frequency of different sub-classifications within "Irresponsible Riding"
sub_classification_frequency = irresponsible_riding_df['Sub Classification'].value_counts()

# Print the frequency of sub-classifications
print("Frequency of Sub-Classifications within 'Irresponsible Riding':")
print(sub_classification_frequency)


# In[27]:


# Group the data by Sub Classification and Sanction Classification within "Irresponsible Riding"
sanction_distribution = irresponsible_riding_df.groupby(['Sub Classification', ' Sanction Classification']).size().unstack(fill_value=0)

# Print the sanction distribution table
print("Sanction Distribution for 'Irresponsible Riding' Sub-Classifications:")
print(sanction_distribution)


# In[29]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set up the plot size
plt.figure(figsize=(15, 8))

# Create a box plot for the sanctions across different sub-classifications within "Irresponsible Riding"
sns.boxplot(x='Sub Classification', y=' Sanction Classification', data=irresponsible_riding_df, palette="coolwarm")

# Add title and labels
plt.title("Distribution of Sanctions for 'Irresponsible Riding' Sub-Classifications in 2022 MotoGP Season", fontsize=16)
plt.xlabel("Sub Classification", fontsize=12)
plt.ylabel("Sanction Classification", fontsize=12)

# Show the plot
plt.xticks(rotation=45)
plt.show()


# In[33]:


# Print the column names of the DataFrame
print(df.columns)


# In[61]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Create a pivot table
pivot_table = pd.pivot_table(df, values='Infraction ID', 
                             index='Infringement Type', 
                             columns=' Sanction Classification', 
                             aggfunc='count', fill_value=0)

# Set up the plot size
plt.figure(figsize=(16, 10))

# Create the heatmap with 'PuBuGn' palette
sns.heatmap(pivot_table, annot=True, cmap="PuBuGn", fmt="d", linewidths=0.5, linecolor='gray')

# Rotate x-axis labels with more space and better alignment
plt.xticks(rotation=60, ha='right')

# Add title and labels
plt.title("Sanction Classification by Infringement Type in 2022 MotoGP", fontsize=14)
plt.xlabel("Sanction Classification", fontsize=12)
plt.ylabel("Infringement Type", fontsize=12)

# Adjust the layout
plt.tight_layout()

# Show the plot
plt.show()



# In[63]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Create a pivot table to summarize the count of each sanction by infringement type and sub-classification
pivot_table = pd.pivot_table(df, values='Infraction ID', 
                             index=['Infringement Type', 'Sub Classification'], 
                             columns=' Sanction Classification',  # Note the space before 'Sanction Classification'
                             aggfunc='count', fill_value=0)

# Set up the plot size
plt.figure(figsize=(14, 10))

# Create a heatmap
sns.heatmap(pivot_table, annot=True, fmt='d', cmap="viridis", linewidths=.5)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add title and labels
plt.title("Heatmap of Sanction Classifications by Infringement Type and Sub Classification", fontsize=16)
plt.xlabel("Sanction Classification", fontsize=12)
plt.ylabel("Infringement Type and Sub Classification", fontsize=12)

# Show the plot
plt.show()



# In[69]:


import pandas as pd
import matplotlib.pyplot as plt

# Create a pivot table to count the number of infractions per class and infringement type
df_pivot = pd.pivot_table(df, values='Infraction ID', 
                          index='CLASS', 
                          columns='Infringement Type', 
                          aggfunc='count', fill_value=0)

# Plot the stacked bar chart
df_pivot.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='viridis')

# Add title and labels
plt.title("Proportion of Infringement Types by Class in 2022 MotoGP Season", fontsize=16)
plt.xlabel("Class", fontsize=12)
plt.ylabel("Count of Infringements", fontsize=12)
plt.xticks(rotation=45)

# Show the plot
plt.show()


# In[71]:


# Generate a summary table with counts and percentages
summary_table = df_pivot.apply(lambda x: x / x.sum() * 100, axis=1).round(2)

# Print the summary table
print("Percentage Distribution of Infringement Types by Class:")
print(summary_table)


# In[73]:


import pandas as pd

# Create a pivot table to summarize the count of each sanction by class
pivot_table_class_sanction = pd.pivot_table(df, values='Infraction ID', 
                                            index='CLASS', 
                                            columns=' Sanction Classification', 
                                            aggfunc='count', fill_value=0)

# Calculate the percentage distribution of sanction classifications by class
percentage_distribution_class_sanction = pivot_table_class_sanction.div(pivot_table_class_sanction.sum(axis=1), axis=0) * 100

# Print the percentage distribution
print("Percentage Distribution of Sanction Classifications by Class:")
print(percentage_distribution_class_sanction)


# In[75]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set up the plot size
plt.figure(figsize=(12, 8))

# Create a heatmap of the percentage distribution
sns.heatmap(percentage_distribution_class_sanction, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5, linecolor='gray')

# Add title and labels
plt.title("Heatmap of Sanction Classification by Class in 2022 MotoGP", fontsize=16)
plt.xlabel("Sanction Classification", fontsize=12)
plt.ylabel("Class (MotoGP, Moto2, Moto3)", fontsize=12)

# Adjust the layout
plt.tight_layout()

# Show the plot
plt.show()


# In[83]:


# Get the top riders with the most infringements
top_infringement_riders = infringement_by_rider.sum(axis=1).sort_values(ascending=False).head(10)

# Get the top riders with the most sanctions
top_sanction_riders = sanction_by_rider.sum(axis=1).sort_values(ascending=False).head(10)

print("Top 10 Riders by Number of Infringements:")
print(top_infringement_riders)

print("\nTop 10 Riders by Number of Sanctions:")
print(top_sanction_riders)


# In[103]:


import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'df' is your DataFrame

# Calculate the total number of sanctions for each rider
rider_sanction_count = df['RIDER'].value_counts()

# Select the top 10 riders based on the total number of sanctions
top_10_riders = rider_sanction_count.nlargest(10)

# Set up the plot size
plt.figure(figsize=(12, 8))

# Create a bar plot
ax = top_10_riders.plot(kind='bar', color='skyblue')

# Add title and labels
plt.title("Top 10 Riders by Number of Times Sanctioned in 2022 MotoGP Season", fontsize=16)
plt.xlabel("Rider", fontsize=12)
plt.ylabel("Number of Sanctions", fontsize=12)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Annotate each bar with the number of sanctions
for i in ax.containers:
    ax.bar_label(i, label_type='edge', fontsize=10)

# Show the plot
plt.tight_layout()
plt.show()


# In[119]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is your DataFrame

# Drop duplicates based on 'Infraction ID' and 'CIRCUIT' to ensure each infraction is counted only once
df_unique_infractions = df.drop_duplicates(subset=['Infraction ID', 'CIRCUIT'])

# Group by circuit and count the total number of unique sanctions issued
circuit_sanctions = df_unique_infractions.groupby('CIRCUIT').size().sort_values(ascending=False).head(10)

# Set up the plot size
plt.figure(figsize=(10, 6))

# Create a bar plot for the total sanctions per circuit
sns.barplot(x=circuit_sanctions.values, y=circuit_sanctions.index, palette="muted")

# Annotate each bar with the count of sanctions issued
for index, value in enumerate(circuit_sanctions.values):
    plt.text(value + 0.2, index, str(value), va='center')

# Add title and labels
plt.title("Top 10 Circuits by Number of Sanctions Issued in 2022 MotoGP Season", fontsize=16)
plt.xlabel("Number of Sanctions Issued", fontsize=12)
plt.ylabel("Circuit", fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()


# In[121]:


# Generate textual summary for the top 10 circuits by the number of unique sanctions issued

# Create the textual summary
circuit_sanction_summary = circuit_sanctions.reset_index()
circuit_sanction_summary.columns = ['Circuit', 'Number of Unique Sanctions Issued']

# Print the textual summary
print("Top 10 Circuits by Number of Unique Sanctions Issued:")
for index, row in circuit_sanction_summary.iterrows():
    print(f"{index + 1}. {row['Circuit']}: {row['Number of Unique Sanctions Issued']} unique sanctions issued.")


# In[125]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Group the data by 'STAGE' and count the unique 'Infraction ID' to avoid multiple counts of the same infraction
stage_sanctions = df.groupby('STAGE')['Infraction ID'].nunique().sort_values(ascending=False).head(10)

# Print the summary statistics
print("Top Race Stages by Number of Unique Sanctions Issued:")
print(stage_sanctions)

# Set up the plot size
plt.figure(figsize=(10, 6))

# Create a bar plot for the total sanctions per stage
sns.barplot(x=stage_sanctions.values, y=stage_sanctions.index, palette="Blues_d")

# Annotate each bar with the count of sanctions
for index, value in enumerate(stage_sanctions.values):
    plt.text(value + 0.2, index, str(value), va='center')

# Add title and labels
plt.title("Total Sanctions by Stage of Event in 2022 MotoGP Season", fontsize=16)
plt.xlabel("Number of Unique Sanctions", fontsize=12)
plt.ylabel("Stage of Event", fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()


# In[127]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Ensure we work with unique sanctions based on Infraction ID
unique_sanctions = df.drop_duplicates(subset=['Infraction ID'])

# Group by team and count the total number of unique sanctions
team_sanctions = unique_sanctions.groupby('TEAM').size().sort_values(ascending=False).head(10)

# Set up the plot size
plt.figure(figsize=(12, 8))

# Create a bar plot for the total unique sanctions per team
sns.barplot(x=team_sanctions.values, y=team_sanctions.index, palette="muted")

# Annotate each bar with the count of sanctions
for index, value in enumerate(team_sanctions.values):
    plt.text(value + 0.2, index, str(value), va='center')

# Add title and labels
plt.title("Top 10 Teams by Number of  Sanctions Issued in 2022 MotoGP Season", fontsize=16)
plt.xlabel("Number of Unique Sanctions", fontsize=12)
plt.ylabel("Team", fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()


# In[135]:


import pandas as pd



# Inspect unique values in the 'HEARING?' column
unique_hearing_values = df['HEARING?'].unique()
print("Unique values in HEARING?:", unique_hearing_values)

# Inspect unique values in the 'RIGHT OF APPEAL?' column
unique_appeal_values = df['RIGHT OF APPEAL?'].unique()
print("Unique values in RIGHT OF APPEAL?:", unique_appeal_values)


# In[137]:


import pandas as pd


# Create a copy of the original DataFrame to clean the data
cleaned_df = df.copy()

# Cleaning the 'HEARING?' column
cleaned_df['HEARING?'] = cleaned_df['HEARING?'].str.strip()  # Remove leading/trailing spaces
cleaned_df['HEARING?'] = cleaned_df['HEARING?'].replace({
    'Y': 'Y',
    'Y ': 'Y',
    'N': 'N',
    'N ': 'N',
    'Unclear': 'N'  # Assuming 'Unclear' means no hearing was conducted
})

# Cleaning the 'RIGHT OF APPEAL?' column
cleaned_df['RIGHT OF APPEAL?'] = cleaned_df['RIGHT OF APPEAL?'].str.strip()  # Remove leading/trailing spaces
cleaned_df['RIGHT OF APPEAL?'] = cleaned_df['RIGHT OF APPEAL?'].replace({
    'Yes': 'Y',
    'Yes + hearing request': 'Y',
    'Y': 'Y',
    'Y + hearing request': 'Y',
    'No': 'N',
    'No - only hearing request': 'N',
    'N - only hearing request': 'N',
    'N  - only hearing request': 'N',
    'N': 'N'
})

# Handling missing values (if necessary)
cleaned_df['RIGHT OF APPEAL?'] = cleaned_df['RIGHT OF APPEAL?'].fillna('N')  # Assuming missing values mean no right of appeal

# Verify the cleaning process
print("Cleaned unique values in HEARING?:", cleaned_df['HEARING?'].unique())
print("Cleaned unique values in RIGHT OF APPEAL?:", cleaned_df['RIGHT OF APPEAL?'].unique())

# If you want to save the cleaned DataFrame to a new file
# cleaned_df.to_csv('cleaned_dataset.csv', index=False)


# In[165]:


import pandas as pd

# Assuming cleaned_df is your cleaned DataFrame

# Aggregate by Infraction ID to ensure unique sanctions
aggregated_df = cleaned_df.groupby('Infraction ID').first().reset_index()

# Now perform your hearing analysis on the aggregated data
hearing_by_infringement = aggregated_df.groupby(['Infringement Type', 'HEARING?']).size().unstack(fill_value=0)
print(hearing_by_infringement)


# In[169]:


import matplotlib.pyplot as plt
import numpy as np

# After aggregating by Infraction ID
# Aggregated Data: infringement_types, no_hearing, yes_hearing

# Ensure the data is aggregated correctly
infringement_types = hearing_by_infringement.index.tolist()
no_hearing = hearing_by_infringement['N'].tolist() if 'N' in hearing_by_infringement.columns else []
yes_hearing = hearing_by_infringement['Y'].tolist() if 'Y' in hearing_by_infringement.columns else []

# Create a grouped horizontal bar chart
plt.figure(figsize=(10, 6))
width = 0.35  # the width of the bars

# Plot horizontal bars
bars_no_hearing = plt.barh(np.arange(len(infringement_types)) - width/2, no_hearing, width, label='No Hearing', color='lightgray')
bars_yes_hearing = plt.barh(np.arange(len(infringement_types)) + width/2, yes_hearing, width, label='Hearing Conducted', color='steelblue')

# Add annotations (values) to the bars
for i, bar in enumerate(bars_no_hearing):
    plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, str(no_hearing[i]), va='center')

for i, bar in enumerate(bars_yes_hearing):
    plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, str(yes_hearing[i]), va='center')

# Labels and Title
plt.ylabel('Infringement Type')  # y-axis now reflects infringement types
plt.xlabel('Number of Cases')  # x-axis reflects the counts of cases
plt.title('Hearing Frequency by Infringement Type')
plt.yticks(np.arange(len(infringement_types)), infringement_types)
plt.legend()
plt.tight_layout()
plt.show()


# In[171]:


import pandas as pd

# Assuming cleaned_df is your cleaned DataFrame

# Step 1: Aggregate by Infraction ID to ensure unique sanctions
aggregated_df = cleaned_df.groupby('Infraction ID').first().reset_index()

# Step 2: Perform the hearing analysis on the aggregated data
hearing_by_infringement = aggregated_df.groupby(['Infringement Type', 'HEARING?']).size().unstack(fill_value=0)

# Step 3: Convert the hearing frequency data into a readable format for text output
text_output = hearing_by_infringement.reset_index()

# Step 4: Print the text output
print(text_output.to_string(index=False))


# In[173]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Assuming cleaned_df is your cleaned DataFrame

# Step 1: Aggregate by Infraction ID to ensure unique sanctions
aggregated_df = cleaned_df.groupby('Infraction ID').first().reset_index()

# Step 2: Perform the appeal analysis on the aggregated data
appeal_by_infringement = aggregated_df.groupby(['Infringement - Sub Classification', 'RIGHT OF APPEAL?']).size().unstack(fill_value=0)

# Display the text output for appeals by infringement type
print("Appeals by Infringement Type:")
print(appeal_by_infringement.to_string(index=False))

# Get the data ready for horizontal bar plotting
infringement_types = appeal_by_infringement.index.tolist()
no_appeal = appeal_by_infringement['N'].tolist() if 'N' in appeal_by_infringement.columns else []
yes_appeal = appeal_by_infringement['Y'].tolist() if 'Y' in appeal_by_infringement.columns else []

# Create a grouped horizontal bar chart
plt.figure(figsize=(10, 6))
width = 0.35  # the width of the bars

# Plot horizontal bars
bars_no_appeal = plt.barh(np.arange(len(infringement_types)) - width/2, no_appeal, width, label='No Appeal', color='lightgray')
bars_yes_appeal = plt.barh(np.arange(len(infringement_types)) + width/2, yes_appeal, width, label='Appeal Conducted', color='steelblue')

# Add annotations (values) to the bars
for i, bar in enumerate(bars_no_appeal):
    plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, str(no_appeal[i]), va='center')

for i, bar in enumerate(bars_yes_appeal):
    plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, str(yes_appeal[i]), va='center')

# Labels and Title
plt.ylabel('Infringement - Sub Classification')  # y-axis now reflects infringement types
plt.xlabel('Number of Cases')  # x-axis reflects the counts of cases
plt.title('Appeals by Infringement - Sub Classification')
plt.yticks(np.arange(len(infringement_types)), infringement_types)
plt.legend(title='Right of Appeal')
plt.tight_layout()
plt.show()


# In[175]:


import pandas as pd

# Assuming cleaned_df is your cleaned DataFrame

# Step 1: Aggregate by Infraction ID to ensure unique sanctions
aggregated_df = cleaned_df.groupby('Infraction ID').first().reset_index()

# Step 2: Perform the appeal analysis on the aggregated data
appeal_by_infringement = aggregated_df.groupby(['Infringement - Sub Classification', 'RIGHT OF APPEAL?']).size().unstack(fill_value=0)

# Step 3: Convert the appeal frequency data into a readable format for text output
appeal_text_output = appeal_by_infringement.reset_index()

# Step 4: Print the text output
print(appeal_text_output.to_string(index=False))



# In[177]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Aggregate by Infraction ID to ensure unique sanctions
aggregated_df = cleaned_df.groupby('Infraction ID').first().reset_index()

# Step 2: Create a cross-tabulation of hearing and appeal occurrences using the aggregated data
hearing_appeal_crosstab = pd.crosstab(aggregated_df['HEARING?'], aggregated_df['RIGHT OF APPEAL?'])

# Display the cross-tabulation
print("Cross-tabulation of Hearings and Appeals:")
print(hearing_appeal_crosstab)

# Step 3: Plot the cross-tabulation as a heatmap to visualize the relationship
sns.heatmap(hearing_appeal_crosstab, annot=True, fmt='d', cmap='Blues')
plt.title('Correlation Between Hearings and Appeals')
plt.xlabel('Right of Appeal?')
plt.ylabel('Hearing Conducted?')
plt.show()


# In[183]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Group by the unique combination of panel members and count the sanctions
sanctions_by_panel_group = df.groupby('PANEL MEMBERS').size().sort_values(ascending=False)

# Set up the plot size
plt.figure(figsize=(12, 8))

# Create a bar plot for the total number of sanctions associated with each panel group
sns.barplot(x=sanctions_by_panel_group.values, y=sanctions_by_panel_group.index, palette="muted")

# Annotate each bar with the count of sanctions
for index, value in enumerate(sanctions_by_panel_group.values):
    plt.text(value + 0.2, index, str(value), va='center')

# Add title and labels
plt.title("Sanctions Associated with Each Panel Group in 2022 MotoGP Season", fontsize=16)
plt.xlabel("Number of Sanctions", fontsize=12)
plt.ylabel("Panel Group", fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()


# In[185]:


# Print the textual output for the sanctions per panel group
print("Sanctions Associated with Each Panel Group:")
print(sanctions_by_panel_group)


# In[195]:


import matplotlib.pyplot as plt
import pandas as pd

# Group by panel members and public sentiment
panel_sentiment_distribution = df.groupby(['PANEL MEMBERS', 'PUBLIC SENTIMENT']).size().unstack(fill_value=0)

# Plot setup
fig, ax = plt.subplots(figsize=(12, 8))

# Plot public sentiment distribution
panel_sentiment_distribution.plot(kind='bar', stacked=True, ax=ax, cmap='viridis')

# Add labels and title
ax.set_title("Public Sentiment Distribution by Panel Group", fontsize=16)
ax.set_xlabel("Panel Group", fontsize=12)
ax.set_ylabel("Count", fontsize=12)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add legend
plt.legend(title="Public Sentiment", bbox_to_anchor=(1.05, 1), loc='upper left')

# Show plot with tight layout
plt.tight_layout()
plt.show()


# In[197]:


# Group by panel members and sentiment categories
panel_sentiment_distribution = df.groupby(['PANEL MEMBERS', 'PUBLIC SENTIMENT']).size().unstack(fill_value=0)

# Print the sentiment distribution
print("Sentiment Distribution by Panel Group:")
print(panel_sentiment_distribution)


# In[203]:





# In[ ]:




