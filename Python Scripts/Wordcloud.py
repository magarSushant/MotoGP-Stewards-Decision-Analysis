#!/usr/bin/env python
# coding: utf-8

# In[7]:


pip install wordcloud matplotlib


# In[9]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt


# In[15]:


import pandas as pd

# Define the file paths
riders_file_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Final Versions\Sentiment_Data_Riders_and_Rider_Other_Updated.xlsx'
teams_file_path = r'C:\Users\magar\OneDrive\Desktop\College Project\Moto GP\The Project\Work Data Set\For Analysis\Final Versions\Sentiment_Data_Team_Filtered.xlsx'

# Load the data
riders_data = pd.read_excel(riders_file_path)
teams_data = pd.read_excel(teams_file_path)


# In[17]:


# Filter for rider posts
riders_posts = riders_data[riders_data['Context (Post/Quotations in Article/Comments)'].str.contains("riders post", case=False, na=False)]

# Filter for team posts
teams_posts = teams_data[teams_data['Context (Post/Quotations in Article/Comments)'].str.contains("teams post", case=False, na=False)]


# In[19]:


# Combine all rider post text into a single string
rider_text = " ".join(post for post in riders_posts['Context (Post/Quotations in Article/Comments)'])

# Combine all team post text into a single string
team_text = " ".join(post for post in teams_posts['Context (Post/Quotations in Article/Comments)'])


# In[21]:


# Check the filtered rider posts
print(riders_posts.head())

# Check the filtered team posts
print(teams_posts.head())


# In[23]:


# Filter out rows with "No Response"
riders_filtered = riders_data[riders_data['Context (Post/Quotations in Article/Comments)'] != 'No Response']
teams_filtered = teams_data[teams_data['Context (Post/Quotations in Article/Comments)'] != 'No Response']


# In[33]:


# Combine text for riders
rider_text = " ".join(post for post in riders_filtered['Context (Post/Quotations in Article/Comments)'])

# Combine text for teams
team_text = " ".join(post for post in teams_filtered['Context (Post/Quotations in Article/Comments)'])

# Print to check the lengths
print(f"Length of rider text: {len(rider_text)}")
print(f"Length of team text: {len(team_text)}")


# In[37]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Combine the text from both riders and teams
combined_text = rider_text + " " + team_text

# Create a word cloud from the combined text
combined_wordcloud = WordCloud(width=1600, height=800, background_color='white').generate(combined_text)

# Plot the word cloud
plt.figure(figsize=(20, 10))  # Increase figure size for higher resolution
plt.imshow(combined_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Combined Word Cloud for Rider and Team Posts', fontsize=24, y=-0.1)  # Move the title down
plt.subplots_adjust(top=0.9)  # Adjust top margin to create space for the title
plt.show()


# In[29]:


# Combine text for riders and teams into one string
combined_text = " ".join(post for post in riders_filtered['Context (Post/Quotations in Article/Comments)']) + " " + \
                " ".join(post for post in teams_filtered['Context (Post/Quotations in Article/Comments)'])

# Print to check the length
print(f"Length of combined text: {len(combined_text)}")


# In[31]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Create a word cloud from the combined text
combined_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(combined_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Combined Word Cloud for Rider and Team Posts')
plt.show()


# In[33]:


# Define custom stopwords
custom_stopwords = set(["good", "today", "time", "quoted", "also", "first", "made", "now", "next", "really", "will", "still"])

# Optionally, add default stopwords from the WordCloud package
from wordcloud import STOPWORDS
custom_stopwords = custom_stopwords.union(STOPWORDS)


# In[36]:


# Combine text for riders and teams into one string
combined_text = " ".join(post for post in riders_filtered['Context (Post/Quotations in Article/Comments)']) + " " + \
                " ".join(post for post in teams_filtered['Context (Post/Quotations in Article/Comments)'])

# Print to check the length
print(f"Length of combined text: {len(combined_text)}")


# In[38]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Create a word cloud from the combined text, excluding the custom stopwords
combined_wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=custom_stopwords).generate(combined_text)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(combined_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Combined Word Cloud for Rider and Team Posts (Filtered)')
plt.show()


# In[3]:


# Create a word cloud from the combined text, excluding the custom stopwords
combined_wordcloud = WordCloud(width=1600, height=800, background_color='white', stopwords=custom_stopwords).generate(combined_text)

# Plot the word cloud
plt.figure(figsize=(20, 10))  # Increase figure size for higher resolution
plt.imshow(combined_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Combined Word Cloud for Rider and Team Posts (Filtered)', fontsize=24)
plt.show()


# In[ ]:




